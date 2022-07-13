import numpy as np
from utils.constants import DocumentInfo
from modules.vector_modules.window_func import ModifiedPert

#Define constants
SUB_VECTOR_COUNT = 16
REDUCE_EMBEDDING_DIMENSION=128


def position_weight_vector(vector):
    '''
    QUESTION: Isn't that the same thing as Zero centering (mean 0) vector?
    Util func to weight position of vector
    '''
    np.mean(vector) # calculates the mean of the array x
    vector-np.mean(vector) # this is euivalent to subtracting the mean of x from each value in x
    vector-=np.mean(vector)
    return vector

def normalize_vector(vector):
    '''
    Turns vector into unit vector
    Formula: v_unit = v/||v||
    Where ||v|| is the norm of v 
    '''
    #normalized_vector = vector / (np.linalg.norm(vector) + 1e-5)
    normalized_vector = vector / (np.sqrt(np.sum(vector**2)) + 1e-5) #Add to avoid divide by zero errors
    return normalized_vector


def get_average_sentence_embeddings(sentence_embeddings):
    '''
    Method One for building Document Vector
    '''
    if len(sentence_embeddings) == 0:
        raise ValueError("Sentence Embeddings cannot be empty")
    return np.mean(sentence_embeddings, axis=0)

def get_average_sentence_embeddings_with_bp(sentence_embeddings, bp_weighter, document):
    '''
    Method Two for building Document Vector
    '''
    if len(sentence_embeddings) == 0:
        raise ValueError("Sentence Embeddings cannot be empty")
    for idx in range(len(sentence_embeddings)):
        bp_score = bp_weighter.get_weighting_score(document[idx])
        sentence_embeddings[idx] *= bp_score
    return np.mean(sentence_embeddings, axis=0)

def get_document_sub_vector(index_number, sentence_embeddings, bp_weighter, window_weighter, document):
    '''
    Builds a Subvector for a document
    sentence_embeddings
    document is a list of sentences
    '''
    if len(sentence_embeddings) == 0:
        raise ValueError("Sentence Embeddings cannot be empty")

    result = np.zeros(sentence_embeddings[0].shape)

    for sent_idx, sentence_emb in enumerate(sentence_embeddings):
        pert_score = window_weighter.evaluate_distribution(sent_idx, index_number)
        bp_score = bp_weighter.get_weighting_score(document[sent_idx])
       
        sentence_vetcor = sentence_emb  * bp_score * pert_score #TODO: Fix issue with Pert Score. PERT SCORE IS a PROBLEM

        result = np.add(sentence_vetcor, result)
    return result


def concat_document_subvectors(sentence_embeddings, bp_weighter, sub_vec_count, document):
    '''
    Method Two for building Document Vector. Thompson's paper's method
    '''
    mp = ModifiedPert(0, sub_vec_count, len(document)) #Instantiate the weighting classes
    document_vector = None
    for i in range(sub_vec_count):
        sub_vector = get_document_sub_vector(i, sentence_embeddings, bp_weighter, mp, document)
        if document_vector is None:
            document_vector = sub_vector
        else:
            #document_vector += sub_vector
            document_vector = np.concatenate((document_vector, sub_vector)) #TODO: Uncomment when done debug

    return document_vector


#TODO: Pick better version betweeen this and last one
def get_document_vector_v2(sentence_embeddings, bp_weighter, pert_obj, document):
    '''
    Builds a Subvector for a document
    sentence_embeddings
    document is a list of sentences
    '''
    if len(sentence_embeddings) == 0:
        raise ValueError("Sentence Embeddings cannot be empty")

    
    sent_weights = np.array([bp_weighter.get_weighting_score(document[sent_idx]) for sent_idx in range(len(sentence_embeddings))])
    
    scaled_sent_embeds = np.multiply(sentence_embeddings.T, sent_weights).T

    #### BEGIN PERT PART: TODO: move to func, replace magic nums #####
    # equally space sentences
    sent_centers = np.linspace(0, 1, len(scaled_sent_embeds))

    # find weighting for each sentence, for each time slot
    sentence_loc_weights = np.zeros((len(sent_centers), 16))

    for sent_ii, p in enumerate(sent_centers):
        bank_idx = int(p * (100 - 1))  # find the nearest cached pert distribution
        sentence_loc_weights[sent_ii, :] = pert_obj.get_cache_idx(bank_idx)
    
    #### END PERT PART
        
    # make each chunk vector
    doc_chunk_vec = np.matmul(scaled_sent_embeds.T, sentence_loc_weights).T

    doc_vec = doc_chunk_vec.flatten()
    return doc_vec

    

def build_document_vector(document: list, url: str, raw_laser_sentence_embeddings,
                             bp_weighter, pca, pert_obj, sub_vec_count=SUB_VECTOR_COUNT,
                                                doc_vec_method='SENT_ORDER'):
    '''
    sub_vec_count is a hyperparameter that specifies the number of subvectors to compute
    raw_laser_sentence_embeddings are the laser generated sentence embeddings
    '''
    #NOTE: Assume documents are list of sentences. Post tolkenization
    #QUESTION: is shape 2048 or (16, 128)
    
    sentence_embeddings = raw_laser_sentence_embeddings
    sentence_embeddings = pca.transform(raw_laser_sentence_embeddings) #FIT PCA HERE
    
    try:
        #if not (len(document) == 0 and len(sentence_embeddings) == 1):
        assert len(document) == len(sentence_embeddings)
    except Exception as e:
        #import pdb
        #pdb.set_trace()
        print("exception for document build vector")

    if doc_vec_method == 'AVG':  #Avg approach
        document_vector = normalize_vector(get_average_sentence_embeddings(sentence_embeddings))
    elif doc_vec_method == 'AVG_BP': #Avg + BP Down Weighting Approach
        document_vector = normalize_vector(get_average_sentence_embeddings_with_bp(sentence_embeddings, bp_weighter, document))
    else: #Standard Approach (Doc vector builder)
        document_vector = normalize_vector(get_document_vector_v2(sentence_embeddings, bp_weighter, pert_obj, document))
        #document_vector = normalize_vector(concat_document_subvectors(sentence_embeddings, bp_weighter, sub_vec_count, document))

    doc = DocumentInfo(document_vector, url, sentence_embeddings, document)
    return doc

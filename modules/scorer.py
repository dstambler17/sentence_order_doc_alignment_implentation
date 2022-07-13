import time
import math
from numpy import dot
from numpy.linalg import norm
from utils.constants import DocumentInfo, CandidatePair
from modules.langId import LangIdentification
from vec_align.dp_utils import vecalign, print_alignments, make_doc_embedding

class CosineSimilarity(object):

    def __init__(self, metric='cosine',
                 smooth=0, ignore=None, threshold=0.1, batch_size=10000):
        self.name = "Cosine Distance Scorer"
        self.metric = metric
        #self.threshold = threshold
        #self.batch_size = batch_size

    def batched_pairwise_distances(self, X_csr, Y_csr):

        cos_sim_score = dot(X_csr, Y_csr)/(norm(Y_csr)*norm(Y_csr))

        return cos_sim_score

    def score(self, source_vector, target_vector, weighting=None, pool=None):
        #start = time.time()

        #sys.stderr.write(
        #    "Matrix extraction took {0:.5f} seconds\n".format(time.time() - start))

        #start = time.time()
        #del self.vector_extractor

        cos_sim_score = self.batched_pairwise_distances(source_vector, target_vector)

        #sys.stderr.write(
        #    "Scoring took {0:.5f} seconds\n".format(time.time() - start))
        return abs(cos_sim_score)
    

class DocPairReScorer():
    '''
    QUESTION: Is a lower score better here?
    Rescoring formula from Thompson paper
    1/|a(E, F)| Sum_{e,f elem of a(E, F)}( sim(e,f)p(Lang E| e)p(Lang F | f) ) 
    '''
    def _get_sentence_alignments(self, src_doc, tgt_doc, IS_DEBUG=False):
            #Vec align test params: 
        max_size_full_dp, costs_sample_size, num_samps_for_norm, search_buffer_size = 300, 20000, 100, 5

        del_percentile_frac = 0.2
        alignment_max_size = 8

        def make_alignment_types(max_alignment_size):
            # return list of all (n,m) where n+m <= this
            alignment_types = []
            for x in range(1, max_alignment_size):
                for y in range(1, max_alignment_size):
                    if x + y <= max_alignment_size:
                        alignment_types.append((x, y))
            return alignment_types

        final_alignment_types = make_alignment_types(alignment_max_size)
        width_over2 = math.ceil(alignment_max_size / 2.0) + search_buffer_size

        dictified_doc, sentence_embeddings, tokenized_document = src_doc.get_vec_align_info()
        dictified_doc_one, sentence_embeddings_one, tokenized_document_one = tgt_doc.get_vec_align_info()
        
        vecs0 = make_doc_embedding(dictified_doc, sentence_embeddings, tokenized_document, alignment_max_size)
        vecs1 = make_doc_embedding(dictified_doc_one, sentence_embeddings_one, tokenized_document_one, alignment_max_size)

        #TODO: Figure out what the input to vecalign should look like
        stack = vecalign(vecs0=vecs0,
                        vecs1=vecs1,
                        final_alignment_types=final_alignment_types,
                        del_percentile_frac=del_percentile_frac,
                        width_over2=width_over2,
                        max_size_full_dp=max_size_full_dp,
                        costs_sample_size=costs_sample_size,
                        num_samps_for_norm=num_samps_for_norm)
        
        if IS_DEBUG:
            print_alignments(stack[0]['final_alignments'], stack[0]['alignment_scores'])
        return stack[0]['final_alignments']

    def __init__(self, candidate_pair: CandidatePair):
        '''
        QUESTION, How does this formula punish insertions/deletions
        
        '''
        self.src_doc = candidate_pair.src_doc
        self.target_doc = candidate_pair.target_doc

        self.alignments = self._get_sentence_alignments(self.src_doc, self.target_doc)
        self.alignment_size = len(self.alignments) #TODO: Make sure insertion/deletion penalties are included

        lid = LangIdentification()

        self.src_lang_probs = lid.get_lang_id_probabilities_dict(self.src_doc.tokenized_document)
        self.target_lang_probs = lid.get_lang_id_probabilities_dict(self.target_doc.tokenized_document)

        self.src_sentence_vecs = self.src_doc.sentence_embeddings
        self.target_sentence_vecs = self.target_doc.sentence_embeddings
        
        self.distance_scorer = CosineSimilarity()
    
    def get_sentence_alignments(self):
        '''
        Returns Sentence Alignments
        '''
        return self.alignments

    
    def score(self, src_lang, target_lang):
        total_score = 0
        for ii_list, jj_list in self.alignments:
            if len(ii_list) > 0 and len(jj_list) > 0:
                for i in ii_list:
                    for j in jj_list:
                        src_lang_prob = self.src_lang_probs[i].get(src_lang, 0)
                        target_lang_prob = self.target_lang_probs[j].get(target_lang, 0)
                        sim_score = self.distance_scorer.score(self.src_sentence_vecs[i], self.target_sentence_vecs[j])
                        total_score += (src_lang_prob * target_lang_prob * sim_score)
        return total_score / self.alignment_size

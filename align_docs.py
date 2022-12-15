'''
Main File
'''
import argparse
from multiprocessing.sharedctypes import Value
import sys
import math
import json

import time
from pathlib import Path

import subprocess
import os


import numpy as np

from modules.scorer import CosineSimilarity, DocPairReScorer#, EnglishWordExtractor, _ngram_helper
from modules.matcher import ApproximateNearestNeighborSearch, competivie_matching_algorithm, match_based_on_mover_distance
from utils.common import function_timer, load_extracted, map_dic2list, \
                        flatten_2d_list, replace_newline, clean_doc_item, filter_empty_docs, tokenize_doc_to_sentence

from modules.build_document_vector import build_document_vector
from modules.vector_modules.boiler_plate_weighting import LIDFDownWeighting
from sklearn.decomposition import PCA

from modules.get_embeddings import load_embeddings, make_embedding_metadata
from modules.vector_modules.window_func import ModifiedPertV2

from sentence_mover_dist_modules.get_document_weights import get_document_weights


def fit_pca_reducer(embedding_list_src, embedding_list_tgt):
    '''
    Builds PCA Dim Reduction from sample of sentence embeddings
    in the webdomain
    '''
    pca = PCA(n_components=128)
    all_sent_embeds = np.vstack(embedding_list_src + embedding_list_tgt)

    pca_fit_data = all_sent_embeds[np.random.randint(all_sent_embeds.shape[0], size=len(all_sent_embeds) // 6), :]
    pca.fit(pca_fit_data)
    return pca

@function_timer
def rescore_candidates(candidate_list, src_lang, tgt_lang):
    '''
    Assign rescore to each candidate
    '''
    for cand in candidate_list:
        rs = DocPairReScorer(cand)
        rescore = rs.score(src_lang, tgt_lang)
        cand.rescore = rescore
        cand.sentence_alignments = rs.get_sentence_alignments()


@function_timer
def build_src_target_lists(text_list_tokenized, url_list, embedding_list,
                             down_weighter, pca, pert_obj, doc_vector_method, list_size=None):
    '''
    Build a list of source and targets
    '''
    if list_size == None: #set to length of tokenzied list
        list_size = len(text_list_tokenized)
    docs = [build_document_vector(text_list_tokenized[i], url_list[i],
                         embedding_list[i], down_weighter, pca, pert_obj, doc_vec_method=doc_vector_method) for i in range(list_size)] #list containing full info
    search_doc = np.vstack([doc.doc_vector for doc in docs]) #list containing doc vector to be used in KNN for finding candidates
    return search_doc, docs


def write_doc_alignment(outfile, aligned_documents):
    '''
    Write Scores, and urls for each aligned pair
    Also writes aligned sentences
    '''
    with open(outfile, 'w') as f:
        for _, match in enumerate(aligned_documents):
            score, rescore = match.get_scores()
            src_url, target_url = match.get_url_pairs()
            f.write("{0:.5f}\t{1:.5f}\t{2}\t{3}\n".format(score, rescore, src_url, target_url)) #For later (production) only write one score

def write_sent_alignment(outfile, aligned_documents):
    '''
    Writes the aligned sentence pairs for each doc
    '''
    if outfile is None:
        return
    with open(outfile, 'w') as f:
        for _, doc_pair in enumerate(aligned_documents):
            sentence_alignments = doc_pair.get_sentence_alignments()
            src_doc, tgt_doc = doc_pair.get_documents()
            
            src_doc, tgt_doc = src_doc.tokenized_document, tgt_doc.tokenized_document
            for x, y in sentence_alignments:
                if x and y:
                    f.write('[%s]:[%s]\n\n' % (src_doc[x[0]], tgt_doc[y[0]]))



def align_docs(args):
    '''
    NOTE: This implementation is language agnostic
    '''

    docs_english = load_extracted(args.english)
    docs_foreign = load_extracted(args.foreign)
    print(len(docs_english), len(docs_foreign), "INFO")
    
    #Get the non empty docs  #TODO: Potentially seperate the clean logic from empty docs
    docs_english = filter_empty_docs(docs_english) #IMPORTANT: FILTERS OUT HTML
    docs_foreign = filter_empty_docs(docs_foreign) #IMPORTANT: FILTERS OUT HTML
    print(len(docs_english), len(docs_foreign), "NUM NON EMPTY DOCS")
        
    obj_english = map_dic2list(docs_english)
    obj_foreign = map_dic2list(docs_foreign)
    
    english_text_list = obj_english['text']
    foreign_text_list = obj_foreign['text']
    
    #Handle Sentence embedding logic:
    
    embed_path_name = os.path.dirname(args.english)
    embed_path_name = '/'.join(embed_path_name.split('/')[:-1]) + '/embeddings'
    embed_folder_name = embed_path_name + '/' + args.english.split('/')[-1][:-6]

    lookup_dict = make_embedding_metadata(args.english)

    #TODO: replace with more flexible embedding metadata method
    if not lookup_dict and args.no_embed_write:
        with open(embed_folder_name + '/' + 'embedding_lookup.json', 'r') as f:
            lookup_dict = json.load(f)


    embedding_list_eng, lookup_dict = load_embeddings(english_text_list, obj_english['mapping'],
                                                        lookup_dict, args.lang_code_src, embed_folder_name, no_write=args.no_embed_write)
    embedding_list_for, lookup_dict = load_embeddings(foreign_text_list, obj_foreign['mapping'],
                                                        lookup_dict, args.lang_code_tgt, embed_folder_name, no_write=args.no_embed_write)

    #Tokenizes by sentences only    
    #english_text_list_tokenized = [[replace_newline(sent, " ") for sent in sent_tokenize(doc)] for doc in english_text_list]
    #foreign_text_list_tokenized = [[replace_newline(sent, " ") for sent in sent_tokenize(doc)] for doc in foreign_text_list]
    
    english_text_list_tokenized = [tokenize_doc_to_sentence(doc, args.lang_code_src) for doc in english_text_list]
    foreign_text_list_tokenized = [tokenize_doc_to_sentence(doc, args.lang_code_tgt) for doc in foreign_text_list]


    ##Create PCA dim reducer and USE LIDF, also pert object
    lidf_weighter = LIDFDownWeighting(english_text_list_tokenized + foreign_text_list_tokenized)
    pca = fit_pca_reducer(embedding_list_eng, embedding_list_for)
    pert_obj = ModifiedPertV2(None, None)


    #Get Doc Vectors and Doc Objects
    eng_doc_vecs, eng_docs =  build_src_target_lists(english_text_list_tokenized, obj_english['mapping'],
                                                     embedding_list_eng, lidf_weighter, pca, pert_obj, args.doc_vector_method)
    for_doc_vecs, for_docs = build_src_target_lists(foreign_text_list_tokenized, obj_foreign['mapping'],
                                                     embedding_list_for, lidf_weighter, pca, pert_obj, args.doc_vector_method)
    
    #In this part, build initial candidate pairs based on different approaches
    
    if args.first_scoring_method == 'approx_knn':
        #Get candidate list from Approx KNN
        vector_dimension_size = eng_doc_vecs.shape[1]
        searcher = ApproximateNearestNeighborSearch(vector_dimension_size)
        candidate_list = searcher.build_candidate_pair_list(eng_docs, for_docs, eng_doc_vecs, for_doc_vecs, K_VAL=args.k_val)
        print("Len cand list is %s" % len(candidate_list))
    elif args.first_scoring_method == 'sentence_mover_distance':
        #Get Document weights and update the document_object TODO: reduce dup logic below
        print('Using Distance Movers method')
            
        eng_weights = get_document_weights(english_text_list_tokenized, args.lang_code_src)
        for_weights = get_document_weights(foreign_text_list_tokenized, args.lang_code_tgt)
        if len(eng_weights) != len(eng_docs) or len(for_weights) != len(for_docs):
            raise ValueError("Weight lists and Doc Obj lists must have the same size but \
                             Got src_weights_len: %d, src_docs_len: %d, tgt_weights_len: %d, tgt_docs_len %d" 
                             % (len(eng_weights), len(eng_docs), len(for_weights), len(for_docs)))

        for idx, weight in enumerate(eng_weights):
            eng_docs[idx].weights = weight
        for idx, weight in enumerate(for_weights):
            for_docs[idx].weights = weight
        
        candidate_list = match_based_on_mover_distance(eng_docs, for_docs)
        print("Len cand list is %s" % len(candidate_list))
    else:
        candidate_list = []
        
    #Rescore and align docs
    rescore_candidates(candidate_list, 'en', 'fr')
    aligned_documents = competivie_matching_algorithm(candidate_list, use_rescore=args.use_rescore, method=args.first_scoring_method)
    print("Len aligned_documents is %s" % len(aligned_documents))

    #Write alignments
    write_doc_alignment(args.output_matches, aligned_documents)
    #write_sent_alignment(args.output_sentences, aligned_documents) TODO: Potentially remove call to this
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--english', help='path to the extracted English text', required=True)
    parser.add_argument(
        '--foreign', help='path to the translated foreign text', required=True)

    parser.add_argument('--output_matches', help='path to output sent alignments to', required=True)
    parser.add_argument('--output_sentences', help='path to output sent alignments to', required=True)
    
    parser.add_argument('--threshold', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=10000)

    parser.add_argument('--lang_code_src', help="Lang code for source lang", default='en')
    parser.add_argument('--lang_code_tgt', help="Lang code for target lang", default='fr')

    parser.add_argument('--doc_vector_method', help="Doc vector building method. Can be: AVG, AVG_BP, or SENT_ORDER", default='SENT_ORDER')
    parser.add_argument('--no_embed_write', help="Set to true if you don't want to write embeddings", action='store_true')
    
    parser.add_argument('--k_val', help="K Value for Approx K nearest enighbors search", type=int, required=True)
    parser.add_argument('--first_scoring_method', help="Decide which method to get initial scores", default='approx_knn',
                        choices=["approx_knn", 'sentence_mover_distance'])
    
    parser.add_argument('--use_rescore', help="Set to true if you wish to use rescore in competitvie matching", action='store_true')
    
    
    args = parser.parse_args()

    sys.stderr.write("threshold: {0}\n".format(args.threshold))
    sys.stderr.write("batch_size: {0}\n".format(args.batch_size))
    align_docs(args)

    


#!/usr/bin/env python

'''
Builds Docs from embeddings
'''

import argparse
import sys
import os

#import os
#sys.path.append(os.path.abspath('../../utils'))


from collections import defaultdict
from utils.common import load_extracted, get_doc_filename_from_url, Page


DATA_DIR_PATH = '/home/dstambl2/doc_alignment_implementations/data'



def build_src_tgt_mapping(matches_filepath, threshold, is_buck_aligner):
    map_e2f = {}
    map_f2e = {}
    with open(DATA_DIR_PATH + '/' + matches_filepath, 'r') as f_mapping:
        for line in f_mapping:
            if is_buck_aligner == 1:
                score, e, f = line.strip().split('\t')
            else:
                score, _, e, f = line.strip().split('\t')
            if float(score) < float(threshold):
                continue

            map_e2f[e] = f
            map_f2e[f] = e
    print(len(list(map_e2f.keys())), len(list(map_f2e.keys())), "COUNT OF MAPPING!!!!")
    return map_e2f, map_f2e

def write_doc_text(name, lang, texts):
    name = name.split('/')[-1]
    #TODO: sinhala_data/aligned_sentence_pairs to passed in variable
    path = '%s/sinhala_data/aligned_sentence_pairs/buck_baseline/%s-%s.temp_texts' % (DATA_DIR_PATH, name , lang) 
    sys.stdout.write("THIS IS THE PATH: %s " % path)
    with open(path, 'w') as f:
        f.writelines(texts)
    sys.stdout.write("LINES HAVE BEEN WRITTEN")

def print_docs(docs, matches_filepath, src_lang, tgt_lang, domain):
    '''
    Prints document
    '''
    not_found = []

    src_texts = []
    tgt_texts = []

    for k in docs:
        en_url, fr_url = k
        if 'en_text' not in docs[k]:
            not_found.append("not found(en): {0}".format(en_url))
            continue

        if 'fr_text' not in docs[k]:
            not_found.append("not found(fr): {0}".format(fr_url))
            continue

        en_text, fr_text = docs[k]['en_text'], docs[k]['fr_text']
        src_texts.append(en_text)
        tgt_texts.append(fr_text)
        
        
        with open ('%s/sinhala_data/aligned_sentence_pairs/buck_baseline/%s.matches-combined_docs.txt' % (DATA_DIR_PATH, domain), 'a') as f:
            f.write("{0}\t{1}\t{2}\t{3}\t\n".format(en_url, fr_url, en_text, fr_text))


    if len(not_found):
        sys.stderr.write("Number of documents without matches: {0}\n".format(len(not_found)))
        for n in not_found:
            sys.stderr.write("{0}\n".format(n))
    
     
    write_doc_text(matches_filepath.split('.matches')[0], src_lang, src_texts)
    write_doc_text(matches_filepath.split('.matches')[0], tgt_lang, tgt_texts)

def load_docs(matches_filepath, threshold, doc_text_folder, src_lang, tgt_lang, is_buck_aligner):
      
    map_e2f, map_f2e = build_src_tgt_mapping(matches_filepath, threshold, is_buck_aligner)
    docs = defaultdict(dict) 

    if not map_e2f or not map_f2e:
        return docs
    
    #print(list(map_e2f.keys())[0], map_e2f[list(map_e2f.keys())[0]])
    
    #file_name_src = get_doc_filename_from_url(list(map_e2f.keys())[0], src_lang) NOTE: These two lines are for debug
    #file_name_tgt = get_doc_filename_from_url(list(map_e2f.keys())[0], tgt_lang)

    #LOAD MANY MANY DOCS WITH FOR LOOP
    doc_dict = {}
    for doc_path in os.listdir('%s/%s' % (DATA_DIR_PATH, doc_text_folder)):
        '''file_name_src = get_doc_filename_from_url(src_url, src_lang)
        file_name_tgt = get_doc_filename_from_url(tgt_url, tgt_lang)
        
        document_src = load_extracted(DATA_DIR_PATH + '/' + doc_text_folder + '/' + file_name_src)
        document_tgt = load_extracted(DATA_DIR_PATH + '/' + doc_text_folder + '/' + file_name_tgt)
        
        doc_dict.update({**document_src, **document_tgt})
        '''
        full_doc_path = '%s/%s/%s' % (DATA_DIR_PATH, doc_text_folder, doc_path)
        doc = load_extracted(full_doc_path)
        doc_dict.update(doc)
    

    for url, line in doc_dict.items():
        if url in map_e2f:
            key = (url, map_e2f[url])
            docs[key]['en_text'] = eval(line).text

        elif url in map_f2e:
            key = (map_f2e[url], url)
            docs[key]['fr_text'] = eval(line).text

    return docs 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--matches_file', help='path to the file with matched documents in base data dir', required=True)
    parser.add_argument('-d', '--doc_text_folder', help='Dataset folder that contains all data. Ex: wmt16/processed', required=True)
    parser.add_argument('-t', '--threshold', help='documents with lower match score will be skipped', default=0.1, type=float, required=False)
    parser.add_argument('-e', '--src_lang', help='src langs', default='en', type=str, required=False)
    parser.add_argument('-f', '--target_lang', help='target langs', default='fr', type=str, required=False)
    parser.add_argument('-b', '--is_buck_aligner', help='buck aligner', default='1', required=False)

    args = parser.parse_args()

    docs = load_docs(args.matches_file, args.threshold, args.doc_text_folder, args.src_lang, args.target_lang, int(args.is_buck_aligner))
    domain = args.matches_file.split('.matches')[0].split('/')[-1]
    print_docs(docs, args.matches_file, args.src_lang, args.target_lang, domain)

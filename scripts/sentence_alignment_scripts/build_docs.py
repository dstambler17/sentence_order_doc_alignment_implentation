#!/usr/bin/env python

'''
Builds Docs from embeddings
'''

import argparse
import sys

from collections import defaultdict
from utils.common import load_extracted, get_doc_filename_from_url, Page


DATA_DIR_PATH = '/home/dstambl2/doc_alignment_implementations/data/'



def build_src_tgt_mapping(matches_filepath, threshold):
    map_e2f = {}
    map_f2e = {}

    with open(DATA_DIR_PATH + matches_filepath, 'r') as f_mapping:
        for line in f_mapping:
            score, _, e, f = line.strip().split('\t')
            if float(score) < float(threshold):
                continue

            map_e2f[e] = f
            map_f2e[f] = e
    
    return map_e2f, map_f2e

def write_doc_text(name, lang, texts):
    name = name.split('/')[-1]
    path = '%swmt16_test/aligned_sentences/%s-%s.temp_texts' % (DATA_DIR_PATH, name , lang)
    with open(path, 'w') as f:
        f.writelines(texts)


def print_docs(docs, matches_filepath, src_lang, tgt_lang):
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
        print("{0}\t{1}\t{2}\t{3}\t".format(en_url, fr_url, en_text, fr_text))


    if len(not_found):
        sys.stderr.write("Number of documents without matches: {0}\n".format(len(not_found)))
        for n in not_found:
            sys.stderr.write("{0}\n".format(n))
    
    write_doc_text(matches_filepath.split('.matches')[0], src_lang, src_texts)
    write_doc_text(matches_filepath.split('.matches')[0], tgt_lang, tgt_texts)

def load_docs(matches_filepath, threshold, doc_text_folder, src_lang, tgt_lang):
      
    map_e2f, map_f2e = build_src_tgt_mapping(matches_filepath, threshold)
    docs = defaultdict(dict) 

    if not map_e2f or not map_f2e:
        return docs
    
    file_name_src = get_doc_filename_from_url(list(map_e2f.keys())[0], src_lang)
    file_name_tgt = get_doc_filename_from_url(list(map_e2f.keys())[0], tgt_lang)

    document_src = load_extracted(DATA_DIR_PATH + doc_text_folder + '/' + file_name_src)
    document_tgt = load_extracted(DATA_DIR_PATH + doc_text_folder + '/' + file_name_tgt)
    
    doc_dict = {**document_src, **document_tgt}

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

    args = parser.parse_args()

    docs = load_docs(args.matches_file, args.threshold, args.doc_text_folder, args.src_lang, args.target_lang)
    print_docs(docs, args.matches_file, args.src_lang, args.target_lang)

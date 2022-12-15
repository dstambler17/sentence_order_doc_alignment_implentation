# Build embeddings in parallel way
# Note, might need to replace with C++ implementation
# Similar to how its done in script
# Builds Document Embeddings
import numpy as np
import subprocess
import os
import time
from pathlib import Path

import argparse
import sys
import math
import json

from threading import Lock
#from utils.worker_manager import WorkerManager, MultithreadParams

import sys
sys.path.append('/home/dstambl2/doc_alignment_implementations/thompson_2021_doc_align')

from utils.common import function_timer, load_extracted, map_dic2list, clean_doc_item
from modules.get_embeddings import make_embedding_metadata, write_embeddings, WriteEmbeddingsParams



def save_embeddings(text_list, url_list, lookup_dict, lang_code, path, domain):
    '''
    NOTE: We assume that text_list[i]
    corresponds w/ url_list[i]
    '''
    if len(text_list) != len(url_list):
        raise ValueError("Text list and url list must be of same length")
    
    
    new_embeddings = 0
    lock = Lock()

    def write_embeddings_handler(write_embed_params : WriteEmbeddingsParams):
        embed_file_path, doc_text, token_lang, url = write_embed_params.get_params()
        write_embeddings(embed_file_path, doc_text, token_lang)
        
        nonlocal new_embeddings
        #nonlocal lock
        lookup_dict[url] = embed_file_path
        new_embeddings += 1
        

    #Set up worker manager
    #worker_manager = WorkerManager(write_embeddings_multithread_handler, 20)
    #worker_manager.start()

    for ii in range(len(text_list)):
        url, doc_item = url_list[ii], text_list[ii]
        
        #IMPORTANT: Extract text only from "Page" item
        doc_text = clean_doc_item(doc_item)

        if url not in lookup_dict:
            embed_file_path = "%s/embeddings/%s/%s.%s.emb" % (path, domain, ii,lang_code)
            
            write_embed_params = WriteEmbeddingsParams(embed_file_path, doc_text, lang_code, url)
            write_embeddings_handler(write_embed_params)
            #worker_manager.put()
            

    #worker_manager.stop()

    if new_embeddings >= 1: #Save new copy of updated json lookup dict
        with open(path + "/embeddings/" + domain + "/embedding_lookup.json", 'w') as f:
            json.dump(lookup_dict, f)
    return lookup_dict



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang_code_src', help="Lang code for Source Lang", default='en')
    parser.add_argument('--lang_code_tgt', help="Lang code for Target Lang", default='fr')

    parser.add_argument(
        '--src_file', help='path to the extracted English text', required=True)
    parser.add_argument(
        '--tgt_file', help='path to the translated foreign text', required=True)
    parser.add_argument('--domain_name', help='domain name. Will create an embedding sub dir', required=True)
    
    parser.add_argument('--dirname', help='data durname extension. Example: cc_aligned_si_data', required=True)

    args = parser.parse_args()

    
    lang_code_english = args.lang_code_src
    lang_code_foreign = args.lang_code_tgt

    doc_lang_src = load_extracted(args.src_file)
    doc_lang_tgt = load_extracted(args.tgt_file)
    print(len(doc_lang_src), len(doc_lang_tgt), "INFO")

    obj_lang_src = map_dic2list(doc_lang_src)
    obj_lang_tgt = map_dic2list(doc_lang_tgt)
    
    lang_src_text_list = obj_lang_src['text']
    lang_tgt_text_list = obj_lang_tgt['text']

    #Define dur name base
    DATA_DUR_NAME_BASE = '/home/dstambl2/doc_alignment_implementations/data/'
    dirname = "%s/%s" % (DATA_DUR_NAME_BASE, args.dirname) #os.path.dirname(args.lang_tgt) 

    #Handle Sentence embedding logic:
    lookup_dict = make_embedding_metadata('',
             DEFAULT_DIR=dirname,
             EMBED_PATH="/embeddings/%s/embedding_lookup.json" % args.domain_name)
    
    
    lookup_dict = save_embeddings(lang_src_text_list, obj_lang_src['mapping'],
                                                        lookup_dict, args.lang_code_src, dirname, args.domain_name)
    lookup_dict = save_embeddings(lang_tgt_text_list, obj_lang_tgt['mapping'],
                                                        lookup_dict, args.lang_code_tgt, dirname, args.domain_name)



import os
import json
import numpy as np
from collections import defaultdict, namedtuple
from random import shuffle, randint, sample, uniform, choice

from doc_align_classifier.consts import *

from utils.common import load_extracted, \
    filter_empty_docs, regex_extractor_helper
from modules.get_embeddings import read_in_embeddings
from modules.vector_modules.boiler_plate_weighting import LIDFDownWeighting
from modules.vector_modules.window_func import ModifiedPertV2
from sklearn.model_selection import train_test_split


def split_train_val_test_sets():
    embed_chunk_paths = []
    for subdir, dirs, files in os.walk(BASE_EMBED_DIR):
        if subdir != BASE_EMBED_DIR:
            embed_chunk_paths.append(subdir)
            
    embed_chunk_paths = sorted(embed_chunk_paths, key= lambda x: len(x))

    #For now split into 0.7/0.1/0.2 split
    #Split into train, test and val sets, remove last one for split since it will go into train set
    train_ind, test_ind = train_test_split(list(range(len(embed_chunk_paths[:-1]))), test_size=0.2, random_state=1)
    train_ind, val_ind = train_test_split(train_ind, test_size=0.125, random_state=1) # 0.125 x 0.8 = 0.1

    train_ind.append(len(embed_chunk_paths) -1) #Add last imbalenced idx to train set

    print(len(train_ind), len(val_ind), len(test_ind))

    chunks_paths_train = [embed_chunk_paths[t] for t in train_ind]
    chunks_paths_val = [embed_chunk_paths[t] for t in val_ind]
    chunks_paths_test = [embed_chunk_paths[t] for t in test_ind]
    
    return chunks_paths_train, chunks_paths_val, chunks_paths_test



''' The following two funcs are only for getting sent embeds '''
def get_base_embedding(url, embedding_file_path, lang_code):

    chunk = regex_extractor_helper(CHUNK_RE, embedding_file_path)
    doc_path = '%s/%s.%s.gz' % (BASE_PROCESSED_PATH, chunk, lang_code)
    url_doc_dict = filter_empty_docs(load_extracted(doc_path))
    if url not in url_doc_dict:
        print("missing url in get_base_embedding %s for doc path %s" % (url, doc_path))
        #return noise
        return np.random.random((1,1024))
    doc_text = url_doc_dict[url]
        
    _, sent_embeds = read_in_embeddings(doc_text, embedding_file_path, lang_code)
    
    return sent_embeds


def load_embed_pairs(src_url, tgt_url, embed_dict,
                    src_lang=SRC_LANG_CODE,
                    tgt_lang=TGT_LANG_CODE):
    src_path = embed_dict[src_url]
    tgt_path = embed_dict[tgt_url]
    line_embeddings_src = get_base_embedding(src_url, src_path, src_lang)
    line_embeddings_tgt = get_base_embedding(tgt_url, tgt_path, tgt_lang)
    
    #print(line_embeddings_src.shape, line_embeddings_tgt.shape)
    return line_embeddings_src, line_embeddings_tgt
''' END OF PURE SENT EMBED FUNCS '''

def get_matching_url_dicts(input_path = ALIGNED_PAIRS_DOC):
    src_to_tgt = {}
    tgt_to_src = {}
    with open(input_path, 'r') as fp:
        for row in fp:
            src, tgt = row.split('\t')
            src_to_tgt[src] = tgt
            tgt_to_src[tgt]  = src
    return src_to_tgt, tgt_to_src


def load_embed_dict(chunks_paths):
    embed_dict = {}
    
    for chunk_path in chunks_paths:
        embed_dict_path = '%s/embedding_lookup.json' % (chunk_path)
        embed_dict_chunk = {}
        with open(embed_dict_path, 'r') as f:
            embed_dict_chunk = json.load(f)
        embed_dict.update(embed_dict_chunk)
                
    return embed_dict


''' Sampling logic start'''
def create_positive_samples(embed_dict, src_to_tgt_map, tgt_to_src_map, data_list, DOMAIN=''):
    '''
    Builds positive samples
    '''
    for src_url, tgt_url in src_to_tgt_map.items():
        src_url, tgt_url = src_url.strip(), tgt_url.strip()
        if src_url in embed_dict and tgt_url in embed_dict and (DOMAIN in src_url and DOMAIN in tgt_url):
            src_embed_path, tgt_embed_path = embed_dict[src_url], embed_dict[tgt_url]
            c = CandidateTuple(src_embed_path, tgt_embed_path, src_url, tgt_url, 1, None)
            data_list.append(c)


def create_all_possible_neg_pairs(src_to_tgt_map, tgt_to_src_map):
    
    src_url_list, tgt_url_list = list(src_to_tgt_map.keys()), list(tgt_to_src_map.keys())
    domain_dict = defaultdict(lambda: defaultdict(list)) #Ex{dom: {src: [], tgt: []}}
    for url in tgt_url_list:
        #url = url.strip()
        base_url = regex_extractor_helper(BASE_DOMAIN_RE, url).strip()
        domain_dict[base_url]['tgt'].append(url)
    
    for url in src_url_list:
        #url = url.strip()
        base_url = regex_extractor_helper(BASE_DOMAIN_RE, url).strip()
        domain_dict[base_url]['src'].append(url)
    
    negative_sample_dict = {}
    #Loop through all domains and create final negative pairing
    for domain, values in domain_dict.items():
        sample_list = []
        src_urls, tgt_urls = values['src'], values['tgt']
        
        if len(src_urls) > 100:
            shuffle(src_urls)
        if len(tgt_urls) > 100:
            shuffle(tgt_urls)
        for src_url in src_urls[:min(100, len(src_urls))]:
            for tgt_url in tgt_urls[:min(100, len(src_urls))]:
                if src_to_tgt_map[src_url] != tgt_url and tgt_to_src_map[tgt_url] != src_url:
                    sample_list.append((src_url, tgt_url))
        if len(sample_list) > 0:
            negative_sample_dict[domain] = sample_list
    return negative_sample_dict
        
    

def same_domain_neg_sample_helper(negative_sample_dict):
    '''
    IDEA: randomly modify docs  
    Helper function for returning
    Negative domains of same idx

    Algo: 1) Pick random domain
         2) Pick random sample from that domain
         3) pop that sample
    '''
    domain_list = list(negative_sample_dict.keys())
    domain = domain_list[randint(0, len(domain_list) - 1)]
    
    neg_pair_list = negative_sample_dict[domain]
    neg_pair_idx = randint(0, len(neg_pair_list) - 1)
    src_url, tgt_url = neg_pair_list[neg_pair_idx]
    
    #Update the dict to remove neg pair
    neg_pair_list.pop(neg_pair_idx)
    if len(neg_pair_list) == 0:
        negative_sample_dict.pop(domain)
    else:
        negative_sample_dict[domain] = neg_pair_list

    return src_url, tgt_url


def create_negative_samples_diff_docs(embed_dict, src_to_tgt_map, tgt_to_src_map, data_list, precent_cutoff):
    '''
    Builds negative samples
    '''

    number_pos_samples = len(data_list)
    visited_urls, MAX_INSTANCES_ALLOWED = defaultdict(int), 10 
    
    negative_sample_dict = create_all_possible_neg_pairs(src_to_tgt_map, tgt_to_src_map)    
    
    
    print("URL LIST LENGTHS", number_pos_samples, len(embed_dict), len(src_to_tgt_map), len(tgt_to_src_map))
    num_samples = int(number_pos_samples * precent_cutoff)
    
    loop_counter = 0 #NOTE: FOR DEBUG
    for i in range(num_samples):
        
        while True:
            src_url, tgt_url = same_domain_neg_sample_helper(negative_sample_dict)

            #Repeat until all conditions are met
            if (visited_urls[src_url.strip()] < MAX_INSTANCES_ALLOWED and \
                visited_urls[tgt_url.strip()] < MAX_INSTANCES_ALLOWED and \
                src_to_tgt_map[src_url] != tgt_url and tgt_to_src_map[tgt_url] != src_url):
                
                src_url, tgt_url = src_url.strip(), tgt_url.strip()
                if src_url in embed_dict and tgt_url in embed_dict:
                    src_embed_path, tgt_embed_path = embed_dict[src_url], embed_dict[tgt_url]
                    c = CandidateTuple(src_embed_path, tgt_embed_path, src_url, tgt_url, 0, None)
                    data_list.append(c)
                
                    visited_urls[src_url] += 1
                    visited_urls[tgt_url] += 1
                    
                break
            else:
                loop_counter += 1

    print("LOOP COUNTER LEN %s" % loop_counter)
    print(len(data_list))

def create_negative_samples_from_aligned_pairs(embed_dict, src_to_tgt_map, tgt_to_src_map, data_list, precent_cutoff):
    '''
    From positive samples, create negative ones, to increase sensitivity of classifier
    By dropping random handful of sents
    In general, the idx method will deal with sent dropping/ adding
    '''
    pos_list = [(src_url, tgt_url) for src_url, tgt_url in src_to_tgt_map.items()]
    
    sample_size = int(len(pos_list) * precent_cutoff)
    if sample_size >= 1:
        pos_list = pos_list * 5
        shuffle(pos_list)
    else:
        pos_list = sample(pos_list, sample_size)

    #Sample half of list and alternate which methods should be used
    method_idx, methods = 0, ["add", "delete", "sub"]
    
    for src_url, tgt_url in pos_list:
        src_url, tgt_url = src_url.strip(), tgt_url.strip()
        if src_url in embed_dict and tgt_url in embed_dict:
            src_embed_path, tgt_embed_path = embed_dict[src_url], embed_dict[tgt_url]
            c = CandidateTuple(src_embed_path, tgt_embed_path, src_url, tgt_url, 0, methods[method_idx]) 
            method_idx  = (method_idx + 1) % len(methods)
            data_list.append(c)

def calc_cos_sim(src, tgt):
    '''
    Calc cosine dist in range 0-2
    '''
    return (np.dot(src, tgt) / (np.linalg.norm(src) * np.linalg.norm(tgt))) + 1


def is_valid_negative_pair(src_doc_embed, tgt_doc_embed, y_match_label, filter_threshold):
    '''
    Neg pair filtering. Removes pairs that are too
    similar so as to not make training to difficult
    Returns boolean
    
    NOTE: filter_threshold * 2 scales it out to 0-2 scale
        ex: 0.9 will become 1.8
    '''
    if y_match_label == 1:
        return True #technically true, since, positive pair
    cosine_sim = calc_cos_sim(src_doc_embed, tgt_doc_embed)
    if cosine_sim >= (float(filter_threshold) * 2):
        return False          
    return True

        
def build_pair_dataset(embed_dict, src_to_tgt_map, tgt_to_src_map, use_augment=True, DOMAIN=''):
    '''
    Create positive and negative url tuple pairs
    Makeup of dataset will be:
        100% pure positive pairs
        500% pure negative pairs
        500% negative pairs that are close to each other
    '''
    data_list = []
    create_positive_samples(embed_dict, src_to_tgt_map, tgt_to_src_map, data_list, DOMAIN=DOMAIN) #First Create positive samples
    print(len(data_list))
    
    subset_src_to_tgt_map = {k: v for k,v in src_to_tgt_map.items() if k.strip() in embed_dict and v.strip() in embed_dict}
    subset_tgt_to_src_map = {k: v for k,v in tgt_to_src_map.items() if k.strip() in embed_dict and v.strip() in embed_dict}
    
    if use_augment:
        diff_neg_precent, close_neg_percent = 5, 5
        create_negative_samples_diff_docs(embed_dict, subset_src_to_tgt_map, subset_tgt_to_src_map, data_list, diff_neg_precent)  #Now create negative samples
        create_negative_samples_from_aligned_pairs(embed_dict, subset_src_to_tgt_map, subset_tgt_to_src_map, data_list, close_neg_percent)  #create negative samples by modding positive ones
    else:
        create_negative_samples_diff_docs(embed_dict, subset_src_to_tgt_map, subset_tgt_to_src_map, data_list, 5) #Create equal number of pos and neg samples

    #Shuffle and return data
    shuffle(data_list)
    print(len(data_list))

    return data_list



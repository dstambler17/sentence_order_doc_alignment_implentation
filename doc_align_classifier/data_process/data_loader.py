import math
import numpy as np
import linecache

from collections import defaultdict, namedtuple
from sklearn.decomposition import PCA
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from doc_align_classifier.consts import *
from doc_align_classifier.data_process.augmented_utils import create_augmented_negative_sample
from doc_align_classifier.data_process.data_utils import *

from utils.common import load_extracted, map_dic2list, \
    filter_empty_docs, regex_extractor_helper, tokenize_doc_to_sentence
from modules.get_embeddings import read_in_embeddings
from modules.build_document_vector import build_document_vector
from modules.vector_modules.boiler_plate_weighting import LIDFDownWeighting
from modules.vector_modules.window_func import ModifiedPertV2
from utils.lru_cache import LRUCache

#DEFINE Helper functions for building doc vectors
#NOTE: Much of this info will be stored in an LRU cache

def fit_pca_reducer_debug(embedding_list_src, embedding_list_tgt):
    '''
    Builds PCA Dim Reduction from sample of sentence embeddings
    in the webdomain
    '''
    all_sent_embeds = np.vstack(embedding_list_src + embedding_list_tgt)

    pca = PCA(n_components=128)
    divide_num = 1
    if len(all_sent_embeds) // 6 >= 128:
        divide_num = 6
    elif len(all_sent_embeds) // 5 >= 128:
        divide_num = 5
    elif len(all_sent_embeds) // 4 >= 128:
        divide_num = 4
    elif len(all_sent_embeds) // 3 >= 128:
        divide_num = 3
    elif len(all_sent_embeds) // 2 >= 128:
        divide_num = 2
    elif len(all_sent_embeds) // 1 >= 128:
        divide_num = 1
    else:
        sent_size = all_sent_embeds.shape[0]
        num_iters = int(math.ceil(128 / sent_size))        
        all_sent_embeds = np.repeat(all_sent_embeds, repeats=num_iters, axis=0)
        

    my_rand_int = np.random.randint(all_sent_embeds.shape[0], size=len(all_sent_embeds) // divide_num)
    pca_fit_data = all_sent_embeds[my_rand_int, :]
    pca.fit(pca_fit_data)
    return pca


class CachedData:
    '''
    Keeps organized cache of data
    Keys will be domain_name
    Since for each domain, we want the source and target lang info
    '''
    def __init__(self, src_text_list_tokenized,
                       src_embed_list,
                       src_url_list,
                       tgt_text_list_tokenized,
                       tgt_embed_list,
                       tgt_url_list,
                       ):
        self.src_text_list_tokenized = src_text_list_tokenized
        self.src_embed_list = src_embed_list
        self.src_url_list = src_url_list

        self.tgt_text_list_tokenized = tgt_text_list_tokenized
        self.tgt_embed_list = tgt_embed_list
        self.tgt_url_list = tgt_url_list
        
        self.lidf_weighter = LIDFDownWeighting(src_text_list_tokenized + tgt_text_list_tokenized)
        self.pca = fit_pca_reducer_debug(src_embed_list, tgt_embed_list)
    
    def get_fitted_objects(self):
        '''
        Return PCA and LIDF
        '''
        return self.pca, self.lidf_weighter

    def get_src_data(self):
        return self.src_text_list_tokenized, self.src_embed_list, self.src_url_list
    
    def get_tgt_data(self):
        return self.tgt_text_list_tokenized, self.tgt_embed_list, self.tgt_url_list


def load_embeds_for_domain(embed_dict, lang_code, text_list, url_list):
    '''
    Load in embeds for a domain
    ''' 
    embed_list = []
    try:
        for ii in range(len(text_list)):
            url, doc_text = url_list[ii], text_list[ii]
            #if url in embed_dict: #TEMP FIX: TODO: Rerun embeds and issue of missing URLS here or delete sentences that miss embeds
            embed_file_path = embed_dict[url]
            try:
                _, embeddings = read_in_embeddings(doc_text, embed_file_path, lang_code)
                embed_list.append(embeddings)
            except Exception as e:
                print(embed_file_path, lang_code, "EMBED EXCEPTION", e)

    except KeyError as e: #For debugging
        print(e)
        print("EXCEPTION OCCURED in load_embeds_for_domain")
    
    return embed_list


''' Domain Specific Chunk helper data'''

def get_all_chunks_with_domain(embed_dict, base_domain):
    '''
    First get a list of all chunks that contain the domain_name
    Second get all docs in src lang and tgt lang
    Third, get pca, ldf weighter and more
    '''
    chunks = [regex_extractor_helper(CHUNK_RE, value) \
                for key, value in embed_dict.items() if base_domain.strip() in key.strip().lower()]
    if len(chunks) == 0:
        print("WAIT, CHUNK LIST IS EMPTY, so CHUNK_RE error")
    return list(set(chunks))

def load_all_chunks_for_domain(chunk_list, base_domain, lang_code):
    '''
    Given a list of domain chunks and the domain_name
    Get a domain doc dict
    '''
    domain_doc_dict = {}
    for chunk in chunk_list:
        doc_path = '%s/%s.%s.gz' % (BASE_PROCESSED_PATH, chunk, lang_code)
        url_doc_dict = filter_empty_docs(load_extracted(doc_path))
        match_doc_dict = {}
        for k, v in url_doc_dict.items():
            if base_domain.strip().lower() in k.strip().lower():
                match_doc_dict[k] = v
        domain_doc_dict.update(match_doc_dict)
    return domain_doc_dict


def get_all_relevant_domain_data(chunk_list, base_domain, lang_code, embed_dict):
    '''
    return text_list_tokenized, embed_list, url list
    '''
    domain_doc_dict = load_all_chunks_for_domain(chunk_list, base_domain, lang_code) 
    obj_domain = map_dic2list(domain_doc_dict)
    
    text_list = obj_domain['text']
    text_list_tokenized = [tokenize_doc_to_sentence(doc, lang_code) for doc in text_list]
    
    url_list = [url.strip() for url in obj_domain['mapping']]
    embed_list = load_embeds_for_domain(embed_dict, lang_code,
                                        text_list, url_list)
    
    return text_list_tokenized, embed_list, url_list



def get_doc_embedding(url, text_list_tokenized,
                      url_list,
                      embedding_list,
                      lidf_weighter,
                      pca,
                      pert_obj,
                      doc_vec_method):
    '''
    Call doc embedding method
    '''
    i = url_list.index(url)
    
    return build_document_vector(text_list_tokenized[i],
                        url_list[i],
                        embedding_list[i],
                        lidf_weighter,
                        pca,
                        pert_obj,
                        doc_vec_method=doc_vec_method).doc_vector


def handle_doc_embed_logic(embed_dict, src_url, tgt_url,
                         src_lang_code, tgt_lang_code, doc_vector_method,
                         pert_obj, 
                         lru_cache,
                         augment_negative_data_method):
    
   
    base_domain_src = regex_extractor_helper(BASE_DOMAIN_RE, src_url).strip()
    base_domain_tgt = regex_extractor_helper(BASE_DOMAIN_RE, tgt_url).strip()
    
    #NOTE: NOT ALL Pairs share same base domain, ex: www.buyaas.com, si.buyaas.com, so url regex was adjusted     
    if base_domain_src != base_domain_tgt:
        print(base_domain_src, base_domain_tgt, "DIFFERENT DOMAINS")
    assert base_domain_src == base_domain_tgt  
    
    
    chunk_list_src = get_all_chunks_with_domain(embed_dict, base_domain_src) 
    chunk_list_tgt = get_all_chunks_with_domain(embed_dict, base_domain_src)
    
    cd = lru_cache.get(base_domain_src)
    if cd == -1:
        src_text_list_tokenized, src_embed_list, src_url_list = \
            get_all_relevant_domain_data(chunk_list_src, base_domain_src, src_lang_code, embed_dict)
        
        tgt_text_list_tokenized, tgt_embed_list, tgt_url_list = \
            get_all_relevant_domain_data(chunk_list_tgt, base_domain_tgt, tgt_lang_code, embed_dict)
        cd = CachedData(src_text_list_tokenized, src_embed_list, src_url_list, 
                        tgt_text_list_tokenized, tgt_embed_list, tgt_url_list)
        lru_cache.put(base_domain_src, cd)
    #else:
    #    print("cache hit")
    src_text_list_tokenized, src_embed_list, src_url_list = cd.get_src_data()
    tgt_text_list_tokenized, tgt_embed_list, tgt_url_list = cd.get_tgt_data()
    pca, lidf_weighter = cd.get_fitted_objects()
    
    #Handle negative augment data logic
    if augment_negative_data_method is not None:
        print(augment_negative_data_method, "method")
        create_augmented_negative_sample(src_url, src_text_list_tokenized, src_url_list, src_embed_list,
                            tgt_url, tgt_text_list_tokenized, tgt_url_list, tgt_embed_list,
                            augment_negative_data_method)
    
    src_doc_embed = get_doc_embedding(src_url, src_text_list_tokenized, src_url_list, src_embed_list,
                      lidf_weighter, pca, pert_obj, doc_vector_method)
    tgt_doc_embed = get_doc_embedding(tgt_url, tgt_text_list_tokenized, tgt_url_list, tgt_embed_list,
                      lidf_weighter, pca, pert_obj, doc_vector_method)
    return src_doc_embed, tgt_doc_embed


#First get embed_dict, src_to_tgt_map, tgt_to_src_map
#Then build pairset
#Finally, at each idx, just get embed_src, embed_tgt, y
class DocEmbedDataset(Dataset):
    
    """
    DocEmbeddingDataset
    """
    
    def __init__(self, chunks_paths_list, src_lang, tgt_lang, doc_vector_method="SENT_ORDER", cache_capacity=750,
                 use_augment=True, domain='', filter_threshold=0.9, use_atten=False): #NOTE: BE SURE TO ALLOCATE LOTS OF MEM
      #NOTE: Domain only uses one exclusive domain
      self.src_to_tgt_pairs, self.tgt_to_src_pairs = get_matching_url_dicts()
      self.embed_dict = load_embed_dict(chunks_paths_list)
      self.data_list = build_pair_dataset(self.embed_dict, self.src_to_tgt_pairs, self.tgt_to_src_pairs,
                                          use_augment=use_augment,
                                          DOMAIN=domain
                                          )
      
      self.src_lang = src_lang
      self.tgt_lang = tgt_lang
      
      if doc_vector_method not in ['AVG', 'AVG_BP', 'SENT_ORDER']:
        raise ValueError("""Doc Vec Method must be one of the following:
                            AVF, AVG_BP, SENT_ORDER
                            Not found: %s
                            """ % doc_vector_method)
      self.doc_vector_method = doc_vector_method
      self.pert_obj = ModifiedPertV2(None, None)
      self.lru_cache = LRUCache(cache_capacity)
      
      self.filter_threshold = filter_threshold
      self.use_atten = use_atten
      
      self.exception_counter = 0
      
      #self.lock = Lock()

    def __len__(self):
        """
        Get length of the dataset
        """
        return len(self.data_list)

    def __getitem__(self,
                    idx):
        """
        Gets the two vectors and target
        """
        _, _, src_url, tgt_url, y_match_label, augment_negative_data_method = self.data_list[idx]
        #print("CACHE LIST: %s " %  list(self.lru_cache.keys())) #REMOVE AFTER DEBUG
        #with self.lock:
        #print(y_match_label, "Y LABEL", src_url, tgt_url, augment_negative_data_method)
        '''src_doc_embedding, tgt_doc_embedding = src_doc_embedding[0], tgt_doc_embedding[0] '''
        try:
          if self.use_atten:
              src_doc_embedding, tgt_doc_embedding = load_embed_pairs(src_url, tgt_url, 
                                                                self.embed_dict,
                                                                src_lang=self.src_lang,
                                                                tgt_lang=self.tgt_lang)
              src_doc_embedding = np.stack(src_doc_embedding)
              tgt_doc_embedding = np.stack(tgt_doc_embedding)
              
          else:                                               
            src_doc_embedding, tgt_doc_embedding = handle_doc_embed_logic(self.embed_dict,
                                                                            src_url, tgt_url,
                                                                            self.src_lang, self.tgt_lang,
                                                                            self.doc_vector_method,
                                                                            self.pert_obj,
                                                                            self.lru_cache,
                                                                            augment_negative_data_method)
        except Exception as e:
          print(e, "DATA LOADING EXCEPTION, RETURNING NOISE TO BE NOT WRITTEN")
          self.exception_counter += 1
          return np.random.random((2048)), np.random.random((2048)), -1 # In rare exception cases, return this 'hacky label' so that it doesn't get saved

        #NOTE: Hack, set label to -1 in cases where pairs are deemed below the threshold
        # This way, too similar neg pairs will be false
        if self.use_atten:
            is_valid_pair = is_valid_negative_pair(np.mean(src_doc_embedding, axis=0), np.mean(tgt_doc_embedding, axis=0), y_match_label, self.filter_threshold)
        else:
            is_valid_pair = is_valid_negative_pair(src_doc_embedding, tgt_doc_embedding, y_match_label, self.filter_threshold)
        if not is_valid_pair:
            y_match_label = -1
        return src_doc_embedding, tgt_doc_embedding, y_match_label


class DocEmbedDatasetFast(Dataset):
    
    """
    DocEmbeddingDataset
    """
    
    def __init__(self, file_path, src_lang, tgt_lang, transforms=None): #NOTE: BE SURE TO ALLOCATE LOTS OF MEM
      self.src_lang = src_lang
      self.tgt_lang = tgt_lang
      self.file_path = file_path
      
      
      self.file_len = sum(1 for line in open(file_path))
      self.transforms = transforms
      
    
    def __len__(self):
      return self.file_len
    
    def __getitem__(self, idx):
      
      def str_float_list_to_np(input):
        '''
        Helper function
        '''
        input = input.replace(']','').replace('[', '')
        input = [float(item) for item in input.split(',')]
        return np.asarray(input)
    
      line_cache_idx = idx + 1
      data_record  = linecache.getline(self.file_path, line_cache_idx)
      src_raw, tgt_raw, label_str = data_record.split('\t')
      src_doc_embedding, tgt_doc_embedding = str_float_list_to_np(src_raw), str_float_list_to_np(tgt_raw)
      
      if self.transforms is not None:
        src_doc_embedding = self.transforms(src_doc_embedding)
      
      if self.transforms is not None:
        tgt_doc_embedding = self.transforms(tgt_doc_embedding)

      return src_doc_embedding, tgt_doc_embedding, int(label_str)

def get_data_loaders(use_augment=True, domain='', filter_threshold=0.9, use_attn=False):
    chunks_paths_train, chunks_paths_val, chunks_paths_test = split_train_val_test_sets()

    train_dataset=DocEmbedDataset(chunks_paths_train, SRC_LANG_CODE, TGT_LANG_CODE, use_augment=use_augment, domain=domain, filter_threshold=filter_threshold, use_atten=use_attn)
    validation_dataset=DocEmbedDataset(chunks_paths_val, SRC_LANG_CODE, TGT_LANG_CODE, use_augment=use_augment, domain=domain, filter_threshold=filter_threshold, use_atten=use_attn)
    test_dataset=DocEmbedDataset(chunks_paths_test, SRC_LANG_CODE, TGT_LANG_CODE, use_augment=use_augment, domain=domain, filter_threshold=filter_threshold, use_atten=use_attn)

    train_dataloader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True) #prefetch_factor=5, prefetch maybe could help 
    validation_dataloader = DataLoader(validation_dataset, batch_size=VAL_BATCH_SIZE, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    return train_dataloader, validation_dataloader, test_dataloader, train_dataset, validation_dataset, test_dataset


def get_fast_data_loaders(path_ext, train_batch_size=TRAIN_BATCH_SIZE, val_batch_size=VAL_BATCH_SIZE):
    '''
    Gets Fast Data Loaders (where doc vecs were already produced)
    '''    
    TRAIN_PATH = '/home/dstambl2/doc_alignment_implementations/data/cc_aligned_si_data/%s/train/data.txt' % path_ext
    VALID_PATH = '/home/dstambl2/doc_alignment_implementations/data/cc_aligned_si_data/%s/valid/data.txt' % path_ext
    TEST_PATH = '/home/dstambl2/doc_alignment_implementations/data/cc_aligned_si_data/%s/test/data.txt' % path_ext
    
    
    train_dataset = DocEmbedDatasetFast(TRAIN_PATH, SRC_LANG_CODE, TGT_LANG_CODE, transforms=None)
    validation_dataset = DocEmbedDatasetFast(VALID_PATH, SRC_LANG_CODE, TGT_LANG_CODE, transforms=None)
    test_dataset = DocEmbedDatasetFast(TEST_PATH, SRC_LANG_CODE, TGT_LANG_CODE, transforms=None)
    
    print("DATASET LENGTHS: train - %s  valid - %s test - %s" % (len(train_dataset), len(validation_dataset), len(test_dataset)))

    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, drop_last=True) #prefetch_factor=5, prefetch maybe could help 
    validation_dataloader = DataLoader(validation_dataset, batch_size=val_batch_size, shuffle=False, drop_last=True) #TODO: Remove drop_last=True when debugging
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    return train_dataloader, validation_dataloader, test_dataloader, train_dataset, validation_dataset, test_dataset
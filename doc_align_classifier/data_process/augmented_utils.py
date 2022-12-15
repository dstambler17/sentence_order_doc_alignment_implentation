import math
import numpy as np
from collections import defaultdict, namedtuple
from doc_align_classifier.consts import *

from random import shuffle, randint, sample, uniform, choice


def create_augment_helper(src_text_list_tokenized, tgt_text_list_tokenized, augment_negative_data_method):
    '''
    Goal is to pick which to augment (src or tgt) and to confirm that adding/subing is possible
    This logic is here, because when we add or sub, we are assuming that other docs exist
    1) Randomly pick if augment_negative_data_method is delete
    2) If add/sub, pick rand or larger of two
    '''
    rv = uniform(0, 1)
    random_change_item = "src" if rv >= 0.5 else "tgt"
    
    if augment_negative_data_method in ["add", "sub"]:
        if len(tgt_text_list_tokenized) == 1 and len(src_text_list_tokenized) == 1:
            augment_negative_data_method = "delete"
            item_to_change = random_change_item
        elif len(tgt_text_list_tokenized) == 1 and len(src_text_list_tokenized) > 1:
            item_to_change = "src"
        elif len(tgt_text_list_tokenized) > 1 and len(src_text_list_tokenized) == 1:
            item_to_change = "tgt"
        else:
            item_to_change = random_change_item
    
    else:
        item_to_change = random_change_item
    
    return augment_negative_data_method, item_to_change


#Cell for dealing with augmentation
def create_augmented_negative_sample(src_url, src_text_list_tokenized, src_url_list, src_embed_list,
                                    tgt_url, tgt_text_list_tokenized, tgt_url_list, tgt_embed_list,
                                    augment_negative_data_method
                                    ):
    '''
    Builds negative sample from eithers
    1 dropping a few sentences
    2. Substituting sents from another doc
    3. Adding sentences from other docs
    
    Algo
    1) Randomly pick if modifying source of target first
        If add/sub, pick domains with more data
    2) If both src and tgt have only 1 doc, default to delete method
    3) For deletion, pick random num of rand indexes of modified item
    4) For sub, delete a random index and sub out rand sents with another doc
    5) For add, just tack on rand sents

    '''
    METHODS = ["add", "delete", "sub"]
    if augment_negative_data_method not in METHODS:
        print("method not in list")
        return
    
    
    def helper_delete_multiple_element(list_object, indices):
        '''
        Helper function taken from
        https://thispointer.com/python-remove-elements-from-list-by-index/
        '''
        indices = sorted(indices, reverse=True)
        for idx in indices:
            if idx < len(list_object):
                list_object.pop(idx)
    
    def delete_logic(url, change_text_list_tokenized, change_url_list, change_embed_list):
        '''
        Handles deletion logic
        '''
        #First get data that will be augmented
        i = change_url_list.index(url)
        change_doc = change_text_list_tokenized[i]
        
        num_sents = change_embed_list[i].shape[0]
        
        #Note call add logic in this case
        if num_sents == 1:
            add_logic(url, change_text_list_tokenized, change_url_list, change_embed_list)
            return
        change_sent_embeds = change_embed_list[i].tolist()
        
        #Randomly pick how many and which sents to drop, then drop them
        num_sents_to_drop = randint(1, max(int(math.ceil((len(change_doc) - 1) / 5)), 1))
        
        #Make sure that there is always going to be one sentence left
        num_sents_to_drop = min(num_sents_to_drop, num_sents - 1)
        
        index_values = sample(list(enumerate(change_doc)), num_sents_to_drop)
        drop_index_vals = [x[0] for x in index_values]
        
        helper_delete_multiple_element(change_doc, drop_index_vals)
        helper_delete_multiple_element(change_sent_embeds, drop_index_vals)
        
        assert len(change_doc) == len(change_sent_embeds)
        
        #Update the altered doc 
        change_text_list_tokenized[i] = change_doc
        change_embed_list[i] = np.asarray(change_sent_embeds)
        
    def sub_logic(url, change_text_list_tokenized, change_url_list, change_embed_list):
        '''
        Handles substitution logic
        '''
        #First get data that will be augmented
        i = change_url_list.index(url)
        change_doc = change_text_list_tokenized[i]
        change_sent_embeds = change_embed_list[i]
        
        #Randomly pick how many sents to sub
        num_sents_to_sub = randint(1, max(int(math.ceil((len(change_doc) - 1) / 3)), 1))
        #Randomly pick which doc
        sub_idx = choice([j for j in range(len(change_text_list_tokenized)) if j!=i])
        
        #Randomly pick sents to sub with
        sub_doc, sub_embed_matrix = change_text_list_tokenized[sub_idx], change_embed_list[sub_idx]
        
        cut_off_point = min(len(change_doc), len(sub_doc))
        num_sents_to_sub = min(num_sents_to_sub, len(sub_doc[:cut_off_point]))
        index_values = sample(list(enumerate(sub_doc[:cut_off_point])), num_sents_to_sub)
        sub_index_vals = [x[0] for x in index_values]
        
        for idx in sub_index_vals:
            change_doc[idx] = sub_doc[idx]
            change_sent_embeds[idx] = sub_embed_matrix[idx]
        
        assert len(change_doc) == len(change_sent_embeds)
        #Update the altered doc
        change_text_list_tokenized[i] = change_doc
        change_embed_list[i] = change_sent_embeds

    
    def add_logic(url, change_text_list_tokenized, change_url_list, change_embed_list):
        '''
        Handles addition logic
        '''
        #First get data that will be augmented
        i = change_url_list.index(url)
        change_doc = change_text_list_tokenized[i]
        change_sent_embeds = change_embed_list[i]
        
        #Randomly pick how many sents to adds
        num_sents_to_add = randint(1, max(int(math.ceil((len(change_doc) - 1) / 5)), 2))
        #Randomly pick which doc
        add_idx = choice([j for j in range(len(change_text_list_tokenized)) if j!=i])
        
        #Randomly pick sents to add
        add_doc, add_embed_matrix = change_text_list_tokenized[add_idx], change_embed_list[add_idx]
        
        index_values = sample(list(enumerate(add_doc)), num_sents_to_add)
        add_index_vals = [x[0] for x in index_values]
        add_doc_sents = [add_doc[idx] for idx in add_index_vals]
        add_embed_sents = [add_embed_matrix[idx] for idx in add_index_vals]
        
        #Add sents
        change_doc += add_doc_sents
        change_sent_embeds = np.append(change_sent_embeds, np.asarray(add_embed_sents), axis=0)
        
        assert len(change_doc) == len(change_sent_embeds)
        
        #Update the altered doc
        change_text_list_tokenized[i] = change_doc
        change_embed_list[i] = change_sent_embeds
    
    
    augment_negative_data_method, item_to_change = create_augment_helper(src_text_list_tokenized,
                                                                         tgt_text_list_tokenized,
                                                                         augment_negative_data_method)
    if item_to_change == "src":
        url, change_text_list_tokenized, change_url_list, change_embed_list = src_url, src_text_list_tokenized, src_url_list, src_embed_list
    else:
        url, change_text_list_tokenized, change_url_list, change_embed_list = tgt_url, tgt_text_list_tokenized, tgt_url_list, tgt_embed_list
    
    if augment_negative_data_method == "add":
        #Add from one doc to another
        add_logic(url, change_text_list_tokenized, change_url_list, change_embed_list)
    elif augment_negative_data_method == "delete":
        delete_logic(url, change_text_list_tokenized, change_url_list, change_embed_list)
        #delete random number of sentences from source and target
    elif augment_negative_data_method == "sub":
        sub_logic(url, change_text_list_tokenized, change_url_list, change_embed_list)
        #swap sentences 

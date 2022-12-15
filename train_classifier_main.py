
import os
import json
import time
import re
import json
import math
import torch
import argparse

import numpy as np
import torch.nn as nn

from collections import defaultdict, namedtuple

## External Libraries
from doc_align_classifier.data_process.data_loader import get_data_loaders, get_fast_data_loaders
from doc_align_classifier.model import DocAlignerClassifier, SentenceInputAttnClassifier
from doc_align_classifier.train.train_src import train, eval_acc_and_loss_func
from doc_align_classifier.consts import DocVecPair

def main(path_extn, use_self_attention_model = False):
    if use_self_attention_model:
        train_dataloader, validation_dataloader, test_dataloader, train_dataset, validation_dataset, test_dataset = \
            get_fast_data_loaders(path_extn, train_batch_size=1, val_batch_size=1)
    else:
        train_dataloader, validation_dataloader, test_dataloader, train_dataset, validation_dataset, test_dataset = \
            get_fast_data_loaders(path_extn)

    print("Data loaded in")
    #Define the model, and kick off training
    LEARNING_RATE = 0.001 #NOTE: CAN ADJUST
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("device ready", device)
    if use_self_attention_model:
        model = SentenceInputAttnClassifier(1024)
    else:
        model = DocAlignerClassifier(4096) #Each doc vec is 2048, so times 2 will be 4096
    model = model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
    #optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.BCELoss() #nn.BCEWithLogitsLoss()

    # Since its much faster than using doc vector
    train(
            model,
            optimizer,
            loss_fn,
            LEARNING_RATE,
            train_dataloader,
            validation_dataloader,
            device,
            epochs=50
        )
    #print(train_dataset.exception_counter, "exception counter")

    #Eval on test set
    #model_checkpoint = torch.load('/home/dstambl2/doc_alignment_implementations/thompson_2021_doc_align/devtools/models/train_loop_model_5')
    #model.load_state_dict(model_checkpoint['model_state_dict'])
    #eval_acc_and_loss_func(model, test_dataloader, device, loss_fn, is_train = False, verbose = 1)


def save_doc_vec_data(path, filter_threshold, use_attn=False, use_augment=True, domain=''):
    '''
    Use this method to run just the dataset classes to create and save datasets of doc vecs
    Should speed up training quite a bit
    '''
    if domain != '':
        path = '%s/%s' % (path, domain) 
    BASE_PATH='/home/dstambl2/doc_alignment_implementations/data/cc_aligned_si_data/%s' % path  #ex: classifier_train_1_10_large_data'
    _, _, _, train_dataset, validation_dataset, test_dataset = get_data_loaders(use_augment=use_augment, domain=domain, filter_threshold=filter_threshold, use_attn=use_attn)
    
    def write_file_data(data_obj, extension, base_path=BASE_PATH):
        write_path='%s/%s/data.txt' % (base_path, extension)
        with open(write_path, 'a') as f:
            src_embed, tgt_embed, y_label = data_obj
            f.write('%s\t%s\t%s\n' % (src_embed, tgt_embed, y_label))
    
    
    def iterate_through_data(dataset, extension):
         #NOTE: Remember when loading in to conv to numpy
        for i in range(len(dataset)):
            src_doc_embedding, tgt_doc_embedding, y_match_label = dataset[i]
            if y_match_label != -1: #Don't save if invalid label
                data_obj = DocVecPair(src_doc_embedding.tolist(), tgt_doc_embedding.tolist(), y_match_label)
                write_file_data(data_obj, extension)
    
    print("Saving train w/ len: %s" % len(train_dataset))
    iterate_through_data(train_dataset, 'train')
    print("Saving valid w/ len: %s" % len(validation_dataset))
    iterate_through_data(validation_dataset, 'valid')
    print("Saving test w/ len: %s" % len(test_dataset))
    iterate_through_data(test_dataset, 'test')
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--is_training', help="training or domain",action="store_true")
    parser.add_argument('--use_self_attention_model', help="Model to use for training", action='store_true')
    parser.add_argument('--domain', help="Domain name to filter on", default='')
    parser.add_argument('--path', help="Path to write data to", default='')
    parser.add_argument('--filter_threshold', help="Cosine neg dist to filter thresh", default='')
    parser.add_argument('--use_augment', help="Tells trainer whether to use augment or not", action='store_true')
    

    args = parser.parse_args()
    if args.is_training:
        main(args.path, use_self_attention_model = args.use_self_attention_model)
    else:
        print("AUG")
        if args.domain is None:
            args.domain = ''
        save_doc_vec_data(args.path, args.filter_threshold, use_attn=args.use_self_attention_model, use_augment=args.use_augment, domain=args.domain)
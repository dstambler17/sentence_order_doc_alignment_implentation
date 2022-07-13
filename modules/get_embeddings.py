import subprocess
import os
from tabnanny import verbose
import time
import sys
import json

import numpy as np
from pathlib import Path
from threading import Lock
from nltk.tokenize import sent_tokenize
from utils.worker_manager import WorkerManager, MultithreadParams


from utils.common import function_timer, flatten_2d_list, replace_newline, clean_doc_item

  
#  Add LASER module to system path and import
LASER = os.environ.get("LASER")
sys.path.append("%s/source" % LASER)
from embed import LaserMainArgs, embed_main



#TODO: refactor this infleible function
def make_embedding_metadata(path, DEFAULT_DIR='', EMBED_PATH="/embeddings/embedding_lookup.json"):
    '''
    Creates embeddings folder and url to path look up file if none exists
    '''
    dirname = os.path.dirname(path)

    if not dirname:
        dirname = DEFAULT_DIR
    
    if not os.path.exists(dirname + '/'.join(EMBED_PATH.split('/')[:-1])):
        os.makedirs(dirname + '/'.join(EMBED_PATH.split('/')[:-1]))
        

    if not os.path.exists(dirname + EMBED_PATH):
        embed_lookup_doc = Path(dirname + EMBED_PATH)
        embed_lookup_doc.touch(exist_ok=True)
        with open(dirname + EMBED_PATH, 'w') as f:
            f.write("{}")
    
    json_data = None
    with open(dirname + EMBED_PATH, 'r') as f:
        json_data = json.load(f)
    
    return json_data

#QUESTION: SHOULD WE ONLY EMBED THE SAME LINE ONCE EVEN FOR DOC ALIGNMENT?
#NOTE: Looks like there are better results when embedding by sentence?
def read_in_embeddings(doc_text, embed_file):
    """
    Given a text file with candidate sentences and a corresponing embedding file,
       make a maping from candidate sentence to embedding index, 
       and a numpy array of the embeddings

    TAKEN FROM VEC ALIGN
    """
    doc_text = clean_doc_item(doc_text)
    sent2line = dict()
    #tokenized_text = flatten_2d_list([item.split("\n") for item in  sent_tokenize(doc_text)])
    tokenized_text = [replace_newline(sent, " ") for sent in sent_tokenize(doc_text)] #TEMP

    num_sents = 0
    for ii, line in enumerate(tokenized_text):
        #if line.strip() in sent2line:    
        #raise Exception('got multiple embeddings for the same line')
        sent2line[line.strip()] = ii
        num_sents += 1

    line_embeddings = np.fromfile(embed_file, dtype=np.float32, count=-1)
    if line_embeddings.size == 0:
        print(embed_file, "EMPTY")
        raise Exception('Got empty embedding file')
    
    if num_sents == 0:
        print(line_embeddings.size, "NUM SENTS TOO SMALL")
        num_sents = 1
    laser_embedding_size = line_embeddings.size // num_sents  # currently hardcoded to 1024
    if laser_embedding_size != 1024:
        #import pdb
        #pdb.set_trace()
        raise Exception('expected an embedding size of 1024, got %s', laser_embedding_size)
    #logger.info('laser_embedding_size determined to be %d', laser_embedding_size)
    line_embeddings.resize(line_embeddings.shape[0] // laser_embedding_size, laser_embedding_size)
    return sent2line, line_embeddings


class WriteEmbeddingsParams(MultithreadParams):
    def __init__(self, embed_file_path, doc_text, token_lang, url):
        self.embed_file_path = embed_file_path
        self.doc_text = doc_text
        self.token_lang = token_lang
        self.url = url
        super(WriteEmbeddingsParams, self).__init__()
    
    def get_params(self):
        return self.embed_file_path, self.doc_text, self.token_lang, self.url


def run_write_subprocess(encoder, token_lang, bpe_codes, embed_file_path, text_input):
    '''
    Old method to write embeddings by calling the lazer python script
    Using the subprocess module
    '''
    #command_one = "cat %s" % (dirname + "/temp.txt")

    command_one = "echo %s" % text_input #cat %s input_file, TRY the echo appraoch to save embeddings
    command_two = """python3 %s/source/embed.py --encoder %s --token-lang %s --bpe-codes %s --output %s --verbose""" % (os.environ.get("LASER"), encoder, token_lang, bpe_codes, embed_file_path)
    
    #start = time.time()
    ps = subprocess.Popen(command_one.split(" "), stdout=subprocess.PIPE)
    output = subprocess.check_output(command_two.split(" "), stdin=ps.stdout)
    ps.wait()
    
    #command_two = """python3 %s/source/embed.py --encoder %s --token-lang %s --bpe-codes %s --output %s --verbose""" % (os.environ.get("LASER"), encoder, token_lang, bpe_codes, embed_file_path)
    #command_one = "echo %s" % text_input #cat %s input_file, TRY the echo appraoch to save embeddings
    #subprocess.run()


def run_write(encoder, token_lang, bpe_codes, embed_file_path, text_input):
    '''
    New method, calls LASER directly via PYTHON
    '''

    laser_embed_args = LaserMainArgs(encoder=encoder,
                                     token_lang=token_lang,
                                     bpe_codes=bpe_codes,
                                     output=embed_file_path,
                                     verbose=True)    
    embed_main(laser_embed_args, text_input)

#Special multithreading function
def write_embeddings(embed_file_path, doc_text, token_lang):
    '''
    Calls laser embeddings to write doc embeddings
    '''
    model_dir = os.environ.get("LASER") +  "/models"
    encoder="%s/bilstm.93langs.2018-12-26.pt" % (model_dir) 
    bpe_codes="%s/93langs.fcodes" % (model_dir)

    #TODO: Confirm if this is the right way to do this
    new_text = "\n".join([replace_newline(sent, " ") for sent in sent_tokenize(doc_text)])

    run_write_subprocess(encoder, token_lang, bpe_codes, embed_file_path, new_text)
    #run_write(encoder, token_lang, bpe_codes, embed_file_path, new_text)



@function_timer
def load_embeddings(text_list, url_list, lookup_dict, lang_code, path, no_write=False):
    '''
    NOTE: We assume that text_list[i]
    corresponds w/ url_list[i]
    '''
    if len(text_list) != len(url_list):
        raise ValueError("Text list and url list must be of same length")
    
    embedding_list = []
    new_embeddings = 0
    lock = Lock()

    def write_embeddings_multithread_handler(write_embed_params : WriteEmbeddingsParams):
        embed_file_path, doc_text, token_lang, url = write_embed_params.get_params()
        write_embeddings(embed_file_path, doc_text, token_lang)
        
        #nonlocal embedding_list
        nonlocal new_embeddings
        nonlocal lock
        with lock:
            lookup_dict[url] = embed_file_path
            new_embeddings += 1
        
        _, embeddings = read_in_embeddings(doc_text, embed_file_path)
        

    #Set up worker manager
    worker_manager = WorkerManager(write_embeddings_multithread_handler, 20)
    worker_manager.start()

    for ii in range(len(text_list)):
        url, doc_text = url_list[ii], text_list[ii]
        
        if url not in lookup_dict and not no_write: #No write param: skip writing
            embed_file_path = "%s/%s.%s.emb" % (path, ii,lang_code)
            
            write_embed_params = WriteEmbeddingsParams(embed_file_path, doc_text, lang_code, url)
            worker_manager.put(write_embed_params, ii)
            
            #write_embeddings(write_embed_params)

    worker_manager.stop()

    #Handle reading after all writing is done
    #Avoid parallization issues. Reading is very fast
    for ii in range(len(text_list)):
        url, doc_text = url_list[ii], text_list[ii]
        embed_file_path = lookup_dict[url]
        _, embeddings = read_in_embeddings(doc_text, embed_file_path)
        embedding_list.append(embeddings)


    if new_embeddings >= 1 and not no_write: #Save new copy of updated json lookup dict
        with open(path + "/embeddings/embedding_lookup.json", 'w') as f:
            json.dump(lookup_dict, f)
    return embedding_list, lookup_dict



if __name__ == "__main__":
    pass
'''
Parses logs for errors that occured during q-sub:
Moves logs and pre embed files to new folders
'''

import argparse
import os
import shutil


BASE_DATA_PATH = '/home/dstambl2/doc_alignment_implementations/data'

def contains_errors(file_text):
    '''
    Parse Logs for errors
    '''
    error_msgs =  ['OSError', 'Traceback', 'CUDA error', 'RuntimeError: CUDA out of memory']
    for msg in error_msgs:
        if msg in file_text:
            return True
            
    return False

#TODO: Use command line to get list of files, extract processed and move to new folder
     # Then delete og emebed folder and log paths
def identify_and_save_error_logs(log_path, log_retry_path):
    '''
    IDs and moves error logs
    '''
    error_domains = []
    error_log_file_names = []
    for filename in os.listdir(log_path):
        f_path = os.path.join(log_path, filename)
        with open(f_path, 'r') as f:
            file_text = f.read()
            if contains_errors(file_text):
                domain = filename.split('.out')[0]
                error_domains.append(domain)

                error_log_file_names.append(filename)
                #move file
                shutil.move(f_path, "%s/%s" % (log_retry_path, filename)) #TODO: replace with move when not testing scipt
    print(error_domains, len(error_domains))
    return {d : True for d in error_domains}

def copy_preprocessed_files(retry_path, processed_data_path, domain_error_dict):
    '''
    Moves files to new path
    '''
    print(processed_data_path)
    for filename in os.listdir(processed_data_path):
        file_domain = filename[:-6]
        if file_domain in domain_error_dict:
            
            f_path = os.path.join(processed_data_path, filename)
            shutil.copyfile(f_path, "%s/%s" % (retry_path, filename))

def delete_error_embed_data(embedding_data_path, domain_error_dict):
    '''
    Deletes old embed data
    '''
    embeds_list = [os.path.join(embedding_data_path, name) for name in os.listdir(embedding_data_path) \
                    if os.path.isdir(os.path.join(embedding_data_path, name)) \
                        and name in domain_error_dict]
    #print(len(embeds_list), "EMBED LIST", embeds_list)
    for embed_path in embeds_list:
        shutil.rmtree(embed_path)


def set_up_embed_retrys(log_path, processed_data_path, embedding_data_path):
    embed_retry_path = "%s/embed_to_retry_processed" % '/'.join(processed_data_path.split('/')[:-1])
    log_retry_path =  "%s/embed_error_logs" % log_path
    if not os.path.exists(embed_retry_path):
        os.makedirs(embed_retry_path)

    if not os.path.exists(log_retry_path):
        os.makedirs(log_retry_path)
    
    domain_error_dict= identify_and_save_error_logs('%s/embed' % log_path, log_retry_path)
    print(embed_retry_path)
    copy_preprocessed_files(embed_retry_path, processed_data_path,  domain_error_dict)
    delete_error_embed_data(embedding_data_path, domain_error_dict)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_path', help="Path to logs", required=True)

    parser.add_argument('--processed_data_path', help="Path to Data", required=True)
    parser.add_argument('--embedding_data_path', help="Path to Data", required=True)

    args = parser.parse_args()

    log_path = "%s/%s" % (BASE_DATA_PATH, args.log_path)
    processed_data_path = "%s/%s" % (BASE_DATA_PATH, args.processed_data_path)
    embedding_data_path = "%s/%s" % (BASE_DATA_PATH, args.embedding_data_path)
    
    set_up_embed_retrys(log_path, processed_data_path, embedding_data_path)
    
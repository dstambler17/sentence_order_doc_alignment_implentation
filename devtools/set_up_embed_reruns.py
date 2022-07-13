'''
Parses logs for errors that occured during q-sub:
Moves logs and pre embed files to new folders
'''

import argparse
import os
import shutil

def contains_errors(file_text):
    '''
    Parse Logs for errors
    '''
    error_msgs =  ['OSError', 'Traceback', 'CUDA error']
    for msg in error_msgs:
        if msg in file_text:
            return True
            
    return False

def identify_error_logs(log_path, log_retry_path):
    '''
    IDs and moves error logs
    '''
    error_domains = []
    for filename in os.listdir(log_path):
        f_path = os.path.join(log_path, filename)
        with open(f_path, 'r') as f:
            file_text = f.read()
            if contains_errors(file_text):
                domain = filename.split('.out')[0]
                error_domains.append(domain)

                #copy file
                shutil.copyfile(f_path, "%s/%s" % (log_retry_path, filename))
    print(error_domains)
    return {d : True for d in error_domains}

def move_files(retry_path, processed_data_path, domain_error_dict):
    '''
    Moves files to new path
    '''
    print(processed_data_path)
    for filename in os.listdir(processed_data_path):
        file_domain = filename[:-6]
        if file_domain in domain_error_dict:
            f_path = os.path.join(processed_data_path, filename)
            shutil.copyfile(f_path, "%s/%s" % (retry_path, filename))

def set_up_embed_retrys(log_path, data_path):
    embed_retry_path = "%s/embed_to_retry_processed" % data_path
    log_retry_path =  "%s/embed_error_logs" % log_path
    if not os.path.exists(embed_retry_path):
        os.makedirs(embed_retry_path)

    if not os.path.exists(log_retry_path):
        os.makedirs(log_retry_path)
    
    domain_error_dict= identify_error_logs('%s/embed' % log_path, log_retry_path)
    move_files(embed_retry_path, '%s/processed' % data_path,  domain_error_dict)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_path', help="Path to logs", required=True)

    parser.add_argument('--data_path', help="Path to Data", required=True)

    args = parser.parse_args()

    set_up_embed_retrys(args.log_path, args.data_path)
import csv
import sys
import gzip
import base64
import argparse
from collections import namedtuple

BASE_PATH="/home/dstambl2/doc_alignment_implementations/data"


Page = namedtuple(
    "Page", "url, html, text, mime_type, encoding, lang")

def get_file_size(input_path) -> int:
    maxInt = sys.maxsize
    num_lines = 0
    with open(input_path, "rb") as fd:
        reader = csv.DictReader((line.decode('utf-8').replace('\0','') for line in fd), dialect="excel-tab")
        #data = open('/home/dstambl2/doc_alignment_implementations/data/en_XX-si_LK.tsv', 'rb').read()
        #print(data.find('\0'))
        csv.field_size_limit(maxInt)
        num_lines = sum(1 for line in reader)
        fd.seek(0, 0)
    return num_lines


def write_lett(page, fh):
    html = page.html
    if isinstance(html, str):
        html = html.encode('utf-8')

    text = page.text
    if isinstance(text, str):
        text = text.encode('utf-8')
    #print(type(page.lang), type(page.mime_type), type(page.encoding), type(page.url), type(page.url), type(base64.b64encode(html)), type(base64.b64encode(text)))
    lang = page.lang.encode('utf-8')
    mime_type = page.mime_type.encode('utf-8')
    encoding = page.encoding.encode('utf-8')
    url = page.url.encode('utf-8')
    
    fh.write(b"\t".join(
        [lang,
         mime_type,
         encoding,
         url,
         base64.b64encode(html),
         base64.b64encode(text)]) + b"\n")

        

def write_target_file(base_filepath, file_num, lang, corpus):
    filepath = "%s_%s.%s.gz" % (base_filepath, file_num, lang)
    print(filepath)
    #file_path = "/home/dstambl2/doc_alignment_implementations/data/" + filepath
    with gzip.open(filepath, 'wb') as f:
        for k, v in corpus.items():
            f.write(('%s\t%s\n' % (k, v)).encode('utf-8'))

def write_lett_file(lett_base_path, file_num, source_corpus, target_corpus):
    total_corpus = {**source_corpus, **target_corpus}
    new_lett_path ="%s_%s.lett.gz" % (lett_base_path, file_num)
    with gzip.open(new_lett_path, 'ab') as fl:
        for _, page in total_corpus.items():
            write_lett(page, fl)

#Note: creating lett files just in case they are needed for later
def read_and_save_files(num_chunks, file_size, 
                        input_path, processed_path, lett_path,
                        src_lang, tgt_lang):
    
    maxInt = sys.maxsize
    file_num = 0
    size_per_file = file_size // int(num_chunks)
    with open(input_path, "rb") as fd:
        reader = csv.DictReader((line.decode('utf-8').replace('\0','') for line in fd), dialect="excel-tab")
    
        src_corpus, tgt_corpus, idx = {}, {}, 0
        for row in reader:
            try:
                if idx == size_per_file:
                    #Write files 
                    write_target_file(processed_path, file_num, src_lang, src_corpus)
                    write_target_file(processed_path, file_num, tgt_lang, tgt_corpus)
                    write_lett_file(lett_path, file_num, src_corpus, tgt_corpus)
                    
                    src_corpus, tgt_corpus, idx = {}, {}, 0
                    file_num += 1
                    
                
                vals = []
                for _, v in row.items():
                    vals.append(v)
                _, src_url, src_txt, tgt_url, tgt_txt, *_ = vals
                if len(vals) > 5:
                    print("EXCEPTION FOR File Num: %s, idx: %s len_val_list: %s" % (file_num, idx, len(vals)))
                p_src = Page(src_url, '', src_txt, '', '', src_lang)
                p_tgt = Page(tgt_url, '', tgt_txt, '', '', tgt_lang)
                
                src_corpus[src_url] = p_src
                tgt_corpus[tgt_url] = p_tgt

                csv.field_size_limit(maxInt)
            except OverflowError:
                print('Exception', idx)
                maxInt = int(maxInt/10)
            idx += 1
        
        #Write once more after loop
        write_target_file(processed_path, file_num, src_lang, src_corpus)
        write_target_file(processed_path, file_num, tgt_lang, tgt_corpus)
        write_lett_file(lett_path, file_num, src_corpus, tgt_corpus)



#### URL PAIR MATCHING FUNCS ####

def write_url_file(base_filepath, src_lang, tgt_lang, url_list):
    filepath = "%s_%s_%s.pairs" % (base_filepath, src_lang, tgt_lang)
    print(filepath)
    with open(filepath, 'w') as f:
        for src_url, tgt_url in url_list:
            f.write('%s\t%s\n' % (src_url, tgt_url))

def build_target_url_pair_list(input_path):
    '''
    Builds list of src and tgt urls
    '''
    maxInt = sys.maxsize
    idx = 0
    url_pairs = []
    with open(input_path, "rb") as fd:
        reader = csv.DictReader((line.decode('utf-8').replace('\0','') for line in fd), dialect="excel-tab")
        for row in reader:
            try:                
                vals = []
                for _, v in row.items():
                    vals.append(v)
                _, src_url, _, tgt_url, _, *_ = vals
                if len(vals) > 5:
                    print("EXCEPTION FOR idx: %s len_val_list: %s" % (idx, len(vals)))
                csv.field_size_limit(maxInt)
                url_pairs.append((src_url, tgt_url))
            except OverflowError:
                print('Exception', idx)
                maxInt = int(maxInt/10)
            idx += 1
    return url_pairs
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_lang', help='source language', default='en')
    parser.add_argument('--tgt_lang', help='target language', default='si')
    parser.add_argument('--input_file_exten', help='Source File to Process', required=True)
    
    #NOTE: THESE three are required if build_url_pairs is not set
    parser.add_argument('--lett_out_exten', help='Lett Folder/file_name to write to')
    parser.add_argument('--out_exten', help='Processed Folder/file_name to write to')
    parser.add_argument('--num_splits', help='Number of Output Splits', default= 100, type=int)

    
    parser.add_argument('--build_url_pairs', help='Only process url pairs', action="store_true")

    args = parser.parse_args()
    
    input_file_path = "%s/%s" % (BASE_PATH, args.input_file_exten)
    
    if args.lett_out_exten and args.out_exten:
        lett_out_path = "%s/%s" % (BASE_PATH, args.lett_out_exten)
        out_file_path = "%s/%s" % (BASE_PATH, args.out_exten)
    elif (not args.lett_out_exten or not args.out_exten) and not args.build_url_pairs:
        print("ERROR, missing args lett_out_exten and out_exten")
        exit()
    
    if args.build_url_pairs: #Only write url pairs
        url_list = build_target_url_pair_list(input_file_path)
        print("WRITING URL DATA of length %s" % len(url_list))
        write_url_file("%s/cc_aligned" % (BASE_PATH), args.src_lang, args.tgt_lang, url_list)
    else:
        file_size = get_file_size(input_file_path)
        print(input_file_path, file_size, lett_out_path, out_file_path)
        print("WRITING DATA")
        read_and_save_files(args.num_splits, file_size, 
                            input_file_path, out_file_path, lett_out_path,
                            args.src_lang, args.tgt_lang)
    
    
    
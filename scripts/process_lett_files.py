#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Taken from WMT Shaed task repo
https://github.com/christianbuck/wmt16-document-alignment-task/blob/master/lett.py
'''

import base64
import gzip
import sys
from collections import namedtuple
import tarfile

Page = namedtuple(
    "Page", "url, html, text, mime_type, encoding, lang")


def read_lett_iter(f, decode=True):
    fh = f
    fh.seek(0)
    if f.name.endswith('.gz'):
        fh = gzip.GzipFile(fileobj=fh, mode='r')
    
    for line in fh:
        try:
            line = line.decode('utf-8')
        except UnicodeDecodeError as e:
            print(e, "trying ISO-8859-1 format")
            line = line.decode('ISO-8859-1')
        
        lang, mime, enc, url, html, text = line[:-1].split("\t")

        html = base64.b64decode(html)
        text = base64.b64decode(text)
        
        if decode:
            html = html.decode("utf-8")
            text = text.decode("utf-8")

        p = Page(url, html, text, mime, enc, lang)
        
        yield p
    


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


def read_lett(f, source_language, target_language):
    source_corpus, target_corpus = {}, {}
    for p in read_lett_iter(f):

        if p.lang == source_language:
            source_corpus[p.url] = p
        elif p.lang == target_language:
            target_corpus[p.url] = p
        else:  # ignore all other languages
            pass
    return source_corpus, target_corpus

def read_doc_file(f, source_language, target_language, decode=True):
    '''
    Read Document in cases where input is already aligned docs
    (when lett files are missing)
    '''
    #Code to manip file pointers from above (NOTE: Might not be necessary)
    fh = f
    fh.seek(0)
    if f.name.endswith('.gz'):
        fh = gzip.GzipFile(fileobj=fh, mode='r')

    source_corpus, target_corpus = {}, {}
    for line in fh:
        try:
            line = line.decode('utf-8')
        except UnicodeDecodeError as e:
            print(e, "trying ISO-8859-1 format")
            line = line.decode('ISO-8859-1')
        url_src, url_tgt, text_src, text_tgt, _ = line[:-1].split("\t")

        text_src = base64.b64decode(text_src)
        text_tgt = base64.b64decode(text_tgt)
        
        if decode:
            text_src = text_src.decode("utf-8")
            text_tgt = text_tgt.decode("utf-8")

        p_src = Page(url_src, '', text_src, '', '', source_language)
        p_tgt = Page(url_tgt, '', text_tgt, '', '', target_language)
        
        source_corpus[url_src] = p_src
        target_corpus[url_tgt] = p_tgt
    
    return source_corpus, target_corpus


def write_target_file(filepath, corpus):
    print(filepath)
    #file_path = "/home/dstambl2/doc_alignment_implementations/data/" + filepath
    with gzip.open(filepath, 'wb') as f:
        for k, v in corpus.items():
            f.write(('%s\t%s\n' % (k, v)).encode('utf-8'))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'lettfile', help='input lett file', type=argparse.FileType('rb'))
    parser.add_argument('-slang', help='source language', default='en')
    parser.add_argument('-tlang', help='target language', default='fr')
    
    #NOTE: If this is set to true, you aren't actually processing lett files, but already aligned documents
    parser.add_argument('--procssed_format', help='In xz form there are 5 fields per line, all txt no html, one for each lang', action="store_true")
    parser.add_argument('--lett_writer_path', help='In xz form we want to create lett files', default='')

    args = parser.parse_args()

    # read source and target corpus
    if args.procssed_format:
        source_corpus, target_corpus = read_doc_file(
            args.lettfile, args.slang, args.tlang)
    else:
        source_corpus, target_corpus = read_lett(
            args.lettfile, args.slang, args.tlang)

    base_file_name = args.lettfile.name.split("/")[-1].split(".lett")[0]
    domain_base = '/'.join(args.lettfile.name.split('/')[:-1])
    
    write_target_file(domain_base + '/' + base_file_name + "." + args.slang + '.gz', source_corpus)
    write_target_file(domain_base + '/' + base_file_name + "." + args.tlang + '.gz', target_corpus)

    if args.lett_writer_path:
        total_corpus = {**source_corpus, **target_corpus}
        new_lett_path = args.lett_writer_path + '/' + base_file_name + ".lett" + '.gz'
        with gzip.open(new_lett_path, 'ab') as fl:
            for _, page in total_corpus.items():
                write_lett(page, fl)
            

    sys.stderr.write("Read %d %s docs and %d %s docs from %s\n" %
                     (len(source_corpus), args.slang,
                      len(target_corpus), args.tlang, args.lettfile.name))
    

    

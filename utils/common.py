
import gzip
import re
import time

from collections import defaultdict, namedtuple
from contextlib import contextmanager

URL_RE = re.compile(r'https?\://(.*?)/', flags=re.IGNORECASE)

Page = namedtuple(
    "Page", "url, html, text, mime_type, encoding, lang")


def clean_doc_item(doc_item):
    ### Given a "Page Object"
    if doc_item.strip()[:5] == "Page(":
        return eval(doc_item).text
    return doc_item


'''CREDIT: BUCK's CODE'''
@contextmanager
def open_gzip_or_plain(file_path):

    def decode_text(file_handler):
        for line in file_handler:
            yield line.decode('utf-8')

    f = None
    try:
        if file_path[-3:] == ".gz":
            f = gzip.open(file_path, 'rb')
            yield decode_text(f)
        else:
            f = open(file_path, 'r')
            yield f

    except Exception:
        raise Exception("Error occured while loading a file!")

    finally:
        if f:
            f.close()

'''
def build_mappings(file_path_from, file_path_to, column=None, dem='\t'):
    mapping = {}

    def next_or_next_in_column(handler):
        if not column:
            return next(handler, None)

        text = next(handler, None)
        if text:
            return text.split(dem)[column]

        return text

    with open_gzip_or_plain(file_path_from) as f_from, open_gzip_or_plain(file_path_to) as f_to:
        line_from = next_or_next_in_column(f_from)
        line_to = next_or_next_in_column(f_to)

        while line_from and line_to:
            line_from = line_from.strip()
            mapping[line_from] = line_to.strip()

            line_from = next_or_next_in_column(f_from)
            line_to = next_or_next_in_column(f_to)

    return mapping
'''

'''
def check_lengths(file_path_from, file_path_to, throw=True):
    f1_lines = 0
    f2_lines = 0
    with open_gzip_or_plain(file_path_from) as f:
        for _ in f:
            f1_lines += 1

    with open_gzip_or_plain(file_path_to) as f:
        for _ in f:
            f2_lines += 1

    if throw and f1_lines != f2_lines:
        raise Exception("Files must have the same number of lines!\
                            {0}: {1}, {2}: {3}".format(file_path_from,f1_lines,file_path_to, f2_lines))

    return f1_lines == f2_lines
'''

'''CREDIT: BUCK CODE'''
def load_extracted(filepath):
    with open_gzip_or_plain(filepath) as f:
        documents = defaultdict(list)
        for line in f:
            line_split = line.strip().split('\t', 1)
            if len(line_split) != 2:
                continue

            url, text = line_split
            documents[url].append(text)

        return {d: "\n".join(documents[d]) for d in documents}


def filter_empty_docs(docs_dict):
    new_dict = {}
    for k, v in docs_dict.items():
        clean_doc = clean_doc_item(v)
        if len(clean_doc) > 0:
            new_dict[k] = clean_doc
    return new_dict


'''CREDIT: BUCK CODE'''
def map_dic2list(documents):
    mapping = []
    text = []

    for idx, d in enumerate(documents):
        mapping.append(d)
        text.append(documents[d])

    return {
        'text': text,
        'mapping': mapping
    }

def get_doc_filename_from_url(url, lang):
    '''
    Given a url, loads in a url
    '''
    base_url_arr = URL_RE.findall(url)
    if not base_url_arr:
        return None
    file_name = "%s.%s.gz" % (base_url_arr[0], lang)
    return file_name


def regex_extractor_helper(compiled_regex, raw_text):
    '''
    Given Compiled regex, and raw text pull item
    '''
    matches = compiled_regex.findall(raw_text)
    if not matches:
        return None
    return matches[0]


def flatten_2d_list(_2d_list):
    flat_list = []
    # Iterate through the outer list
    for element in _2d_list:
        if type(element) is list:
            # If the element is of type list, iterate through the sublist
            for item in element:
                flat_list.append(item)
        else:
            flat_list.append(element)
    return flat_list

def replace_newline(input, replacement=""):
    return input.replace("\n", replacement)


def function_timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        return_val = func(*args, **kwargs)
        print("Function %s took %.2f seconds" % (func.__name__,  time.time() - start))
        return return_val
    return wrapper

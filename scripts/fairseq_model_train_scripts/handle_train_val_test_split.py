import os
import argparse
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split

def clean_up_base(base_path, src_lang, tgt_lang):
    '''
    Delete files if exists
    '''
    subsets = ["train", "valid", "test"]
    for name in subsets:
        src_write_file = base_path + '/' + name + '.' + src_lang
        tgt_write_file = base_path + '/' + name + '.' + tgt_lang
        if os.path.exists(src_write_file):
            os.remove(src_write_file)
        if os.path.exists(tgt_write_file):
            os.remove(tgt_write_file)


def write_dataset_contents(base_path, file_name, src_lang, tgt_lang, line_src, line_tgt):
    src_write_file = base_path + '/' + file_name + '.' + src_lang
    tgt_write_file = base_path + '/' + file_name + '.' + tgt_lang
    with open(src_write_file, "a") as src_fp, open(tgt_write_file, "a") as tgt_fp:
        src_fp.write(line_src)
        tgt_fp.write(line_tgt)


def handle_splits(src_file, tgt_file, src_lang, tgt_lang, write_path):
    clean_up_base(write_path, src_lang, tgt_lang) #first make sure those files are deleted
    with open(src_file) as file1, open(tgt_file) as file2:
        counter = 0
        for line1, line2 in zip(file1, file2):
            counter += 1
            if counter == 9:
                write_dataset_contents(write_path, 'valid', src_lang, tgt_lang, line1, line2)
            elif counter == 10:
                write_dataset_contents(write_path, 'test', src_lang, tgt_lang, line1, line2)
                counter = 0
            else:
                write_dataset_contents(write_path, 'train', src_lang, tgt_lang, line1, line2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang_code_src', help="Lang code for Source Lang", default='en')
    parser.add_argument('--lang_code_tgt', help="Lang code for Target Lang", default='fr')

    parser.add_argument(
        '--src_file', help='path to the files ', required=True)
    parser.add_argument(
        '--tgt_file', help='path to the translated foreign text', required=True)
    parser.add_argument(
        '--output_path', help='path to folder where train, valid and tests sets are dumped', required=True)

    args = parser.parse_args()

    handle_splits(args.src_file, args.tgt_file, args.lang_code_src, args.lang_code_tgt, args.output_path)



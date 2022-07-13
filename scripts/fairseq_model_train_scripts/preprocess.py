'''
Script credit: Steven Tan
'''

import argparse
from pathlib import Path
from icu_tokenizer import Tokenizer
from tqdm import tqdm
import os


def parse_args():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lang", type=str, required=True)
    args = parser.parse_args()
    return args


def main(args):
    tokenizer = Tokenizer(lang=args.lang)
    if os.path.exists(args.output):
        os.remove(args.output)
    fout = open(args.output, "a")
    with open(args.input, "r", errors="ignore") as f:
        for i, line in enumerate(tqdm(f)):
            if len(line.strip()) < 1:
                continue
            cur_line = " ".join(tokenizer.tokenize(line))
            fout.write(cur_line)
            fout.write("\n")
    fout.close()


if __name__ == "__main__":
    args = parse_args()
    main(args)

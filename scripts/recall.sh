#!/bin/bash

DATA_ROOT="/home/dstambl2/doc_alignment_implementations/data"

ground_truth_pairs_path=$1
cand_folder_path=$DATA_ROOT/$2
target_lang=$3

source /home/dstambl2/miniconda3/etc/profile.d/conda.sh
conda activate "mt_dev"
PYTHONPATH="/home/dstambl2/doc_alignment_implementations/thompson_2021_doc_align"

#Cat all aligned pairs into one candidate file
cat $cand_folder_path/*.matches > $cand_folder_path/all_candidates.matches

#/home/dstambl2/doc_alignment_implementations/thompson_2021_doc_align/recall_tests/wmt16_test.pairs
#Opt: add -d and list of domains like -d belnois.com meatballwiki.org

python3 recall_tests/test_recall.py \
    --all_aligned_urls $ground_truth_pairs_path \
    --aligned_pair_doc $cand_folder_path/all_candidates.matches \
    -l $target_lang \
    -d bugadacargnel.com  golftrotter.com  kicktionary.de  minelinks.com  pawpeds.com  rehazenter.lu  schackportalen.nu  www.antennas.biz  www.bonnke.net  www.krn.org  www.pawpeds.com  www.summerlea.ca
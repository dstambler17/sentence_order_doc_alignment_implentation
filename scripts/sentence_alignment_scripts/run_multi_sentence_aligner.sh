#!/bin/bash

resources='hostname=c*,mem_free=30G,ram_free=20G,gpu=1'
log_dir_base='/home/dstambl2/doc_alignment_implementations/data/logs'
ROOT_DATA_DIR='/home/dstambl2/doc_alignment_implementations/data'

lang_code_src=$1
lang_code_tgt=$2


log_path_extension=$3 #Ex: sinhala_data/sentence_align/buck'
sent_pairs_path_extension=$4 #Ex: sinhala_data/aligned_sentence_pairs/buck_baseline
doc_pairs_path_extension=$5 #Ex: sinhala_data/aligned_doc_pairs/buck_baseline
doc_folder=$6 #Ex: sinhala_data/processed_pre_11_06_2018

rm $ROOT_DATA_DIR/$sent_pairs_path_extension/*

for entry in "$ROOT_DATA_DIR"/$doc_pairs_path_extension/*.matches
do
    
    entry_no_suffix=${entry::-8}
    IFS='/' read -ra path_arr <<< "$entry_no_suffix" #split
    domain_name=${path_arr[-1]}
    #echo $domain_name
    qsub -N sent_align_buck -j y -o $log_dir_base/$log_path_extension/${domain_name}_sent_align.out \
        -l $resources \
        /home/dstambl2/doc_alignment_implementations/thompson_2021_doc_align/scripts/sentence_alignment_scripts/run_sent_aligner.sh \
        sinhala_data/aligned_sentence_pairs/buck_baseline \
        sinhala_data/aligned_doc_pairs/buck_baseline \
        sinhala_data/processed_pre_11_06_2018 \
        $lang_code_src $lang_code_tgt \
        $domain_name 1
done


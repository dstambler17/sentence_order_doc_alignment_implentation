#!/bin/bash

resources='hostname=c*,mem_free=30G,ram_free=20G'
log_dir_base='/home/dstambl2/doc_alignment_implementations/data/logs'

input_path=$1
lang_code_src=$2
lang_code_tgt=$3

log_path_extension=$4 #Ex: wmt16_dev/align/k_1_no_rescore
output_path_extension=$5 #Ex: aligned_doc_pairs/k_1_no_rescore
k_val=$6

#wget -O /tmp/lid.176.bin https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin

idx=0
#Loop through and kick off embedding jobs
for entry in "$input_path"/embeddings/*/
do
    IFS='/' read -ra path_arr <<< "$entry" #split
    domain_name=${path_arr[-1]}

    src_path="${input_path}/processed/${domain_name}.${lang_code_src}.gz"
    tgt_path="${input_path}/processed/${domain_name}.${lang_code_tgt}.gz"
    out_doc_pairs_path="${input_path}/${output_path_extension}/${domain_name}.${lang_code_src}-${lang_code_tgt}.${lang_code_tgt}.matches"
    out_sent_pairs_path="${input_path}/aligned_sentences/${domain_name}.${lang_code_src}-${lang_code_tgt}.${lang_code_tgt}.aligned"

    
    qsub -j y -o $log_dir_base/$log_path_extension/$domain_name.out \
        -l $resources \
        /home/dstambl2/doc_alignment_implementations/thompson_2021_doc_align/scripts/run_single_domain_aligner.sh \
        ${src_path} ${tgt_path} ${lang_code_src} ${lang_code_tgt} \
        ${out_doc_pairs_path} ${out_sent_pairs_path} ${k_val}

done
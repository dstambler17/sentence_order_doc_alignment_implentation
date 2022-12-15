#!/bin/bash

input_path=$1
lang_code_src=$2
lang_code_tgt=$3
log_extension=$4
gpu_num=$5
data_path_extension=$6 #example: cc_aligned_si_data

resources="hostname=c*,mem_free=16G,ram_free=16G,gpu=${gpu_num}"
log_dir_base="/home/dstambl2/doc_alignment_implementations/data/logs/${log_extension}"


idx=0
#Loop through and kick off embedding jobs
for entry in "$input_path"/*
do
    entry_no_gz=${entry::-3}
    lang_code=${entry_no_gz: -2}
    
    #get domain_name
    entry_no_lang_gz=${entry::-6}
    IFS='/' read -ra path_arr <<< "$entry_no_lang_gz" #split
    domain_name=${path_arr[-1]}
     
    #Avoid double embedding, so skip over all seen domains
    if ((idx % 2)); then
        idx=$((idx+1))
        continue
    fi

    idx=$((idx+1))

    src_path="${entry_no_lang_gz}.${lang_code_src}.gz" #TODO: custom lang code in this embed script
    tgt_path="${entry_no_lang_gz}.${lang_code_tgt}.gz"
    qsub -N si_data_embed_job  -j y -o $log_dir_base/$domain_name.out \
        -l $resources \
        /home/dstambl2/doc_alignment_implementations/thompson_2021_doc_align/scripts/build_single_embeds.sh \
        ${src_path} ${tgt_path} ${lang_code_src} ${lang_code_tgt} ${domain_name} ${data_path_extension}
    echo "Kicked off job number ${idx}"
    
done
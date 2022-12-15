#!/bin/bash

src_path=$1
tgt_path=$2
lang_code_src=$3
lang_code_tgt=$4
domain_name=$5
data_path_extension=$6 #example: cc_aligned_si_data

echo $src_path
echo $tgt_path
echo $domain_name

source /home/dstambl2/miniconda3/etc/profile.d/conda.sh
conda activate "mt_dev"
PYTHONPATH="/home/dstambl2/doc_alignment_implementations/thompson_2021_doc_align"
LASER="/home/dstambl2/LASER"

export LASER="/home/dstambl2/LASER"
source /home/gqin2/scripts/acquire-gpu
#for _ in $(seq $num_gpus); do source /home/gqin2/scripts/acquire-gpu; done 

python3 /home/dstambl2/doc_alignment_implementations/thompson_2021_doc_align/scripts/build_embedding_files.py \
    --src_file ${src_path} --tgt_file ${tgt_path} \
    --lang_code_src ${lang_code_src} --lang_code_tgt ${lang_code_tgt} \
    --domain_name ${domain_name} \
    --dirname ${data_path_extension}

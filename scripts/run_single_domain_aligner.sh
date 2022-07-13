#!/bin/bash
src_path=$1
tgt_path=$2
lang_code_src=$3
lang_code_tgt=$4
output_path=$5
aligned_sentence_path=$6
k_val=$7

echo $src_path
echo $tgt_path

source /home/dstambl2/miniconda3/etc/profile.d/conda.sh
conda activate "mt_dev"
PYTHONPATH="/home/dstambl2/doc_alignment_implementations/thompson_2021_doc_align"
LASER="/home/dstambl2/LASER"

export LASER="/home/dstambl2/LASER"
#ldd --version
#python -m pip install open3d==0.10
#source /home/gqin2/scripts/acquire-gpu

#Goal, print out all doc pair alignments and all sentence alignments
#wget -O /tmp/lid.176.bin https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin

python3 /home/dstambl2/doc_alignment_implementations/thompson_2021_doc_align/align_docs.py \
    --english $src_path \
    --foreign $tgt_path \
    --lang_code_src $lang_code_src\
    --lang_code_tgt $lang_code_tgt\
    --output_matches $output_path    \
    --output_sentences $aligned_sentence_path \
    --threshold 0.0     \
    --batch_size 1000 \
    --doc_vector_method SENT_ORDER \
    --no_embed_write --k_val $k_val \
    --use_rescore



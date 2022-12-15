##!/bin/bash


resources="hostname=c*,mem_free=20G,ram_free=30G"
log_dir_base="/home/dstambl2/doc_alignment_implementations/data/logs/BLEU_SCORE_MT_TRAIN_SIN"

input_path=$1
output_path=$2
SRC=$3
TGT=$4


qsub -N preprocess_fairseq_downstream -j y -o $log_dir_base/downstream_fairseq_preprocess.out \
    -l $resources \
    /home/dstambl2/doc_alignment_implementations/thompson_2021_doc_align/scripts/fairseq_model_train_scripts/preprocess.sh \
    $input_path $output_path $SRC $TGT
echo "Kicked off downstream MT fairseq preprocesses training job"

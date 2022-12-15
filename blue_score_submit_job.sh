##!/bin/bash


resources="hostname=c*,mem_free=20G,ram_free=30G,gpu=1"
log_dir_base="/home/dstambl2/doc_alignment_implementations/data/logs/BLEU_SCORE_MT_TRAIN"

tgt_path=$1
src_lang=$2

qsub -N fairseq_downstream_train -j y -o $log_dir_base/downstream_fairseq.out \
    -l $resources \
    /home/dstambl2/doc_alignment_implementations/thompson_2021_doc_align/scripts/fairseq_model_train_scripts/train.sh \
        $tgt_path $src_lang
echo "Kicked off downstream MT fairseq training job"

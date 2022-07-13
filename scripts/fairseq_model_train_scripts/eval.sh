##!/bin/bash

source /home/dstambl2/miniconda3/etc/profile.d/conda.sh
conda activate "mt_dev"


#TODO: replace
model_path="/home/dstambl2/doc_alignment_implementations/data/wmt16_test/fairseq_downstream_task/checkpoints/checkpoint_best.pt"
data_path="/home/dstambl2/doc_alignment_implementations/data/wmt16_test/fairseq_downstream_task/data-bin/tokenized.en-fr"

#NOTE, will use this to eval models if needed
fairseq-generate $data_path \
    --path $model_path \
    --batch-size 128 \
    --beam 5 \
    --seed 1 \
    --scoring bleu \
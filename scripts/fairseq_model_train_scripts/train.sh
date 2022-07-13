##!/bin/bash

source /home/dstambl2/miniconda3/etc/profile.d/conda.sh
conda activate "mt_dev"

DATA_ROOT=/home/dstambl2/doc_alignment_implementations/data

dataset_path=$1 #Ex: wmt16_test/fairseq_downstream_task
src_lang=$2 #Ex: wmt16_test/fairseq_downstream_task
tgt_lang=$3 #Ex: wmt16_test/fairseq_downstream_task

input_data=${DATA_ROOT}/${dataset_path}/data-bin/tokenized.${src_lang}-${tgt_lang}

echo $input_data

fairseq-train ${input_data} \
    --arch transformer --dropout 0.1 --attention-dropout 0.1  --activation-dropout 0.1  \
    --encoder-embed-dim 256 --encoder-ffn-embed-dim 512  --encoder-layers 3  \
    --encoder-attention-heads 8 --encoder-learned-pos \
    --decoder-embed-dim 256 --decoder-ffn-embed-dim 512 \
    --decoder-layers 3 --decoder-attention-heads 8  --decoder-learned-pos \
    --max-epoch 10 --optimizer adam --lr 5e-4 --batch-size 128 --seed 1 \
    --save-dir ${DATA_ROOT}/${dataset_path}/checkpoints

echo "fairseq-train ${input_data} \
    --arch transformer  \
    --dropout 0.1  \
    --attention-dropout 0.1   \
    --activation-dropout 0.1  \
    --encoder-embed-dim 256  \
    --encoder-ffn-embed-dim 512  \
    --encoder-layers 3  \
    --encoder-attention-heads 8  \
    --encoder-learned-pos \
    --decoder-embed-dim 256 \
    --decoder-ffn-embed-dim 512 \
    --decoder-layers 3 --decoder-attention-heads 8  --decoder-learned-pos \
    --max-epoch 10 --optimizer adam --lr 5e-4 --batch-size 128 --seed 1"

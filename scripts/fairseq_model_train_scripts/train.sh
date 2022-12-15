##!/bin/bash

source /home/dstambl2/miniconda3/etc/profile.d/conda.sh
conda activate "mt_dev"

DATA_ROOT=/home/dstambl2/doc_alignment_implementations/data
FAIRSEQ='/home/dstambl2/fairseq'

dataset_path=$1 #Ex: wmt16_test/fairseq_downstream_task
SRC=$2 #Ex: en

DATA_BIN=${DATA_ROOT}/${dataset_path}/data-bin #/tokenized.${src_lang}-${tgt_lang}

source /home/gqin2/scripts/acquire-gpu
#echo $input_data

python $FAIRSEQ/train.py \
    $DATA_BIN/tokenized.en-fr \
    --source-lang $SRC --target-lang en \
    --arch transformer \
    --encoder-layers 5 --decoder-layers 5 \
    --encoder-embed-dim 512 --decoder-embed-dim 512 \
    --encoder-ffn-embed-dim 2048 --decoder-ffn-embed-dim 2048 \
    --encoder-attention-heads 2 --decoder-attention-heads 2 \
    --encoder-normalize-before --decoder-normalize-before \
    --dropout 0.4 --attention-dropout 0.2 --relu-dropout 0.2 \
    --weight-decay 0.0001 \
    --label-smoothing 0.2 --criterion label_smoothed_cross_entropy \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0 \
    --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-7 \
    --lr 1e-3 \
    --max-tokens 4000 \
    --update-freq 4 \
    --max-epoch 100 --save-interval 10

echo 'fairseq-train ${input_data} \
    --arch transformer --dropout 0.1 --attention-dropout 0.1  --activation-dropout 0.1  \
    --encoder-embed-dim 256 --encoder-ffn-embed-dim 512  --encoder-layers 3  \
    --encoder-attention-heads 8 --encoder-learned-pos \
    --decoder-embed-dim 256 --decoder-ffn-embed-dim 512 \
    --decoder-layers 3 --decoder-attention-heads 8  --decoder-learned-pos \
    --max-epoch 10 --optimizer adam --lr 5e-4 --batch-size 128 --seed 1 \
    --save-dir ${DATA_ROOT}/${dataset_path}/checkpoints'

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

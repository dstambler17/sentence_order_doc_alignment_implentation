
##!/bin/bash

source /home/dstambl2/miniconda3/etc/profile.d/conda.sh
conda activate "mt_dev"
PYTHONPATH="/home/dstambl2/doc_alignment_implementations/thompson_2021_doc_align"
LASER="/home/dstambl2/LASER"


#Creates preprocessed training dataset from aligned sentences
DATA_ROOT=/home/dstambl2/doc_alignment_implementations/data
PYTHON_ROOT=/home/dstambl2/doc_alignment_implementations/thompson_2021_doc_align

input_folder=$1 #sinhala_data/aligned_sentence_pairs/buck_baseline
output_folder=$2 #ex: sinhala_data/fairseq_downstream_task/buck_baseline
src_lang=$3 #si, the foreign lang
tgt_lang=$4 #en, the tgt lang

input_file_path=${DATA_ROOT}/${input_folder}
output_path=${DATA_ROOT}/${output_folder}

#Concatenate all *-final-{lang}.alignments files
cat $input_file_path/*-final-${src_lang}.alignments > $output_path/raw_data/all_sentences-${src_lang}.alignments
cat $input_file_path/*-final-${tgt_lang}.alignments > $output_path/raw_data/all_sentences-${tgt_lang}.alignments

#TODO: potentially depreicate this logic, or loop
#cat ${file} | \
#  perl $norm_punc en | \
#  perl $rem_non_print_char | \
#  perl $tokenizer -threads 8 -a -l en > ${DATASET}/${lang}.tok2


python3 ${PYTHON_ROOT}/scripts/fairseq_model_train_scripts/handle_train_val_test_split.py \
    --lang_code_src ${src_lang} \
    --lang_code_tgt ${tgt_lang} \
    --src_file $output_path/raw_data/all_sentences-${src_lang}.alignments \
    --tgt_file $output_path/raw_data/all_sentences-${tgt_lang}.alignments --output_path $output_path/raw_data


declare -a data_subsets=("train" "valid" "test")

#Loop through all aligned sents and create tok files, change lang based on file
for entry in "${data_subsets[@]}"
do
    python3 ${PYTHON_ROOT}/scripts/fairseq_model_train_scripts/preprocess.py \
        --input ${output_path}/raw_data/${entry}.${src_lang} \
        --output ${output_path}/${entry}.tok.${src_lang} \
        --lang ${src_lang}
    
    python3 ${PYTHON_ROOT}/scripts/fairseq_model_train_scripts/preprocess.py \
        --input ${output_path}/raw_data/${entry}.${tgt_lang} \
        --output ${output_path}/${entry}.tok.${tgt_lang} \
        --lang ${tgt_lang}
done


#TODO: Concat all files then create train, valid, test datasets, by random split
fairseq-preprocess --source-lang $src_lang \
                   --target-lang $tgt_lang  \
                   --trainpref ${output_path}/train.tok \
                   --validpref ${output_path}/valid.tok \
                   --testpref ${output_path}/test.tok \
                   --destdir ${DATA_ROOT}/${output_folder}/data-bin/tokenized.${src_lang}-${tgt_lang} \
                   --thresholdsrc 2 \
                   --thresholdtgt 2

#!/bin/bash
#Runs sentence alignments on all matches

BASE_DIR="/home/dstambl2/doc_alignment_implementations/data"
RESOURCES='hostname=c*,mem_free=30g,ram_free=20g'
LOG_BASE_DIR='/home/dstambl2/doc_alignment_implementations/data/logs'

SENTENCE_ALIGNER_SCRIPT_PATH="/home/dstambl2/doc_alignment_implementations/thompson_2021_doc_align/scripts/sentence_alignment_scripts/"

data_dir=$1 #ex: wmt16_test
src_lang=$2
tgt_lang=$3

input_path=$BASE_DIR/$data_dir/aligned_doc_pairs
echo $input_path

for entry in "$input_path"/*
do
  IFS='/' read -ra path_arr <<< "$entry" #split
  file_name="${path_arr[-1]}"
  domain_name=${file_name::-8}
  echo $entry
  echo $domain_name

  #skip if contains all pairs
  if [[ $domain_name == "all_candidates" ]]; then
    echo skipping $domain_name
    continue
  fi
  
  qsub -j y -o $LOG_BASE_DIR/$data_dir/align$domain_name.out \
    -l $RESOURCES \
    $SENTENCE_ALIGNER_SCRIPT_PATH/run_sent_aligner.sh \
        $BASE_DIR \
        $data_dir \
        $domain_name \
        $src_lang \
        $tgt_lang
done



'''
  $SENTENCE_ALIGNER_SCRIPT_PATH/run_sent_aligner.sh \
        $BASE_DIR \
        $data_dir \
        $domain_name \
        $src_lang \
        $tgt_lang 
'''
#!/bin/bash

ALIGNER_SCRIPT_PATH="/home/dstambl2/doc_alignment_implementations/buck_aligner"
MEM_RESOURCES='mem_free=20g,ram_free=30g'
LOG_PATH_ROOT='/home/dstambl2/doc_alignment_implementations/data/logs'

# Processes all let and unzips them
#$ALIGNER_SCRIPT_PATH/document-align-buck-malign.bash $ouput_path $lang $domain_name

if [ -z $1 ] ; then
  echo "Provide the file path for the folder with the raw lett files"
  exit
fi

if [ -z $2 ] ; then
  echo "provide the folder where we will dump data to"
  exit
fi

if [ -z $3 ] ; then
  echo "provide the target lamguage id"
  exit
fi

if [ -z $4 ] ; then
  echo "provide the log path"
  exit
fi


input_path=$1
ouput_path=$2
lang=$3

log_dir_base=$LOG_PATH_ROOT/$4 #ex: wmt16_dev/buck_16_baseline
resources="hostname=${5}*,$MEM_RESOURCES" #ex: $5 is either 'c' or b

for entry in "$input_path"/*
do
  
  IFS='/' read -ra path_arr <<< "$entry" #split
  file_name="${path_arr[-1]}"
  domain_name=${file_name::-8}
  echo $domain_name

  cp $entry $ouput_path
  #$ALIGNER_SCRIPT_PATH/document-align-buck-malign.bash $ouput_path $lang $domain_name
  
  qsub -j y -o $log_dir_base/$domain_name.out \
    -l $resources \
    ${ALIGNER_SCRIPT_PATH}/document-align-buck-malign.bash $ouput_path $lang $domain_name
  
done

#echo ${input_path}
#echo ${ouput_path}

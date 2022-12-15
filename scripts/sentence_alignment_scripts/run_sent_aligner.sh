#!/bin/bash

#Takes in path to matches, and builds text doc
#Calls overlaps
#Calls vecalign
source /home/dstambl2/miniconda3/etc/profile.d/conda.sh
conda activate "mt_dev"
export LASER="/home/dstambl2/LASER"
PYTHONPATH="/home/dstambl2/doc_alignment_implementations/thompson_2021_doc_align"
BASE_DATA_PATH="/home/dstambl2/doc_alignment_implementations/data"

#declare all path related variables
aligned_sent_path=$1 #Ex: sinhala_data/aligned_sentence_pairs/buck_baseline_prev
aligned_doc_path=$2 #Ex: #sinhala_data/aligned_doc_pairs/buck_baseline_prev 
raw_document_path=$3 #Ex sinhala_data/processed_pre_11_06_2018

src_lang=$4
tgt_lang=$5

match_domain=$6
is_buck_aligner=$7


alignment_path=${BASE_DATA_PATH}/${aligned_sent_path}
echo $alignment_path
echo "Running build docs"

python3 $PYTHONPATH/build_docs.py \
    --matches $aligned_doc_path/$match_domain.matches \
    --threshold 0.0 -d $raw_document_path \
    --src_lang $src_lang \
    --target_lang $tgt_lang \
    --is_buck_aligner $is_buck_aligner #> $alignment_path/$match_domain-combined_docs.txt

tgt_input_file=${alignment_path}/$match_domain-${tgt_lang}.temp_texts
src_input_file=${alignment_path}/$match_domain-${src_lang}.temp_texts

tgt_overlap_file=${alignment_path}/${match_domain}_overlaps.${tgt_lang}
src_overlap_file=${alignment_path}/${match_domain}_overlaps.${src_lang}

#Build overlaps and their embeddings
echo $tgt_overlap_file
echo $src_overlap_file
source /home/gqin2/scripts/acquire-gpu

python3  $PYTHONPATH/vec_align/overlap.py -i $tgt_input_file -o $tgt_overlap_file -n 5

python3  $PYTHONPATH/vec_align/overlap.py -i $src_input_file -o $src_overlap_file -n 5

echo "CREATED BOTH OVERLAP FILES"
$LASER/tasks/embed/embed.sh $tgt_overlap_file $tgt_lang ${tgt_overlap_file}.emb
$LASER/tasks/embed/embed.sh $src_overlap_file $src_lang ${src_overlap_file}.emb

echo "EMBEDED BOTH OVERLAP FILES"


python3  $PYTHONPATH/vec_align/vecalign.py --alignment_max_size 8 \
    --src $src_input_file \
    --tgt $tgt_input_file \
    --src_embed $src_overlap_file ${src_overlap_file}.emb  \
    --tgt_embed $tgt_overlap_file ${tgt_overlap_file}.emb > $alignment_path/$match_domain-sentence_idxs.txt

echo "RAN VECALIGN"

#After vecalign, build seperate sentence files
python3 $PYTHONPATH/generate_aligned_sentences.py \
    --src_doc_path $src_input_file \
    --tgt_doc_path $tgt_input_file \
    --index_path $alignment_path/${match_domain}-sentence_idxs.txt \
    -o $alignment_path/${match_domain}-final \
    --src_lang $src_lang \
    --tgt_lang $tgt_lang
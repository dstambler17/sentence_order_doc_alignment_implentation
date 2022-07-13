#!/bin/bash

#Takes in path to matches, and builds text doc
#Calls overlaps
#Calls vecalign
source /home/dstambl2/miniconda3/etc/profile.d/conda.sh
conda activate "mt_dev"
export LASER="/home/dstambl2/LASER"
PYTHONPATH="/home/dstambl2/doc_alignment_implementations/thompson_2021_doc_align"

#declare all path related variables
base_path=$1
dataset_folder=$2
match_domain=$3

src_lang=$4
tgt_lang=$5

alignment_path=${base_path}/${dataset_folder}/aligned_sentences
echo $alignment_path
echo "TEST"

#Build doc matches from urls
python3 scripts/sentence_alignment_scripts/build_docs.py \
    --matches $dataset_folder/aligned_doc_pairs/$match_domain.matches \
    --threshold 0.0 -d wmt16_test/processed \
    --src_lang $src_lang \
    --target_lang $tgt_lang > $alignment_path/$match_domain-combined_docs.txt

tgt_input_file=${alignment_path}/$match_domain-${tgt_lang}.temp_texts
src_input_file=${alignment_path}/$match_domain-${src_lang}.temp_texts

tgt_overlap_file=${alignment_path}/${match_domain}_overlaps.${tgt_lang}
src_overlap_file=${alignment_path}/${match_domain}_overlaps.${src_lang}

#Build overlaps and their embeddings
echo $tgt_overlap_file
echo $src_overlap_file

python3 vec_align/overlap.py -i $tgt_input_file -o $tgt_overlap_file -n 5

python3 vec_align/overlap.py -i $src_input_file -o $src_overlap_file -n 5

$LASER/tasks/embed/embed.sh $tgt_overlap_file $tgt_lang ${tgt_overlap_file}.emb
$LASER/tasks/embed/embed.sh $src_overlap_file $src_lang ${src_overlap_file}.emb

python3 vec_align/vecalign.py --alignment_max_size 8 \
    --src $src_input_file \
    --tgt $tgt_input_file \
    --src_embed $src_overlap_file ${src_overlap_file}.emb  \
    --tgt_embed $tgt_overlap_file ${tgt_overlap_file}.emb > $alignment_path/$match_domain-sentence_idxs.txt


#After vecalign, build seperate sentence files
python3 scripts/sentence_alignment_scripts/generate_aligned_sentences.py \
    --src_doc_path $src_input_file \
    --tgt_doc_path $tgt_input_file \
    --index_path $alignment_path/${match_domain}-sentence_idxs.txt \
    -o $alignment_path/${match_domain}-final \
    --src_lang $src_lang \
    --tgt_lang $tgt_lang
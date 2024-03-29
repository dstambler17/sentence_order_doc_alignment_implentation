# Exploiting Sentence order in Document Alignment Implementaion

This Repository implements the following paper by Brain Thompson, on [Document Alignment](https://aclanthology.org/2020.emnlp-main.483.pdf) Credit to the original Authors

This methodology builds a Document Vector from sentence embeddings, genereted by [LASER](https://github.com/facebookresearch/LASER), along side a windowing and boiler plate downweighting function

Aligned Document Pairs are generated using Approximate KNN and Cosine Similarity
These pairs are rescored using [Vecalign](https://aclanthology.org/D19-1136.pdf) to exploit their sentence order, leading to better performance

# Running the Aligner
```
python3 align_docs.py \
    --english $src_path \
    --foreign $tgt_path \
    --lang_code_src $lang_code_src\
    --lang_code_tgt $lang_code_tgt\
    --output_matches $output_path    \
    --output_sentences $aligned_sentence_path \
    --threshold 0.0     \
    --batch_size 1000 \
    --doc_vector_method SENT_ORDER \
    --no_embed_write --k_val $k_val \
    --use_rescore
```

# Running Scripts
All scripts from extracting records from the tarball
to training a language model are shown here in order
Data comes from the [WMT16 Conference Document Alignment Shared Task](https://www.statmt.org/wmt16/bilingual-task.html)

Machine Translation models are trained using the [fairseq library](https://fairseq.readthedocs.io/en/latest/index.html)

Below is the process of running scripts with samples

## Getting the Data
### 1) Get the data
```
wget http://data.statmt.org/wmt16/document-alignment-task/lett.test.tgz
```

### 2) Extract to folder
```
unzip lett.test.tgz -d /home/dstambl2/doc_alignment_implementations/data/wmt16_test/lett_files
```

For reference, here is the logic to parse in tsv data
```
python3 scripts/process_tsv_files.py --input_file_exten en_XX-si_LK.tsv --lett_out_exten cc_aligned_si_data/lett_data/chunk --out_exten cc_aligned_si_data/processed/chunk --num_splits 100
```

### 3) Process the Lett files, saving output to log file
```
scripts/unzip.sh /home/dstambl2/doc_alignment_implementations/data/wmt16_test lett_files processed gz > /home/dstambl2/doc_alignment_implementations/data/logs/wmt16_test/unzip_processing_log.txt
```


## Running Document Alignments
### 1) Building the LASER embeddings for each Doc
```
scripts/build_multi_embeds.sh /home/dstambl2/doc_alignment_implementations/data/wmt16_dev/processed en fr wmt16_dev/embed 2
```

For single domain embeddings, sample:
```
python3 scripts/build_embedding_files.py     --src_file /home/dstambl2/doc_alignment_implementations/data/wmt16_dev/processed/schackportalen.nu.en.gz --tgt_file /home/dstambl2/doc_alignment_implementations/data/wmt16_dev/processed/schackportalen.nu.fr.gz   --lang_code_src en --lang_code_tgt fr --domain_name schackportalen.nu
```


### 2) Running Aligner
```
./scripts/multi_domain_doc_aligner.sh /home/dstambl2/doc_alignment_implementations/data/wmt16_test en fr wmt16_test/align/k_32_no_rescore aligned_doc_pairs/k_32_no_rescore 32
```

To run for a single domain

```
scripts/run_single_domain_aligner.sh         /home/dstambl2/doc_alignment_implementations/data/wmt16_dev/processed/kicktionary.de.en.gz /home/dstambl2/doc_alignment_implementations/data/wmt16_dev/processed/kicktionary.de.fr.gz en fr         /home/dstambl2/doc_alignment_implementations/data/wmt16_dev/aligned_doc_pairs/k_8_no_rescore/kicktionary.de.en-fr.fr.matches /home/dstambl2/doc_alignment_implementations/data/wmt16_dev/aligned_sentences/kicktionary.de.en-fr.fr.aligned 8
```

Or

```
scripts/run_single_domain_aligner.sh         /home/dstambl2/doc_alignment_implementations/data/wmt16_dev/processed/schackportalen.nu.en.gz /home/dstambl2/doc_alignment_implementations/data/wmt16_dev/processed/schackportalen.nu.fr.gz en fr         /home/dstambl2/doc_alignment_implementations/data/wmt16_dev/aligned_doc_pairs/greedy_sentence_method_no_rescore/schackportalen.nu.en-fr.fr.matches /home/dstambl2/doc_alignment_implementations/data/wmt16_dev/aligned_sentences/schackportalen.nu.en-fr.fr.aligned 8
```

### 3) Evaluating Recall
```
scripts/recall.sh /home/dstambl2/doc_alignment_implementations/data/wmt16_train.pairs wmt16_dev/aligned_doc_pairs/k_8_no_rescore fr
```

## Downstream sentence Alignment and Model Training
### 1) Run Sentence Aligner scripts (also can be run with scripts/sentence_alignment_scripts/run_sent_aligner.sh)

```
resources="hostname=b*,mem_free=20G,ram_free=30G,gpu=1"
log_dir_base="/home/dstambl2/doc_alignment_implementations/data/logs/classifier_training"


`cat *.matches > all_matched_pairs.matches`


### Much better script to run for multi sent aligner
```
scripts/sentence_alignment_scripts/run_multi_sentence_aligner.sh en si sinhala_data/sentence_align/buck sinhala_data/aligned_sentence_pairs/buck_baseline sinhala_data/aligned_doc_pairs/buck_baseline sinhala_data/processed_pre_11_06_2018
```

qsub -N sent_align_buck -j y -o /home/dstambl2/doc_alignment_implementations/data/logs/sinhala_data/sentence_align/buck/sent_align.out \
-l hostname=c*,mem_free=120G,ram_free=120G,gpu=1 \
/home/dstambl2/doc_alignment_implementations/thompson_2021_doc_align/scripts/sentence_alignment_scripts/run_sent_aligner.sh \
sinhala_data/aligned_sentence_pairs/buck_baseline \
sinhala_data/aligned_doc_pairs/buck_baseline \
sinhala_data/processed_pre_11_06_2018 \
en si \
all_matched_pairs 1
```

```
python3 scripts/sentence_alignment_scripts/build_docs.py --matches wmt16_test/aligned_doc_pairs/belnois.com.matches --threshold 0.0 -d wmt16_test/processed --src_lang en --target_lang fr > combined_docs.txt
```

```
#Build overlaps and their embeddings
python3 vec_align/overlap.py -i /home/dstambl2/doc_alignment_implementations/data/wmt16_test/aligned_sentences/belnois.com-fr.temp_texts -o TEST_FR_OVERLAPS.fr -n 5
```

```
python3 vec_align/overlap.py -i /home/dstambl2/doc_alignment_implementations/data/wmt16_test/aligned_sentences/belnois.com-en.temp_texts -o TEST_EN_OVERLAPS.en -n 5
```

```
LASER/tasks/embed/embed.sh TEST_FR_OVERLAPS.fr fr TEST_FR_OVERLAPS.fr.emb
LASER/tasks/embed/embed.sh TEST_EN_OVERLAPS.en en TEST_EN_OVERLAPS.en.emb
```

```
python3 vec_align/vecalign.py --alignment_max_size 8 \
    --src /home/dstambl2/doc_alignment_implementations/data/wmt16_test/aligned_sentences/belnois.com-en.temp_texts \
    --tgt /home/dstambl2/doc_alignment_implementations/data/wmt16_test/aligned_sentences/belnois.com-fr.temp_texts \
    --src_embed TEST_EN_OVERLAPS.en TEST_EN_OVERLAPS.en.emb  \
    --tgt_embed TEST_FR_OVERLAPS.fr TEST_FR_OVERLAPS.fr.emb > sentence_idxs.txt
```

```
python3 scripts/sentence_alignment_scripts/generate_aligned_sentences.py --src_doc_path /home/dstambl2/doc_alignment_implementations/data/wmt16_test/aligned_sentences/belnois.com-en.temp_texts --tgt_doc_path /home/dstambl2/doc_alignment_implementations/data/wmt16_test/aligned_sentences/belnois.com-fr.temp_texts --index_path sentence_idxs.txt -o /home/dstambl2/doc_alignment_implementations/data/wmt16_test/aligned_sentences/belnois.com-final
```

### 2) Model training downstream task

Preprocess
```
scripts/fairseq_model_train_scripts/preprocess.sh wmt16_test/aligned_sentences wmt16_test/fairseq_downstream_task en fr
```

Or
```
./submit_preprocess_bleu.sh wmt16_test/aligned_sentences wmt16_test/fairseq_downstream_task fr en

./submit_preprocess_bleu.sh sinhala_data/aligned_sentence_pairs/buck_baseline sinhala_data/fairseq_downstream_task/buck_baseline si en
```

Train a model
```
scripts/fairseq_model_train_scripts/train.sh wmt16_test/fairseq_downstream_task en fr
```

Eval and Get BLUE score
```
scripts/fairseq_model_train_scripts/eval.sh
```



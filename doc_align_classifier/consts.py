import torch
import re
from collections import namedtuple

## Batch Size
TRAIN_BATCH_SIZE = 32
VAL_BATCH_SIZE = 32

## Learning Rate
LR = 0.001

# Epochs (Consider setting high and implementing early stopping)
NUM_EPOCHS = 100

GPU_BOOL = torch.cuda.is_available()
GPU_BOOL

BASE_DATA_PATH="/home/dstambl2/doc_alignment_implementations/data"
BASE_EMBED_DIR = '/home/dstambl2/doc_alignment_implementations/data/cc_aligned_si_data/embeddings'

BASE_PROCESSED_PATH="/home/dstambl2/doc_alignment_implementations/data/cc_aligned_si_data/processed" 
ALIGNED_PAIRS_DOC = '/home/dstambl2/doc_alignment_implementations/data/cc_aligned_en_si.pairs'

SRC_LANG_CODE="en"
TGT_LANG_CODE="si"


CHUNK_RE = re.compile(r'(chunk_\d*(?:_\d*)?)', flags=re.IGNORECASE)
BASE_DOMAIN_RE = re.compile(r'https?\://(?:w{3}\.)?(?:(?:si|en|sinhala|english)\.)?(.*?)/', flags=re.IGNORECASE)


CandidateTuple = namedtuple(
    "CandidateTuple", "src_embed_path, tgt_embed_path, src_url, tgt_url, y_match_label, augment_neg_data_method")

DocVecPair = namedtuple(
    "DocVecPair", "src_doc_embed, tgt_doc_embed, y_match_label")

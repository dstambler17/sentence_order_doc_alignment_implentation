source /home/dstambl2/miniconda3/etc/profile.d/conda.sh
conda activate "mt_dev"
PYTHONPATH="/home/dstambl2/doc_alignment_implementations/thompson_2021_doc_align"
LASER="/home/dstambl2/LASER"

#For kicking off data building jobs
path_extn=$1
use_attn=$2


command="python3 /home/dstambl2/doc_alignment_implementations/thompson_2021_doc_align/train_classifier_main.py --path ${path_extn} --is_training"

if [[ -n "$use_attn" ]] 
then
    echo "Using attn pooling model"
    command="${command} --use_self_attention_model" 
fi

echo $command
exec $command



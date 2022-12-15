source /home/dstambl2/miniconda3/etc/profile.d/conda.sh
conda activate "mt_dev"
PYTHONPATH="/home/dstambl2/doc_alignment_implementations/thompson_2021_doc_align"
LASER="/home/dstambl2/LASER"

path_extn=$1
neg_pair_theshold=$2
use_attention=$3
use_augment=$4
dom=$5

command="python3 /home/dstambl2/doc_alignment_implementations/thompson_2021_doc_align/train_classifier_main.py --path ${path_extn} --filter_threshold ${neg_pair_theshold}"

#For kicking off data building jobs
if [[ -n "$dom" ]] 
then
    command="${command} --domain ${dom}" 
fi

if [[ -n "$use_attention" ]] 
then
    command="${command} --use_self_attention_model" 
fi

if [[ -n "$use_augment" ]] 
then
    command="${command} --use_augment" 
fi

echo $command
exec $command


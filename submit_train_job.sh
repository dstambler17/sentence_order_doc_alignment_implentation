resources="hostname=c*,mem_free=70G,ram_free=70G"
log_dir_base="/home/dstambl2/doc_alignment_implementations/data/logs/classifier_training"

log_name=$1 #.out extension
job_name=$2
path_ext=$3
use_attn=$4

#source /home/gqin2/scripts/acquire-gpu

qsub -N $job_name  -j y -o $log_dir_base/$log_name \
    -l $resources \
    /home/dstambl2/doc_alignment_implementations/thompson_2021_doc_align/train_classifier.sh $path_ext $use_attn
echo "Kicked off training job"

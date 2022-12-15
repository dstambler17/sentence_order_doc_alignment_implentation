resources="hostname=c*,mem_free=20G,ram_free=30G"
log_dir_base="/home/dstambl2/doc_alignment_implementations/data/logs/classifier_training"

threshold=$1
path_extn=$2
log_name=$3 #.out extension
job_name=$4
use_attention=$5
use_augment=$6
dom=$7

qsub -N $job_name  -j y -o $log_dir_base/$log_name \
    -l $resources \
    /home/dstambl2/doc_alignment_implementations/thompson_2021_doc_align/data_build_train_classifier.sh $path_extn $threshold $use_attention $use_augment $dom 
echo "Kicked off training job"

# Processes all let and unzips them
if [ -z $1 ] ; then
  echo "Provide the file path"
  exit
fi

if [ -z $2 ] ; then
  echo "Provide the folder with the raw lett files"
  exit
fi

if [ -z $3 ] ; then
  echo "Provide the folder to dump processed and unzipped files to"
  exit
fi

if [ -z $4 ] ; then
  echo "Specify if file is gz or xz"
  exit
fi

if [ -z $5 ] ; then
  echo "Specify Source Lang"
  exit
fi

if [ -z $6 ] ; then
  echo "Specify Target Lang"
  exit
fi

input_path=$1"/"$2
ouput_path=$1"/"$3

gz_xz=$4
src_lang=$5
tgt_lang=$6

lett_out_path=$7


for entry in "$input_path"/*
do
  
  IFS='/' read -ra path_arr <<< "$entry" #split
  file_name="${path_arr[-1]}"


  if [ "$gz_xz" = "gz" ]; then
    lett_file_path=${ouput_path}/${file_name%%.gz}
    gzip -dkc < $entry > $lett_file_path
    procssed_format=""
    lett_writer_path=""
  elif [ "$gz_xz" = "xz" ]; then
    lett_file_path=${ouput_path}/${file_name%%.xz}
    xz -dkc < $entry > $lett_file_path
    procssed_format="--procssed_format" #Process lett files in specific way if we are feeding in xz files
    lett_writer_path="--lett_writer_path ${lett_out_path}"
  else
      echo "ERROR, ARG NOT GZ XZ"
  fi


  #echo ""
  echo $lett_file_path
  python3 /home/dstambl2/doc_alignment_implementations/thompson_2021_doc_align/scripts/process_lett_files.py \
   ${lett_file_path} \
   -slang $src_lang \
   -tlang $tgt_lang ${procssed_format} ${lett_writer_path}
  rm $lett_file_path
done

#echo ${input_path}
#echo ${ouput_path}

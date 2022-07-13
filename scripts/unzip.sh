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

input_path=$1"/"$2
ouput_path=$1"/"$3


for entry in "$input_path"/*
do
  
  IFS='/' read -ra path_arr <<< "$entry" #split
  file_name="${path_arr[-1]}"

  lett_file_path=${ouput_path}/${file_name%%.gz}

  gzip -dkc < $entry > $lett_file_path
  #gzip -d -k $entry

  #echo ""
  echo $lett_file_path
  python3 /home/dstambl2/doc_alignment_implementations/thompson_2021_doc_align/scripts/process_lett_files.py ${lett_file_path}
  rm $lett_file_path
done

#echo ${input_path}
#echo ${ouput_path}

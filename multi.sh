#!/bin/sh
while getopts d:m:z: flag
do
    case "${flag}" in
        d) data_folder=${OPTARG};;
        m) model_folder=${OPTARG};;
        z) zip_name=${OPTARG};;
    esac
done

echo "Data Folder: $data_folder";
echo "Model Folder: $model_folder";

for n in 1 2 5 10 25
do
  for i in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19
  do
    current_model_folder="$model_folder-$n-RUN-$i"
    if [ -d $current_model_folder ]
    then
      echo "Directory $current_model_folder"
    else
      echo "Start Run: $i"
      # echo "-df $data_folder -model $current_model_folder -n $n"
      python train_neural_network.py -df $data_folder"/train" -model $current_model_folder -n $n
      python test_neural_network.py -df $data_folder"/test" -model $current_model_folder
    fi
  done
done

zip -r "/media/fmuenke/8c63b673-ade7-4948-91ca-aba40636c42c/ai_models/"$zip_name $model_folder
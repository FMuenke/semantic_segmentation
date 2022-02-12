#!/bin/sh
while getopts d:m: flag
do
    case "${flag}" in
        d) data_folder=${OPTARG};;
        m) model_folder=${OPTARG};;
    esac
done

echo "Data Folder: $data_folder";
echo "Model Folder: $model_folder";


for folder in $model_folder/*;
do
  echo "$folder"
  if [ -d "$folder/overlays" ]
    then
      echo "Directory $current_model_folder"
    else
      python test_neural_network.py -df $data_folder -model "$folder"
    fi
done
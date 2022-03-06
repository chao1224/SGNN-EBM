#!/usr/bin/env bash

cd ../src

export dataset_list=(chembl_dense_10 chembl_dense_50 chembl_dense_100)
export time_list=(3 1 1)
export seed_list=(0 1 2 3 4)
export split_method_list=(random_filtered_split)
export seed_list=(0)

export epochs_list=(200)

export mtl_method_list=(uw)
for seed in "${seed_list[@]}"; do
for mtl_method in "${mtl_method_list[@]}"; do
for i in {0..0}; do
    dataset=${dataset_list["$i"]}
    time=${time_list["$i"]}

    for epochs in "${epochs_list[@]}"; do
        for split_method in "${split_method_list[@]}"; do
            export folder="$mtl_method"/"$dataset"/"$split_method"_"$epochs"/"$seed"
            echo "$folder"

            mkdir -p ../output/"$folder"
            mkdir -p ../model_weight/"$folder"

            export output_file=../output/"$folder"/output.txt
            export output_model_file=../model_weight/"$folder"/model

            sbatch --gres=gpu:v100l:1 -c 6 --mem=30G -t "$time":00:00  --account=rrg-bengioy-ad --qos=high --job-name="$mtl_method" \
            --output="$output_file" \
            ./run_main_MTL.sh \
            --mtl_method="$mtl_method" \
            --dataset="$dataset" \
            --seed="$seed" \
            --epochs="$epochs" \
            --split_method="$split_method" \
            --output_model_file="$output_model_file"

        done
    done
done
done
done

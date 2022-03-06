#!/usr/bin/env bash

cd ../src

export mtl_method=gradnorm
export split_method=random_filtered_split
export seed_list=(0 1 2 3 4)
export seed_list=(0)


############ for MTL ##########
export dataset=chembl_dense_10
export time=3
export epochs=200
export alpha=0.1

for seed in "${seed_list[@]}"; do
    export folder="$mtl_method"/"$dataset"/"$alpha"_"$split_method"_"$epochs"/"$seed"
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
    --alpha="$alpha" \
    --epochs="$epochs" \
    --split_method="$split_method" \
    --output_model_file="$output_model_file"
done





# ############ for MTL ##########
# export dataset=chembl_dense_50
# export time=1
# export epochs=200
# export alpha=0.2

# for seed in "${seed_list[@]}"; do
#     export folder="$mtl_method"/"$dataset"/"$alpha"_"$split_method"_"$epochs"/"$seed"
#     mkdir -p ../output/"$folder"
#     mkdir -p ../model_weight/"$folder"
#     export output_file=../output/"$folder"/output.txt
#     export output_model_file=../model_weight/"$folder"/model

#    sbatch --gres=gpu:v100l:1 -c 6 --mem=30G -t "$time":00:00  --account=rrg-bengioy-ad --qos=high --job-name="$mtl_method" \
#    --output="$output_file" \
#    ./run_main_MTL.sh \
#    --mtl_method="$mtl_method" \
#    --dataset="$dataset" \
#    --seed="$seed" \
#    --alpha="$alpha" \
#    --epochs="$epochs" \
#    --split_method="$split_method" \
#    --output_model_file="$output_model_file"
# done





# ############ for MTL ##########
# export dataset=chembl_dense_100
# export time=1
# export epochs=200
# export alpha=0.1

# for seed in "${seed_list[@]}"; do
#     export folder="$mtl_method"/"$dataset"/"$alpha"_"$split_method"_"$epochs"/"$seed"
#     mkdir -p ../output/"$folder"
#     mkdir -p ../model_weight/"$folder"
#     export output_file=../output/"$folder"/output.txt
#     export output_model_file=../model_weight/"$folder"/model

#    sbatch --gres=gpu:v100l:1 -c 6 --mem=30G -t "$time":00:00  --account=rrg-bengioy-ad --qos=high --job-name="$mtl_method" \
#    --output="$output_file" \
#    ./run_main_MTL.sh \
#    --mtl_method="$mtl_method" \
#    --dataset="$dataset" \
#    --seed="$seed" \
#    --alpha="$alpha" \
#    --epochs="$epochs" \
#    --split_method="$split_method" \
#    --output_model_file="$output_model_file"
# done



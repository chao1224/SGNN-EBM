#!/usr/bin/env bash

cd src


energy_function=energy_function_GNN_CE_1st_order
inference_function=GNN_1st_order_inference
gnn_energy_model=GNN_Energy_Model_1st_Order_01
seed_list=(0 1 2 3 4)


for seed in "${seed_list[@]}"; do

        mtl_method=gnn
        dataset=chembl_dense_10
        time=11
        output_folder=../checkpoint/"$mtl_method"/"$dataset"/"$seed"
        mkdir -p "$output_folder"
        rm "$output_folder"/*
        output_file="$output_folder"/output.txt
        output_model_file="$output_folder"/model

        sbatch --gres=gpu:v100l:1 -c 8 --mem=30G -t "$time":59:00 --account=rrg-bengioy-ad --qos=high --job-name=GNN_10 \
        --output="$output_file" \
        ./run_structured_learning.sh \
        --mtl_method=structured_prediction \
        --dataset="$dataset" \
        --energy_function="$energy_function" --inference_function="$inference_function" --gnn_energy_model="$gnn_energy_model" \
        --task_emb_dim=100 \
        --PPI_threshold=0.1 \
        --ebm_GNN_dim=100 \
        --ebm_GNN_layer_num=3 \
        --ebm_GNN_use_concat \
        --filling_missing_data_mode=no_filling \
        --lr_scale=1 \
        --use_batch_norm \
        --use_GCN_for_KG \
        --kg_dropout_ratio=0.2 \
        --output_model_file="$output_model_file" \
        --seed="$seed" \
        --epochs=200 \
        --batch_size=32




        mtl_method=gnn
        dataset=chembl_dense_50
        time=2
        output_folder=../checkpoint/"$mtl_method"/"$dataset"/"$seed"
        mkdir -p "$output_folder"
        output_file="$output_folder"/output.txt
        output_model_file="$output_folder"/model

        sbatch --gres=gpu:v100l:1 -c 8 --mem=30G -t "$time":59:00 --account=rrg-bengioy-ad --qos=high --job-name=GNN_50 \
        --output="$output_file" \
        ./run_structured_learning.sh \
        --mtl_method=structured_prediction \
        --dataset="$dataset" \
        --energy_function="$energy_function" --inference_function="$inference_function" --gnn_energy_model="$gnn_energy_model" \
        --task_emb_dim=50 \
        --PPI_threshold=0.9 \
        --ebm_GNN_dim=100 \
        --ebm_GNN_layer_num=3 \
        --ebm_GNN_use_concat \
        --filling_missing_data_mode=no_filling \
        --lr_scale=1 \
        --no_batch_norm \
        --use_GCN_for_KG \
        --kg_dropout_ratio=0.2 \
        --output_model_file="$output_model_file" \
        --seed="$seed" \
        --epochs=500






        mtl_method=gnn
        dataset=chembl_dense_100
        time=2
        output_folder=../checkpoint/"$mtl_method"/"$dataset"/"$seed"
        mkdir -p "$output_folder"
        output_file="$output_folder"/output.txt
        output_model_file="$output_folder"/model

        sbatch --gres=gpu:v100l:1 -c 8 --mem=30G -t "$time":59:00 --account=rrg-bengioy-ad --qos=high --job-name=GNN_100 \
        --output="$output_file" \
        ./run_structured_learning.sh \
        --mtl_method=structured_prediction \
        --dataset="$dataset" \
        --energy_function="$energy_function" --inference_function="$inference_function" --gnn_energy_model="$gnn_energy_model" \
        --task_emb_dim=50 \
        --PPI_threshold=0.1 \
        --ebm_GNN_dim=100 \
        --ebm_GNN_layer_num=5 \
        --ebm_GNN_use_concat \
        --filling_missing_data_mode=no_filling \
        --lr_scale=1 \
        --use_batch_norm \
        --use_GCN_for_KG \
        --kg_dropout_ratio=0.2 \
        --output_model_file="$output_model_file" \
        --seed="$seed" \
        --epochs=500


done

#!/usr/bin/env bash

cd src

energy_function=energy_function_GNN_EBM_NCE
inference_function=GNN_EBM_GS_inference
gnn_energy_model=GNN_Energy_Model_2nd_Order_01

seed_list=(0 1 2 3 4)


for seed in "${seed_list[@]}"; do

        mtl_method=ebm
        dataset=chembl_dense_10
        time=23
        output_folder=../checkpoint/"$mtl_method"/"$dataset"/"$seed"
        mkdir -p "$output_folder"
        output_file="$output_folder"/output.txt
        output_model_file="$output_folder"/model

        sbatch --gres=gpu:v100l:1 -c 6 --mem=30G -t "$time":59:00 --account=rrg-bengioy-ad --qos=high --job-name=EBM_10 \
        --output="$output_file" \
        ./run_SGNN_EBM.sh \
        --mtl_method=structured_prediction \
        --dataset="$dataset" \
        --energy_function="$energy_function" --inference_function="$inference_function" --gnn_energy_model="$gnn_energy_model" \
        --NCE_mode=gs \
        --use_softmax_energy \
        --task_emb_dim=100 \
        --PPI_threshold=0.1 \
        --ebm_GNN_dim=100 \
        --ebm_GNN_layer_num=3 \
        --ebm_GNN_use_concat \
        --filling_missing_data_mode=gnn \
        --lr_scale=1 \
        --use_batch_norm \
        --use_ebm_as_tilting \
        --GS_iteration=2 \
        --use_GCN_for_KG \
        --kg_dropout_ratio=0.2 \
        --batch_size=16 \
        --structured_lambda=0.1 \
        --use_PPI \
        --output_model_file="$output_model_file" \
        --seed="$seed"





        mtl_method=ebm
        dataset=chembl_dense_50
        time=2
        output_folder=../checkpoint/"$mtl_method"/"$dataset"/"$seed"
        mkdir -p "$output_folder"
        output_file="$output_folder"/output.txt
        output_model_file="$output_folder"/model

        rm -rf "$output_folder"/*

        sbatch --gres=gpu:v100l:1 -c 6 --mem=30G -t "$time":59:00 --account=rrg-bengioy-ad --qos=high --job-name=EBM_50 \
        --output="$output_file" \
        ./run_SGNN_EBM.sh \
        --mtl_method=structured_prediction \
        --dataset="$dataset" \
        --energy_function="$energy_function" --inference_function="$inference_function" --gnn_energy_model="$gnn_energy_model" \
        --NCE_mode=gs \
        --use_softmax_energy \
        --task_emb_dim=50 \
        --PPI_threshold=0.1 \
        --ebm_GNN_dim=100 \
        --ebm_GNN_layer_num=3 \
        --ebm_GNN_use_concat \
        --filling_missing_data_mode=gnn \
        --lr_scale=1 \
        --use_batch_norm \
        --use_ebm_as_tilting \
        --GS_iteration=2 \
        --use_GCN_for_KG \
        --kg_dropout_ratio=0.2 \
        --structured_lambda=0.1 \
        --use_PPI \
        --output_model_file="$output_model_file" \
        --seed="$seed"





        mtl_method=ebm
        dataset=chembl_dense_100
        time=2
        output_folder=../checkpoint/"$mtl_method"/"$dataset"/"$seed"
        mkdir -p "$output_folder"
        output_file="$output_folder"/output.txt
        output_model_file="$output_folder"/model

        sbatch --gres=gpu:v100l:1 -c 6 --mem=30G -t "$time":59:00 --account=rrg-bengioy-ad --qos=high --job-name=EBM_100 \
        --output="$output_file" \
        ./run_SGNN_EBM.sh \
        --mtl_method=structured_prediction \
        --dataset="$dataset" \
        --energy_function="$energy_function" --inference_function="$inference_function" --gnn_energy_model="$gnn_energy_model" \
        --NCE_mode=gs \
        --use_softmax_energy \
        --task_emb_dim=50 \
        --PPI_threshold=0.1 \
        --ebm_GNN_dim=100 \
        --ebm_GNN_layer_num=3 \
        --ebm_GNN_use_concat \
        --filling_missing_data_mode=gnn \
        --lr_scale=1 \
        --use_batch_norm \
        --use_ebm_as_tilting \
        --GS_iteration=2 \
        --use_GCN_for_KG \
        --kg_dropout_ratio=0.2 \
        --structured_lambda=0.1 \
        --use_PPI \
        --output_model_file="$output_model_file" \
        --seed="$seed"

done

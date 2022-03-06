energy_function=energy_function_GNN_EBM_NCE
inference_function=GNN_EBM_GS_inference
gnn_energy_model=GNN_Energy_Model_2nd_Order_01





mtl_method=ebm
dataset=chembl_dense_10
for seed in {0..4}; do
   python eval_SGNN_EBM.py \
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
   --batch_size=32 \
   --structured_lambda=0.1 \
   --use_PPI \
   --seed="$seed" \
   --output_model_file="$mtl_method"/"$dataset"
done




mtl_method=ebm
dataset=chembl_dense_50
for seed in {0..4}; do
     python eval_SGNN_EBM.py \
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
   --seed="$seed" \
   --output_model_file="$mtl_method"/"$dataset"
done




mtl_method=ebm
dataset=chembl_dense_100
for seed in {0..4}; do
      python eval_SGNN_EBM.py \
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
    --seed="$seed" \
    --output_model_file="$mtl_method"/"$dataset"
done



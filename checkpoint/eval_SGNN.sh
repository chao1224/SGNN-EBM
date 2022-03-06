energy_function=energy_function_GNN_CE_1st_order
inference_function=GNN_1st_order_inference
gnn_energy_model=GNN_Energy_Model_1st_Order_01


mtl_method=gnn
dataset=chembl_dense_10
python eval_SGNN.py \
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
--batch=32 \
--output_model_file="$mtl_method"/"$dataset"




mtl_method=gnn
dataset=chembl_dense_50
python eval_SGNN.py \
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
--output_model_file="$mtl_method"/"$dataset"







mtl_method=gnn
dataset=chembl_dense_100
python eval_SGNN.py \
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
--output_model_file="$mtl_method"/"$dataset"




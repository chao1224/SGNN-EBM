import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--eval_every_n_epochs', type=int, default=5)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--lr_scale', type=float, default=1)
parser.add_argument('--use_batch_norm', dest='batch_norm', action='store_true')
parser.add_argument('--no_batch_norm', dest='batch_norm', action='store_false')
parser.set_defaults(batch_norm=False)
parser.add_argument('--decay', type=float, default=0)
parser.add_argument('--num_layer', type=int, default=5)
parser.add_argument('--emb_dim', type=int, default=300)
parser.add_argument('--dropout_ratio', type=float, default=0.2)
parser.add_argument('--mtl_method', type=str, default='structured_prediction',
                    choices=['stl', 'mtl', 'uw', 'gradnorm', 'dwa', 'lbtw', 'structured_prediction'])
parser.add_argument('--graph_pooling', type=str, default='mean')
parser.add_argument('--JK', type=str, default='last')
parser.add_argument('--dataset', type=str, default='chembl_dense_100',
                    choices=['chembl_dense_10', 'chembl_dense_50', 'chembl_dense_100'])
parser.add_argument('--gnn_type', type=str, default='gin')
parser.add_argument('--input_model_file', type=str, default='')
parser.add_argument('--output_model_file', type=str, default='')
parser.add_argument('--input_y_score_file', type=str, default='')
parser.add_argument('--output_y_score_file', type=str, default='')
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--split_method', type=str, default='random_filtered_split',
                    choices=['random_split', 'random_filtered_split', 'pre_split', 'scaffold_split', 'cluster_split'])
parser.add_argument('--pretrain_epochs', type=int, default=100)
parser.add_argument('--pretrain_lr', type=float, default=1e-3)

# for single-task learning
parser.add_argument('--task_batch_size', type=int, default=16)
parser.add_argument('--task_batch_id', type=int, default=0)

# for uncertainty-weighted MTL
parser.add_argument('--lr-decay', type=float, default=2e-3)

# for GradNorm & LBTW
parser.add_argument('--alpha', type=float, default=0.1)

# for DWA
parser.add_argument('--dwa_T', type=float, default=2.)

# for PPI
parser.add_argument('--task_emb_dim', type=int, default=10)
parser.add_argument('--use_PPI', dest='use_PPI', action='store_true')
parser.add_argument('--no_PPI', dest='use_PPI', action='store_false')
parser.set_defaults(use_PPI=False)
parser.add_argument('--use_PPI_every_N_step', type=int, default=100)
parser.add_argument('--PPI_pretrained_epochs', type=int, default=100)
parser.add_argument('--PPI_threshold', type=float, default=0.1)
parser.add_argument('--neg_sample_size', type=int, default=5)
parser.add_argument('--neg_sample_exponent', type=float, default=0.75)
parser.add_argument('--kg_dropout_ratio', type=float, default=0.2)

# for MTL-pretraining
parser.add_argument('--MTL_pretrained_epochs', type=int, default=100)

# for Structured Prediction
parser.add_argument('--energy_function', type=str, default='energy_function_CD_AA',
                    choices=[
                        'energy_function_CD_AA',
                        'energy_function_CD_GS',

                        'energy_function_GNN_CE_1st_order',
                        'energy_function_GNN_CE_2nd_order',
                        'energy_function_GNN_EBM_NCE',
                        'energy_function_GNN_EBM_CD_GS',

                        'energy_function_GNN_EBM_CE_2nd_order_Binary_Task'
                    ])

parser.add_argument('--energy_CD_loss', type=str, default='ce',
                    choices=['ce', 'weighted_ce', 'raw', 'smoothing'])
parser.add_argument('--inference_function', type=str, default='amortized_mean_field_inference_first_order',
                    choices=[
                        'amortized_mean_field_inference_first_order',
                        'amortized_mean_field_inference_second_order',
                        'amortized_mean_field_inference_label_propagation_first_order',
                        'mean_field_variational_inference',
                        'GS_inference',
                        'SGLD_inference',

                        'GNN_1st_order_inference',
                        'GNN_EBM_mean_field_variational_inference',
                        'GNN_EBM_GS_inference',

                        'GNN_EBM_1st_order_inference_Binary_Task',
                        'full',
                    ])
parser.add_argument('--gnn_energy_model', type=str, default='GNN_Energy_Model_1st_Order_01',
                    choices=[
                        'GNN_Energy_Model_1st_Order_01',
                        'GNN_Energy_Model_1st_Order_02',
                        'GNN_Energy_Model_2nd_Order_01',
                        'GNN_Energy_Model_2nd_Order_02',
                        'GNN_Energy_Model_2nd_Order_03',
                    ])
parser.add_argument('--use_ebm_as_tilting', dest='ebm_as_tilting', action='store_true')
parser.add_argument('--no_ebm_as_tilting', dest='ebm_as_tilting', action='store_false')
parser.set_defaults(ebm_as_tilting=False)
parser.add_argument('--use_softmax_energy', dest='softmax_energy', action='store_true')
parser.add_argument('--no_softmax_energy', dest='softmax_energy', action='store_false')
parser.set_defaults(softmax_energy=False)
parser.add_argument('--NCE_mode', type=str, default='uniform',
                    choices=[
                        'uniform', 'gs', 'mean_field', 'ce', 'statistics',
                        'mtl', 'uw', 'gradnorm', 'dwa', 'lbtw', 'gnn', 'ebm',
                    ])
parser.add_argument('--structured_lambda', type=float, default=1.)
parser.add_argument('--second_order_lr_weight', type=float, default=0.001)
parser.add_argument('--Gibbs_init', action='store_true', help='use pretrained MTL model to initilize the Gibbs distribution')
parser.add_argument('--amortized_logits_transform_to_confidence', dest='amortized_logits_transform_to_confidence', action='store_true')
parser.add_argument('--no_amortized_logits_transform_to_confidence', dest='amortized_logits_transform_to_confidence', action='store_false')
parser.set_defaults(amortized_logits_transform_to_confidence=True)
parser.add_argument('--use_GCN_for_KG', dest='use_GCN_for_KG', action='store_true')
parser.add_argument('--no_GCN_for_KG', dest='use_GCN_for_KG', action='store_false')
parser.set_defaults(use_GCN_for_KG=False)
parser.add_argument('--filling_missing_data_mode', type=str, default='no_filling',
                    choices=['no_filling', 'mtl', 'uw', 'gradnorm', 'dwa', 'lbtw', 'gnn', 'ebm'])
parser.add_argument('--filling_missing_data_fine_tuned_epoch', type=int, default=1000000)
parser.add_argument('--ebm_GNN_dim', type=int, default=100)
parser.add_argument('--ebm_GNN_layer_num', type=int, default=3)  # starting with 1, the input h^0 also accounts
parser.add_argument('--ebm_GNN_use_concat', dest='ebm_GNN_use_concat', action='store_true')
parser.add_argument('--ebm_GNN_no_concat', dest='ebm_GNN_use_concat', action='store_false')
parser.set_defaults(ebm_GNN_use_concat=False)

# for Mean-Field (Variational) Inference
# for Gibbs Sampling
parser.add_argument('--GS_iteration', type=int, default=2)  # will rename to sth like mean_field_n_iteration
parser.add_argument('--MFVI_iteration', type=int, default=2)  # will rename to sth like mean_field_n_iteration

parser.add_argument('--GS_learning', type=str, default='last', choices=['last', 'average'])
parser.add_argument('--GS_inference', type=str, default='last', choices=['last', 'average'])

parser.add_argument('--task_sim_method', type=str, default='cosine_sim',
                    choices=['cosine_sim', 'loss_ratio'])
args = parser.parse_args()

if args.energy_function == 'energy_function_GNN_CE_1st_order':
    assert args.inference_function == 'GNN_1st_order_inference'
    assert args.gnn_energy_model in ['GNN_Energy_Model_1st_Order_01', 'GNN_Energy_Model_1st_Order_02']

if args.energy_function == 'energy_function_GNN_CE_2nd_order':
    assert args.inference_function in [
        'GNN_1st_order_inference', 'GNN_EBM_mean_field_variational_inference', 'GNN_EBM_GS_inference'
    ]

if args.energy_function == 'energy_function_GNN_EBM_NCE':
    assert args.inference_function in [
        'GNN_1st_order_inference', 'GNN_EBM_mean_field_variational_inference', 'GNN_EBM_GS_inference'
    ]
    assert not args.NCE_mode in ['uniform', 'gs'] or args.softmax_energy

if args.energy_function == 'energy_function_GNN_EBM_CD_GS':
    assert args.inference_function in [
        'GNN_1st_order_inference', 'GNN_EBM_mean_field_variational_inference', 'GNN_EBM_GS_inference'
    ]

if args.energy_function == 'energy_function_GNN_CE_2nd_order_Binary_Task':
    assert args.inference_function in [
        'GNN_EBM_1st_order_inference_Binary_Task'
    ]


GIBBS_ITER = 1
MEAN_FIELD_ITER = 20
EPS = 1e-8
########## Bunch of classification tasks ##########
if args.dataset == 'chembl_dense_10' or args.dataset == 'chembl_dense_10_ablation':
    args.num_tasks = 382
elif args.dataset == 'chembl_dense_50' or args.dataset == 'chembl_dense_50_ablation':
    args.num_tasks = 152
elif args.dataset == 'chembl_dense_100' or args.dataset == 'chembl_dense_100_ablation':
    args.num_tasks = 132
else:
    raise ValueError('Invalid dataset name.')

if __name__ == "__main__":
    print(args.num_tasks)
    print('arguments\t', args)

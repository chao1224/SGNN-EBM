# Structured Multi-task Learning for Molecular Property Prediction

**AISTATS 2022**

Authors: Shengchao Liu, Meng Qu, Zuobai Zhang, Huiyu Cai, Jian Tang

[[Project Page](https://chao1224.github.io/SGNN-EBM)]
[[Paper]()]
[[ArXiv]()]
[[Code](https://github.com/chao1224/SGNN-EBM)]
[[NeurIPS AI4Science Workshop 2021](https://openreview.net/forum?id=6cWgY5Epwzo)]

This repository provides the source code for the AISTATS'22 paper **Structured Multi-task Learning for Molecular Property Prediction**, with the following contributions:
1. To our best knowledge, we are the **first** to propose a new research problem: multi-task learning with an explicit task relation graph;
2. We construct a domain-specific multi-task dataset with relation graph for drug discovery;
3. We propose **state graph neural network-energy based model (SGNN-EBM)** for task structured modeling in both the latent and output space.

<p align="center">
  <image src="fig/pipeline.png" height="60px"/> 
</p>


## Baselines
For implementation, this repository also provides the following multi-task learning baselines:
- standard single-task learning (STL)
- standard multi-task learning (MTL)
- [Uncertainty Weighing (UW)](https://openaccess.thecvf.com/content_cvpr_2018/papers/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.pdf)
- [GradNorm](https://proceedings.mlr.press/v80/chen18a.html)
- [Dynamic Weight Average (DWA)](https://openaccess.thecvf.com/content_CVPR_2019/papers/Liu_End-To-End_Multi-Task_Learning_With_Attention_CVPR_2019_paper.pdf)
- [Loss-Balanced Task Weighting (LBTW)](https://ojs.aaai.org//index.php/AAAI/article/view/5125)


## Environments
Below is environment built with [pytorch-geometric](https://github.com/pyg-team/pytorch_geometric).
In the future, we will merge it into the [TorchDrug](https://github.com/DeepGraphLearning/torchdrug) package.

```
conda create -n SGNN_EBM python=3.7
conda activate SGNN_EBM
conda install -y -c pytorch pytorch=1.6.0 torchvision
conda install -y matplotlib
conda install -y scikit-learn
conda install -y -c rdkit rdkit=2019.03.1.0
conda install -y -c anaconda beautifulsoup4
conda install -y -c anaconda lxml

wget https://data.pyg.org/whl/torch-1.6.0%2Bcu102/torch_sparse-0.6.9-cp37-cp37m-linux_x86_64.whl
wget https://data.pyg.org/whl/torch-1.6.0%2Bcu102/torch_scatter-2.0.6-cp37-cp37m-linux_x86_64.whl
wget https://data.pyg.org/whl/torch-1.6.0%2Bcu102/torch_spline_conv-1.2.1-cp37-cp37m-linux_x86_64.whl
wget https://data.pyg.org/whl/torch-1.6.0%2Bcu102/torch_cluster-1.5.9-cp37-cp37m-linux_x86_64.whl
pip install torch_sparse-0.6.9-cp37-cp37m-linux_x86_64.whl
pip install torch_scatter-2.0.6-cp37-cp37m-linux_x86_64.whl
pip install torch_spline_conv-1.2.1-cp37-cp37m-linux_x86_64.whl
pip install torch_cluster-1.5.9-cp37-cp37m-linux_x86_64.whl
pip install torch-geometric==1.6.*
```


## Dataset

In this work, we propose a novel dataset with explicit task relation.
Basically it is a molecule property task dataset, where the *task* refers to a binary classification problem on a ChEMBL assay. Each task measures certain biological effects of molecules, *e.g.*, toxicity, inhibition or activation of proteins or whole cellular processes, etc. We focus on tasks that target at proteins. Then we extract the task relation by aggregating the protein-protein interaction (PPI, like String dataset) accordingly.

<p align="center">
  <image src="fig/dataset_preprocess.png" height="60px"/> 
</p>

For the detailed pre-processing steps, please check [this instruction](https://github.com/chao1224/SGNN-EBM/tree/init/datasets/README.md).


## Structured Multi-Task Learning: SGNN-EBM

### Evaluation on the pre-trained models

We also provide the pre-trained model weights and evaluation scripts accordingly.
First you can download the checkpoints [here]().
All the optimal hyper-parameters are provided in the bash scripts.

```bash
cd checkpoint

bash eval_SGNN.sh > eval_SGNN.out
bash eval_SGNN_EBM.sh > eval_SGNN_EBM.out
```

### Training from scratch

Here we provide the script for training the SGNN-EBM (adaptive with pre-trained SGNN) on ChEMBL-10 dataset.
```
cd src

energy_function=energy_function_GNN_CE_1st_order
inference_function=GNN_1st_order_inference
gnn_energy_model=GNN_Energy_Model_1st_Order_01
mtl_method=gnn
dataset=chembl_dense_10
seed=0

python main_SGNN_EBM.py \
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
--seed="$seed"
# --output_model_file="$mtl_method"/"$dataset"
```

Next we provide the script for training the SGNN-EBM (adaptive with pre-trained SGNN) on ChEMBL-10 dataset.
Note that the pre-trained SGNN models is required (either using last script or from the pre-trained weights).

```
cd src

energy_function=energy_function_GNN_EBM_NCE
inference_function=GNN_EBM_GS_inference
gnn_energy_model=GNN_Energy_Model_2nd_Order_01
mtl_method=ebm
dataset=chembl_dense_10
seed=0

python main_SGNN_EBM.py \
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
--seed="$seed"
#--output_model_file="$mtl_method"/"$dataset"
```

## Cite us

```
@inproceedings{liu2022multi,
    title={Structured Multi-task Learning for Molecular Property Prediction},
    author={Liu, Shengchao and Qu, Meng and Zhang, Zuobai and Cai, Huiyu and Tang, Jian},
    booktitle={AISTATS},
    year={2022}
}
```
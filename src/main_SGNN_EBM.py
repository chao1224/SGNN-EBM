import time
import copy
import os
import random
import numpy as np
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import SubsetRandomSampler
from torch_geometric.data import DataLoader

from splitters import random_split, random_filtered_split, random_scaffold_split
from datasets import MoleculeDataset, PPI_dataset
from models import (
    GNN_graphpred,
    GCNNet, TaskEmbeddingModel, TaskEmbeddingModel_BinaryEmbedding,
    MoleculeTaskPredictionModel, MoleculeTaskTaskPredictionModel, PairwiseTaskPredictionModel,
    GNN_Energy_Model_1st_Order_01, GNN_Energy_Model_1st_Order_02, GNN_Energy_Model_2nd_Order_01, GNN_Energy_Model_2nd_Order_02, GNN_Energy_Model_2nd_Order_03, NCE_C_Parameter
    )

from configure import *
from Structured_MTL_EBM_support import *
from Structured_MTL_SGNN_EBM_support import *


def get_task_representation(task_embedding_model, kg_model, task_X):
    task_repr = task_embedding_model(task_X)
    if args.use_GCN_for_KG:
        task_repr = kg_model(task_repr, task_edge)
    return task_repr


def train_PPI(args, device, loader, task_relation_optimizer):
    task_embedding_model.train()
    if args.use_GCN_for_KG:
        kg_model.train()
    task_relation_matching_model.train()

    task_loss_accum, task_acc_accum = 0, 0
    for step, batch in enumerate(loader):
        task_repr = get_task_representation(task_embedding_model, kg_model, task_X)
        node0_pos, node1_pos, node0_neg, node1_neg = \
            batch[0].to(device), batch[1].to(device), batch[2].view(-1).to(device), batch[3].view(-1).to(device)
        N = len(node0_pos)

        pos_pred = task_relation_matching_model(task_repr[node0_pos], task_repr[node1_pos])
        neg_pred = task_relation_matching_model(task_repr[node0_neg], task_repr[node1_neg])

        pred = torch.cat([pos_pred, neg_pred], dim=0)
        label = torch.cat([torch.ones_like(pos_pred), torch.zeros_like(neg_pred)], dim=0)
        task_acc = torch.sum((torch.sigmoid(pred) >= 0.5) == (label > 0.5)).float() / ((1 + args.neg_sample_size) * N)
        task_loss = task_relation_criterion(pred, label)

        task_relation_optimizer.zero_grad()
        task_loss.backward()
        task_relation_optimizer.step()

        task_loss_accum += task_loss.detach().item()
        task_acc_accum += task_acc.detach().item()

    print('Task Loss: {:.5f}, Task Acc: {:.5f}'.format(
        task_loss_accum/len(loader), task_acc_accum/len(loader)
    ))


def train_MTL(args, model, readout_model, device, loader, optimizer):
    model.train()
    readout_model.train()

    loss_acc = 0
    for step, batch in enumerate(loader):
        batch = batch.to(device)
        graph_repr = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch) # graph_repr:  batch_size * emb_dim
        y_pred = readout_model(graph_repr)  # batch_size * num_task

        y_true = batch.y.view(y_pred.shape).to(torch.float64)
        optimizer.zero_grad()

        # whether y is non-null or not.
        is_valid = y_true**2 > 0
        # loss matrix
        loss_mat = mtl_criterion(y_pred.double(), (y_true+1)/2)
        # loss matrix after removing null target
        loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))
        # loss for each task, some task may have 0 valid labels
        valid_task_count_list = is_valid.sum(0)
        loss_each_task_list = (loss_mat.sum(0)) / (valid_task_count_list+EPS)

        loss = loss_each_task_list.sum() / (valid_task_count_list > 0).sum()
        loss.backward()

        optimizer.step()

        loss_acc += loss.detach().item()
    print('Loss: ', loss_acc / len(loader))
    return


def train(args, device, loader, optimizer):
    model.train()
    task_embedding_model.train()
    first_order_prediction_model.train()
    if second_order_prediction_model is not None:
        second_order_prediction_model.train()
    if GNN_energy_model is not None:
        GNN_energy_model.train()
    if args.use_GCN_for_KG:
        kg_model.train()

    first_order_energy_loss_accum, second_order_energy_loss_accum = 0, 0
    for step, batch in enumerate(loader):
        batch = batch.to(device)
        graph_repr = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        task_repr = get_task_representation(task_embedding_model, kg_model, task_X)

        B = len(graph_repr)
        y_true = batch.y.view(B, args.num_tasks).to(torch.float64)

        if args.energy_function in ['energy_function_CD_AA', 'energy_function_CD_GS']:
            _, _, first_order_energy_loss, second_order_energy_loss = energy_function(
                first_order_prediction_model=first_order_prediction_model,
                second_order_prediction_model=second_order_prediction_model,
                graph_repr=graph_repr, task_repr=task_repr, y_true=y_true, task_edge=task_edge,
                first_order_label_weights=first_order_label_weights, second_order_label_weights=second_order_label_weights,
                prior_prediction=prior_prediction, id=batch.id,
                args=args)
            energy_loss = first_order_energy_loss + args.structured_lambda * second_order_energy_loss
            first_order_energy_loss_accum += first_order_energy_loss.detach().item()
            second_order_energy_loss_accum += second_order_energy_loss.detach().item()

        elif args.energy_function in [
            'energy_function_GNN_CE_1st_order',
            'energy_function_GNN_EBM_NCE',
        ]:
            energy_loss = energy_function(
                first_order_prediction_model=first_order_prediction_model,
                second_order_prediction_model=second_order_prediction_model,
                GNN_energy_model=GNN_energy_model,
                prediction_function=prediction_function,
                graph_repr=graph_repr, task_repr=task_repr, y_true=y_true, task_edge=task_edge,
                NCE_C_param=NCE_C_param, prior_prediction=prior_prediction, id=batch.id,
                prior_prediction_logits=prior_prediction_logits,
                args=args)
            first_order_energy_loss_accum += energy_loss.detach().item()

        elif args.energy_function in [
            'energy_function_GNN_CE_2nd_order',
            'energy_function_GNN_EBM_CD_GS',
            'energy_function_GNN_EBM_CE_2nd_order_Binary_Task'
        ]:
            _, _, first_order_energy_loss, second_order_energy_loss = energy_function(
                first_order_prediction_model=first_order_prediction_model,
                second_order_prediction_model=second_order_prediction_model,
                GNN_energy_model=GNN_energy_model,
                prediction_function=prediction_function,
                graph_repr=graph_repr, task_repr=task_repr, y_true=y_true, task_edge=task_edge,
                first_order_label_weights=first_order_label_weights, second_order_label_weights=second_order_label_weights,
                prior_prediction=prior_prediction, id=batch.id,
                args=args)
            energy_loss = first_order_energy_loss + args.structured_lambda * second_order_energy_loss
            first_order_energy_loss_accum += first_order_energy_loss.detach().item()
            second_order_energy_loss_accum += second_order_energy_loss.detach().item()

        else:
            raise ValueError('Energy function {} not included.'.format(args.energy_function))


        optimizer.zero_grad()
        energy_loss.backward()
        optimizer.step()

    print('1st-order energy loss: {:.5f}\t2nd-order energy loss: {:.5f}'.format(
        first_order_energy_loss_accum / len(loader), second_order_energy_loss_accum / len(loader)
    ))


def eval(args, model, device, loader, evaluation_mode):
    model.eval()
    task_embedding_model.eval()
    first_order_prediction_model.eval()
    if second_order_prediction_model is not None:
        second_order_prediction_model.eval()
    if GNN_energy_model is not None:
        GNN_energy_model.eval()
    if args.use_GCN_for_KG:
        kg_model.eval()

    id_list = []
    y_true_list = []
    y_pred_list = []

    with torch.no_grad():
        task_repr = get_task_representation(task_embedding_model, kg_model, task_X)

        for step, batch in enumerate(loader):
            batch = batch.to(device)
            graph_repr = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            id_list.append(batch.id.cpu())

            B = len(graph_repr)
            y_true = batch.y.view(B, args.num_tasks).float()

            if args.inference_function in [
                'GNN_1st_order_inference',
                'GNN_EBM_mean_field_variational_inference',
                'GNN_EBM_GS_inference',
                'GNN_EBM_1st_order_inference_Binary_Task',
            ]:
                y_pred = inference_function(
                    first_order_prediction_model=first_order_prediction_model,
                    second_order_prediction_model=second_order_prediction_model,
                    GNN_energy_model=GNN_energy_model,
                    prediction_function=prediction_function,
                    graph_repr=graph_repr, task_repr=task_repr, y_true=y_true, task_edge=task_edge,
                    prior_prediction=prior_prediction, id=batch.id,
                    prior_prediction_logits=prior_prediction_logits,
                    args=args)
            else:
                y_pred = inference_function(
                    first_order_prediction_model=first_order_prediction_model,
                    second_order_prediction_model=second_order_prediction_model,
                    energy_function=energy_function,
                    graph_repr=graph_repr, task_repr=task_repr, y_true=y_true, task_edge=task_edge,
                    first_order_label_weights=first_order_label_weights, second_order_label_weights=second_order_label_weights,
                    args=args)

            y_true_list.append(y_true.cpu())
            y_pred_list.append(y_pred.cpu())

        id_list = torch.cat(id_list, dim=0).numpy()
        y_true_list = torch.cat(y_true_list, dim=0).numpy()
        y_pred_list = torch.cat(y_pred_list, dim=0).numpy()

        roc_list = []
        invalid_count = 0
        for i in range(y_true_list.shape[1]):
            # AUC is only defined when there is at least one positive data.
            if np.sum(y_true_list[:, i] == 1) > 0 and np.sum(y_true_list[:, i] == -1) > 0:
                is_valid = y_true_list[:, i] ** 2 > 0
                roc_list.append(roc_auc_score((y_true_list[is_valid, i] + 1) / 2, y_pred_list[is_valid, i]))
            else:
                invalid_count += 1

        print('Invalid task count:\t', invalid_count)

        if len(roc_list) < y_true_list.shape[1]:
            print('Some target is missing!')
            print('Missing ratio: %f' % (1 - float(len(roc_list)) / y_true_list.shape[1]))

        roc_list = np.array(roc_list)
        roc_value = np.mean(roc_list)
        print(f'{evaluation_mode}\tROC\t{roc_value}')
    return roc_value, y_true_list, y_pred_list, id_list


def extract_prior_distribution_from_current_model(args, loader_list):
    model.eval()
    task_embedding_model.eval()
    first_order_prediction_model.eval()
    if second_order_prediction_model is not None:
        second_order_prediction_model.eval()
    if GNN_energy_model is not None:
        GNN_energy_model.eval()
    if args.use_GCN_for_KG:
        kg_model.eval()

    id_list = []
    y_pred_list = []

    with torch.no_grad():
        task_repr = get_task_representation(task_embedding_model, kg_model, task_X)

        for loader in loader_list:
            for step, batch in enumerate(loader):
                batch = batch.to(device)
                graph_repr = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                id_list.append(batch.id.cpu())
                B = len(graph_repr)
                y_true = batch.y.view(B, args.num_tasks).float()
                if args.inference_function in [
                    'GNN_1st_order_inference',
                    'GNN_EBM_mean_field_variational_inference',
                    'GNN_EBM_GS_inference',
                    'GNN_EBM_1st_order_inference_Binary_Task',
                ]:
                    y_pred = inference_function(
                        first_order_prediction_model=first_order_prediction_model,
                        second_order_prediction_model=second_order_prediction_model,
                        GNN_energy_model=GNN_energy_model,
                        prediction_function=prediction_function,
                        graph_repr=graph_repr, task_repr=task_repr, y_true=y_true, task_edge=task_edge,
                        prior_prediction=prior_prediction, id=batch.id,
                        prior_prediction_logits=prior_prediction_logits,
                        args=args)
                else:
                    y_pred = inference_function(
                        first_order_prediction_model=first_order_prediction_model,
                        second_order_prediction_model=second_order_prediction_model,
                        energy_function=energy_function,
                        graph_repr=graph_repr, task_repr=task_repr, y_true=y_true, task_edge=task_edge,
                        first_order_label_weights=first_order_label_weights, second_order_label_weights=second_order_label_weights,
                        args=args)
                y_pred_list.append(y_pred.cpu())

        id_list = torch.cat(id_list, dim=0).numpy()
        y_pred_list = torch.cat(y_pred_list, dim=0).numpy()
        N, T = y_pred_list.shape

        prior_pred, prior_pred_logits = [None for _ in range(N)], [None for _ in range(N)]
        for id, pred in zip(id_list, y_pred_list):
            prior_pred[id] = pred > 0
            prior_pred_logits[id] = pred
        prior_pred = torch.LongTensor(prior_pred).to(args.device)
        prior_pred_logits = torch.FloatTensor(prior_pred_logits).to(args.device)
        return prior_pred, prior_pred_logits


def extract_prior_distribution_from_pretrained_model(args):
    def load(dir_, mode):
        data = np.load('{}/{}.npz'.format(dir_, mode))
        y_pred_list = data['y_pred_list']
        id_list = data['id_list']
        return y_pred_list, id_list

    filling_mode = args.filling_missing_data_mode
    dir_ = '../best/{}/{}/{}'.format(filling_mode, args.dataset, args.seed)
    y_pred_train, id_train = load(dir_, 'train')
    y_pred_valid, id_valid = load(dir_, 'valid')
    y_pred_test, id_test = load(dir_, 'test')

    assert sorted(id_train) == sorted(train_indices)
    assert sorted(id_valid) == sorted(valid_indices)
    assert sorted(id_test) == sorted(test_indices)

    y_pred_list = np.concatenate([y_pred_train, y_pred_valid, y_pred_test], axis=0)
    id_list = np.concatenate([id_train, id_valid, id_test], axis=0)
    N, T = y_pred_list.shape

    prior_pred, prior_pred_logits = [None for _ in range(N)], [None for _ in range(N)]
    for id, pred in zip(id_list, y_pred_list):
        prior_pred[id] = pred > 0
        prior_pred_logits[id] = pred
    prior_pred = torch.LongTensor(prior_pred).to(args.device)
    prior_pred_logits = torch.FloatTensor(prior_pred_logits).to(args.device)
    return prior_pred, prior_pred_logits


def save_model(save_best=False):
    if not args.output_model_file == '':
        saver = {
            'model': model.state_dict(),
            'task_embedding_model': task_embedding_model.state_dict(),
            'first_order_prediction_model': first_order_prediction_model.state_dict(),
        }
        if second_order_prediction_model is not None:
            saver['second_order_prediction_model'] = second_order_prediction_model.state_dict()
        if GNN_energy_model is not None:
            saver['GNN_energy_model'] = GNN_energy_model.state_dict()
        if args.use_GCN_for_KG:
            saver['kg_model'] = kg_model.state_dict()
        if readout_model is not None:
            saver['readout_model'] = readout_model.state_dict()
        if task_relation_matching_model is not None:
            saver['task_relation_matching_model'] = task_relation_matching_model.state_dict()
        if NCE_C_param is not None:
            saver['NCE_C_param'] = NCE_C_param.state_dict()

        if save_best:
            print('saving best model...')
            torch.save(saver, args.output_model_file + '_best.pth')
        else:
            torch.save(saver, args.output_model_file + '_final.pth')
    return


if __name__ == '__main__':
    print(args.num_tasks)
    print('arguments\t', args)

    assert args.mtl_method == 'structured_prediction'
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device('cuda:' + str(args.device)) if torch.cuda.is_available() else torch.device('cpu')
    args.device = device

    ########## Set up molecule dataset ##########
    root = '../datasets/' + args.dataset
    dataset = MoleculeDataset(root=root, dataset=args.dataset)
    if args.split_method == 'random_split':
        train_indices, valid_indices, test_indices = random_split(
            dataset, frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=args.seed)
    elif args.split_method == 'random_filtered_split':
        train_indices, valid_indices, test_indices = random_filtered_split(
            dataset, frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=args.seed)
    elif args.split_method == 'scaffold_split':
        train_indices, valid_indices, test_indices = random_scaffold_split(
            dataset, frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=args.seed,
            col_name='scaffold_smiles', root=root
        )
    elif args.split_method == 'cluster_split':
        train_indices, valid_indices, test_indices = random_scaffold_split(
            dataset, frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=args.seed,
            col_name='clusterID', root=root
        )
    else:
        raise ValueError('Split method {} not included.'.format(args.split_method))
    print(f'train: {len(train_indices)}\tvalid: {len(valid_indices)}\ttest: {len(test_indices)}')
    print('first train indices\t', train_indices[:10], train_indices[-10:])
    print('first valid indices\t', valid_indices[:10], valid_indices[-10:])
    print('first test indices\t', test_indices[:10], test_indices[-10:])

    train_sampler = SubsetRandomSampler(train_indices)
    train_dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.num_workers)
    valid_dataloader = test_dataloader = None
    if len(valid_indices) > 0:
        valid_sampler = SubsetRandomSampler(valid_indices)
        valid_dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=valid_sampler, num_workers=args.num_workers)
    if len(test_indices) > 0:
        test_sampler = SubsetRandomSampler(test_indices)
        test_dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=test_sampler, num_workers=args.num_workers)

    ########## Set up energy function ##########
    prediction_function = None
    if args.energy_function == 'energy_function_CD_AA':
        energy_function = energy_function_CD_AA
    elif args.energy_function == 'energy_function_CD_GS':
        energy_function = energy_function_CD_GS
    elif args.energy_function == 'energy_function_GNN_CE_1st_order':
        energy_function = energy_function_GNN_CE_1st_order
        prediction_function = get_GNN_prediction_1st_order_prediction
    elif args.energy_function == 'energy_function_GNN_CE_2nd_order':
        energy_function = energy_function_GNN_CE_2nd_order
        prediction_function = get_GNN_prediction_2nd_order_prediction
    elif args.energy_function == 'energy_function_GNN_EBM_NCE':
        energy_function = energy_function_GNN_EBM_NCE
        prediction_function = get_GNN_prediction_2nd_order_prediction
    elif args.energy_function == 'energy_function_GNN_EBM_CD_GS':
        energy_function = energy_function_GNN_EBM_CD_GS
        prediction_function = get_GNN_prediction_2nd_order_prediction
    elif args.energy_function == 'energy_function_GNN_EBM_CE_2nd_order_Binary_Task':
        energy_function = energy_function_GNN_EBM_CE_2nd_order_Binary_Task
        prediction_function = get_GNN_prediction_Binary_Task_Embedding_CE
    else:
        raise ValueError('Energy function {} not included.'.format(args.energy_function))

    ########## Set up inference function ##########
    if args.inference_function == 'amortized_mean_field_inference_first_order':
        inference_function = amortized_mean_field_inference_first_order
    elif args.inference_function == 'amortized_mean_field_inference_second_order':
        inference_function = amortized_mean_field_inference_second_order
    elif args.inference_function == 'amortized_mean_field_inference_label_propagation_first_order':
        inference_function = amortized_mean_field_inference_label_propagation_first_order
    elif args.inference_function == 'mean_field_variational_inference':
        assert args.amortized_logits_transform_to_confidence
        inference_function = mean_field_variational_inference
    elif args.inference_function == 'GS_inference':
        inference_function = GS_inference
    elif args.inference_function == 'SGLD_inference':
        inference_function = SGLD_inference
    elif args.inference_function == 'GNN_1st_order_inference':
        inference_function = GNN_1st_order_inference
    elif args.inference_function == 'GNN_EBM_mean_field_variational_inference':
        inference_function = GNN_EBM_mean_field_variational_inference
    elif args.inference_function == 'GNN_EBM_GS_inference':
        inference_function = GNN_EBM_GS_inference
    elif args.inference_function == 'GNN_EBM_1st_order_inference_Binary_Task':
        inference_function = GNN_EBM_1st_order_inference_Binary_Task
    else:
        raise ValueError('Inference function {} not included.'.format(args.inference_function))

    ########## Set up assay/task embedding ##########
    task_X = torch.arange(args.num_tasks).to(device)

    ########## Set up molecule model ##########
    model = GNN_graphpred(args.num_layer, args.emb_dim, args.num_tasks, JK=args.JK, drop_ratio=args.dropout_ratio,
                          graph_pooling=args.graph_pooling, gnn_type=args.gnn_type)
    if not args.input_model_file == '':
        model.from_pretrained(args.input_model_file + '.pth')
    model.to(device)

    ########## For assay/task embedding ##########
    if args.energy_function == 'energy_function_GNN_EBM_CE_2nd_order_Binary_Task':
        task_embedding_model = TaskEmbeddingModel_BinaryEmbedding(args.num_tasks, embedding_dim=args.task_emb_dim).to(device)
    else:
        task_embedding_model = TaskEmbeddingModel(args.num_tasks, embedding_dim=args.task_emb_dim).to(device)

    ########## For drug-protein/molecule-task prediction ##########
    first_order_prediction_model, second_order_prediction_model = None, None
    if args.energy_function in ['energy_function_CD_AA']:
        first_order_prediction_model = MoleculeTaskPredictionModel(
            args.emb_dim, args.task_emb_dim, output_dim=2, batch_norm=args.batch_norm).to(device)
        second_order_prediction_model = MoleculeTaskTaskPredictionModel(
            args.emb_dim, args.task_emb_dim, output_dim=4, batch_norm=args.batch_norm).to(device)

    elif args.energy_function in ['energy_function_CD_GS']:
        first_order_prediction_model = MoleculeTaskPredictionModel(
            args.emb_dim, args.task_emb_dim, output_dim=2, batch_norm=args.batch_norm).to(device)
        second_order_prediction_model = MoleculeTaskTaskPredictionModel(
            args.emb_dim, args.task_emb_dim, output_dim=4, batch_norm=args.batch_norm).to(device)

    elif args.energy_function in [
        'energy_function_GNN_CE_1st_order', 'energy_function_GNN_CE_2nd_order',
        'energy_function_GNN_EBM_NCE', 'energy_function_GNN_EBM_CD_GS',
    ]:
        first_order_prediction_model = MoleculeTaskPredictionModel(
            args.emb_dim, args.task_emb_dim, output_dim=args.ebm_GNN_dim*2, batch_norm=args.batch_norm).to(device)
        second_order_prediction_model = MoleculeTaskTaskPredictionModel(
            args.emb_dim, args.task_emb_dim, output_dim=args.ebm_GNN_dim*4, batch_norm=args.batch_norm).to(device)

    elif args.energy_function in ['energy_function_GNN_EBM_CE_2nd_order_Binary_Task']:
        first_order_prediction_model = MoleculeTaskPredictionModel(
            args.emb_dim, args.task_emb_dim, output_dim=args.ebm_GNN_dim, batch_norm=args.batch_norm).to(device)
        second_order_prediction_model = MoleculeTaskTaskPredictionModel(
            args.emb_dim, args.task_emb_dim, output_dim=args.ebm_GNN_dim, batch_norm=args.batch_norm).to(device)

    NCE_C_param = None
    if args.energy_function == 'energy_function_GNN_EBM_NCE':
        NCE_C_param = NCE_C_Parameter(len(dataset)).to(device)

    ########## For GNN-EBM Model ##########
    GNN_energy_model = None
    if args.energy_function in ['energy_function_GNN_CE_1st_order']:
        if args.gnn_energy_model == 'GNN_Energy_Model_1st_Order_01':
            GNN_energy_model = GNN_Energy_Model_1st_Order_01(
                ebm_GNN_dim=args.ebm_GNN_dim, ebm_GNN_layer_num=args.ebm_GNN_layer_num, concat=args.ebm_GNN_use_concat, output_dim=1).to(device)
        elif args.gnn_energy_model == 'GNN_Energy_Model_1st_Order_02':
            GNN_energy_model = GNN_Energy_Model_1st_Order_02(
                ebm_GNN_dim=args.ebm_GNN_dim, ebm_GNN_layer_num=args.ebm_GNN_layer_num, concat=args.ebm_GNN_use_concat, output_dim=1).to(device)
        else:
            raise ValueError('GNN Energy Model {} not included.'.format(args.gnn_energy_model))
        print('GNN_energy_model\n', GNN_energy_model)

    if args.energy_function in [
        'energy_function_GNN_CE_2nd_order', 'energy_function_GNN_EBM_NCE',
        'energy_function_GNN_EBM_CD_GS', 'energy_function_GNN_EBM_CE_2nd_order_Binary_Task'
    ]:
        if args.gnn_energy_model == 'GNN_Energy_Model_2nd_Order_01':
            GNN_energy_model = GNN_Energy_Model_2nd_Order_01(
                ebm_GNN_dim=args.ebm_GNN_dim, ebm_GNN_layer_num=args.ebm_GNN_layer_num, concat=args.ebm_GNN_use_concat).to(device)
        elif args.gnn_energy_model == 'GNN_Energy_Model_2nd_Order_02':
            GNN_energy_model = GNN_Energy_Model_2nd_Order_02(
                ebm_GNN_dim=args.ebm_GNN_dim, ebm_GNN_layer_num=args.ebm_GNN_layer_num, concat=args.ebm_GNN_use_concat).to(device)
        elif args.gnn_energy_model == 'GNN_Energy_Model_2nd_Order_03':
            GNN_energy_model = GNN_Energy_Model_2nd_Order_03(
                ebm_GNN_dim=args.ebm_GNN_dim, ebm_GNN_layer_num=args.ebm_GNN_layer_num, concat=args.ebm_GNN_use_concat).to(device)
        else:
            raise ValueError('GNN Energy Model {} not included.'.format(args.gnn_energy_model))
        print('GNN_energy_model\n', GNN_energy_model)

    ########## Set up task-task knowledge graph dataset ##########
    ppi_dataset = PPI_dataset(args, args.PPI_threshold, neg_sample_size=args.neg_sample_size,
                              neg_sample_exponent=args.neg_sample_exponent)
    ppi_dataloader = DataLoader(ppi_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    print('len of PPI dataset: {}'.format(len(ppi_dataset)))
    ########## Set up task edge list / KG ##########
    task_edge = copy.deepcopy(ppi_dataset.edge_list.transpose(0, 1)).to(device) # M * 2
    ########## Set up GNN for KG ##########
    kg_model = None
    if args.use_GCN_for_KG:
        kg_model = GCNNet(embedding_dim=args.task_emb_dim, hidden_dim=args.task_emb_dim, dropout=args.kg_dropout_ratio).to(device)
    task_relation_matching_model = PairwiseTaskPredictionModel(args.task_emb_dim).to(device)

    ########## Set up 1st and 2nd order task label weights ##########
    first_order_label_weights, second_order_label_weights = extract_amortized_task_label_weights(train_dataloader, task_edge, device, args)
    first_order_label_weights = first_order_label_weights.to(device)
    second_order_label_weights = second_order_label_weights.to(device)

    ######### Set up optimization ##########
    model_param_group = []
    model_param_group.append({'params': model.parameters()})
    model_param_group.append({'params': task_embedding_model.parameters()})
    model_param_group.append({'params': first_order_prediction_model.parameters()})
    if second_order_prediction_model is not None:
        model_param_group.append({'params': second_order_prediction_model.parameters(), 'lr': args.lr*args.lr_scale})
    if GNN_energy_model is not None:
        model_param_group.append({'params': GNN_energy_model.parameters()})
    if args.use_GCN_for_KG:
        model_param_group.append({'params': kg_model.parameters()})
    if NCE_C_param is not None:
        model_param_group.append({'params': NCE_C_param.parameters(), 'lr': args.lr*10})
    print('model_param_group\n', model_param_group)
    optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)

    ########## Loading prior data ##########
    prior_prediction, prior_prediction_logits = None, None
    if args.filling_missing_data_mode in ['mtl', 'mtl_uncertainty', 'gradnorm', 'dwa', 'lbtw', 'mtl_task', 'mtl_task_KG', 'gnn', 'ebm']:
        prior_prediction, prior_prediction_logits = extract_prior_distribution_from_pretrained_model(args)

    ########## Pre-train on link-prediction ##########
    parameters = list(task_relation_matching_model.parameters()) + list(task_embedding_model.parameters())
    if kg_model is not None:
        parameters += list(kg_model.parameters())
    task_relation_optimizer = optim.Adam(parameters, lr=args.lr, weight_decay=args.decay)
    task_relation_criterion = nn.BCEWithLogitsLoss()
    for epoch in range(1, args.PPI_pretrained_epochs + 1):
        print('====PPI pre-trained epoch ' + str(epoch))
        start_time = time.time()
        train_PPI(args, device, ppi_dataloader, task_relation_optimizer)
        print('{:.3f}s'.format(time.time() - start_time))
    print()

    ########## Pre-train on MTL ##########
    readout_model = torch.nn.Linear(args.emb_dim, args.num_tasks).to(device)
    mtl_parameters = list(model.parameters()) + list(readout_model.parameters())
    mtl_optimizer = optim.Adam(mtl_parameters, lr=args.lr, weight_decay=args.decay)
    mtl_criterion = nn.BCEWithLogitsLoss(reduction='none')
    for epoch in range(1, args.MTL_pretrained_epochs + 1):
        print('====MTL pre-trained epoch ' + str(epoch))
        start_time = time.time()
        train_MTL(args, model, readout_model, device, train_dataloader, mtl_optimizer)
        print('{:.3f}s'.format(time.time() - start_time))
    print()

    ########## Main training ##########
    eval(args, model, device, train_dataloader, 'train')
    if valid_dataloader is not None:
        eval(args, model, device, valid_dataloader, 'valid')
    if test_dataloader is not None:
        eval(args, model, device, test_dataloader, 'test')
    best_valid_roc = 0
    for epoch in range(1, args.epochs + 1):
        print('====epoch ' + str(epoch))
        start_time = time.time()
        train(args, device, train_dataloader, optimizer)

        if epoch % args.eval_every_n_epochs == 0:
            # eval(args, model, device, train_dataloader, 'train')
            valid_roc, _, _, _ = eval(args, model, device, valid_dataloader, 'valid')
            _, y_true_list, y_pred_list, id_list = eval(args, model, device, test_dataloader, 'test')
            if valid_roc > best_valid_roc:
                best_valid_roc = valid_roc
                save_model(save_best=True)
                filename = '{}_test'.format(args.output_model_file)
                print('save to \t', filename)
                np.savez(filename, y_true_list=y_true_list, y_pred_list=y_pred_list, id_list=id_list)
                if prior_prediction is not None:
                    # save prior_prediction, prior_prediction_logits
                    filename = '{}_prior'.format(args.output_model_file)
                    np.savez(filename, prior_prediction=prior_prediction.cpu().numpy(), prior_prediction_logits=prior_prediction_logits.cpu().numpy())

        if epoch >= args.filling_missing_data_fine_tuned_epoch and epoch % args.eval_every_n_epochs == 0 and \
                args.filling_missing_data_mode in ['mtl', 'mtl_uncertainty', 'gradnorm', 'dwa', 'lbtw', 'mtl_task', 'mtl_task_KG', 'gnn', 'ebm']:
            print('change missing data filling mode from {} to current best model'.format(args.filling_missing_data_mode))
            prior_prediction, prior_prediction_logits = extract_prior_distribution_from_current_model(args, [train_dataloader, valid_dataloader, test_dataloader])

        print('{:.3f}s'.format(time.time() - start_time))
        print()
    print()

    eval(args, model, device, train_dataloader, 'train')
    valid_roc, _, _, _ = eval(args, model, device, valid_dataloader, 'valid')
    _, y_true_list, y_pred_list, id_list = eval(args, model, device, test_dataloader, 'test')
    if valid_roc > best_valid_roc:
        best_valid_roc = valid_roc
        save_model(save_best=True)
        filename = '{}_test'.format(args.output_model_file)
        print('save to \t', filename)
        np.savez(filename, y_true_list=y_true_list, y_pred_list=y_pred_list, id_list=id_list)
        if prior_prediction is not None:
            # save prior_prediction, prior_prediction_logits
            filename = '{}_prior'.format(args.output_model_file)
            np.savez(filename, prior_prediction=prior_prediction.cpu().numpy(), prior_prediction_logits=prior_prediction_logits.cpu().numpy())

    save_model(save_best=False)
import copy
import time
import numpy as np
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import SubsetRandomSampler
from torch_geometric.data import DataLoader

from datasets import MoleculeDataset
from models import GNN_graphpred, SingleTaskModel
from splitters import random_split, random_filtered_split, random_scaffold_split
from configure import *


def get_loss(y_pred, y_true):
    # whether y is non-null or not.
    is_valid = y_true**2 > 0
    # loss matrix
    loss_mat = criterion(y_pred.double(), (y_true+1)/2)
    # loss matrix after removing null target
    loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))
    # loss for each task, some task may have 0 valid labels
    valid_task_count_list = is_valid.sum(0)
    loss_each_task_list = (loss_mat.sum(0)) / (valid_task_count_list + EPS)
    loss = loss_each_task_list.sum() / ((valid_task_count_list > 0).sum() + EPS)
    return loss


def train(args, model, device, loader, optimizer, task_idx):
    model.train()

    loss_acc = 0
    for step, batch in enumerate(loader):
        batch = batch.to(device)
        y_pred = model(batch).flatten(1)
        y_true = batch.y.view(y_pred.shape[0], -1)[:, task_idx].to(torch.float64)
        optimizer.zero_grad()

        loss = get_loss(y_pred, y_true)
        loss.backward()
        optimizer.step()

        loss_acc += loss.detach().item()
    print('Loss: ', loss_acc / len(loader))
    return


def eval(args, model, device, loader, evaluation_mode, task_idx):
    model.eval()
    y_true = []
    y_scores = []

    with torch.no_grad():
        for step, batch in enumerate(loader):
            batch = batch.to(device)
            y_pred = model(batch).flatten(1)

            y_true.append(batch.y.view(y_pred.shape[0], -1)[:, task_idx].cpu())
            y_scores.append(y_pred.cpu())

    y_true = torch.cat(y_true, dim=0).numpy()
    y_scores = torch.cat(y_scores, dim=0).numpy()

    roc_list = []
    invalid_count = 0
    for i in range(y_true.shape[1]):
        #AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == -1) > 0:
            is_valid = y_true[:,i]**2 > 0
            roc_list.append(roc_auc_score((y_true[is_valid,i] + 1)/2, y_scores[is_valid,i]))
        else:
            invalid_count += 1
    print('{}\tcomplete roc: {}'.format(evaluation_mode, '\t'.join(['{}'.format(x) for x in roc_list])))

    if len(roc_list) < y_true.shape[1]:
        print('Some target is missing!')
        print('Missing ratio: %f' %(1 - float(len(roc_list))/y_true.shape[1]))

    roc_list = np.array(roc_list)
    roc_value = np.mean(roc_list)
    print(f'{evaluation_mode}\tROC\t{roc_value}')
    return roc_list


if __name__ == '__main__':
    print(args.num_tasks)
    print('arguments\t', args)
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda:' + str(args.device)) if torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    root = '../datasets/' + args.dataset
    dataset = MoleculeDataset(root, dataset=args.dataset)
    print('len of ChEMBL\t', len(dataset))

    if args.split_method == 'random_split':
        train_indices, valid_indices, test_indices = random_split(
            dataset, task_idx=None, null_value=0, frac_train=0.6, frac_valid=0., frac_test=0.4, seed=args.seed)
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
    elif args.split_method == 'pre_split':
        # TODO: Load pre-defined split
        train_indices, valid_indices, test_indices = random_split(
            dataset, task_idx=None, null_value=0, frac_train=0.6, frac_valid=0., frac_test=0.4, seed=args.seed)
    else:
        raise ValueError('Split method {} not included.'.format(args.split_method))

    print(f'train: {len(train_indices)}\tvalid: {len(valid_indices)}\ttest: {len(test_indices)}')
    print('first train indices\t', train_indices[:10])
    print('first valid indices\t', valid_indices[:10])
    print('first test indices\t', test_indices[:10])

    train_sampler = SubsetRandomSampler(train_indices)
    train_dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.num_workers)
    valid_dataloader = test_dataloader = None
    if len(valid_indices) > 0:
        valid_sampler = SubsetRandomSampler(valid_indices)
        valid_dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=valid_sampler, num_workers=args.num_workers)
    if len(test_indices) > 0:
        test_sampler = SubsetRandomSampler(test_indices)
        test_dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=test_sampler, num_workers=args.num_workers)

    #set up model
    gnn_model = GNN_graphpred(args.num_layer, args.emb_dim, 1, JK=args.JK, drop_ratio=args.dropout_ratio, graph_pooling=args.graph_pooling, gnn_type=args.gnn_type)
    if not args.input_model_file == '':
        gnn_model.from_pretrained(args.input_model_file + '.pth')
    readout_model = torch.nn.Linear(args.emb_dim, 1)
    model = nn.Sequential(gnn_model, readout_model).to(device)

    print('------ Start Single Task Learning ------')
    train_roc_list, valid_roc_list, test_roc_list = [], [], []

    for task_batch_id, task in enumerate(range(0, args.num_tasks, args.task_batch_size)):
        if task_batch_id != args.task_batch_id:
            continue
        num_task = min(args.num_tasks - task, args.task_batch_size)
        task_idx = slice(task, task + num_task)
        print('Training on task %d: %d' % (task, task + num_task))
        # set up model
        test_model = SingleTaskModel(model, num_task)

        # set up optimizer
        parameters = list(test_model.parameters())
        optimizer = optim.Adam(parameters, lr=args.lr, weight_decay=args.decay)
        criterion = nn.BCEWithLogitsLoss(reduction='none')

        best_epoch, best_roc = 0, 0
        best_model = copy.deepcopy(test_model)
        for epoch in range(1, args.epochs+1):
            print('==== epoch %d ====' % epoch)

            start_time = time.time()
            train(args, test_model, device, train_dataloader, optimizer, task_idx)

            if epoch % args.eval_every_n_epochs == 0:
                if valid_dataloader is not None:
                    roc = np.mean(eval(args, test_model, device, valid_dataloader, 'valid', task_idx))
                if roc > best_roc:
                    best_epoch = epoch
                    best_roc = roc
                    best_model = copy.deepcopy(test_model)
                eval(args, test_model, device, test_dataloader, 'test', task_idx)
            print('took {:.3f}s'.format(time.time() - start_time))

        print("Loading model from epoch {:d} with {:.5f}".format(best_epoch, best_roc))
        test_model = best_model
        train_roc_list.append(eval(args, test_model, device, train_dataloader, 'train', task_idx))
        if valid_dataloader is not None:
            valid_roc_list.append(eval(args, test_model, device, valid_dataloader, 'valid', task_idx))
        if test_dataloader is not None:
            test_roc_list.append(eval(args, test_model, device, test_dataloader, 'test', task_idx))
        print('Task %d: %d finished' % (task, task + num_task))

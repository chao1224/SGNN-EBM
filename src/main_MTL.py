import os
import random
import time
import numpy as np
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import SubsetRandomSampler
from torch_geometric.data import DataLoader

from datasets import MoleculeDataset
from models import GNN_graphpred, GradNormModel

from splitters import random_split, random_filtered_split, random_scaffold_split
from configure import *


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
        loss_mat = criterion(y_pred.double(), (y_true+1)/2)
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


def train_UW(args, model, readout_model, device, loader, optimizer):
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
        loss_mat = criterion(y_pred.double(), (y_true+1)/2)
        # loss matrix after removing null target
        loss_mat = torch.where(is_valid, loss_mat, torch.zeros_like(loss_mat))
        # loss for each task, some task may have 0 valid labels
        valid_task_count_list = is_valid.sum(0)
        task_is_valid = valid_task_count_list > 0

        valid_log_uncertainty = log_uncertainty[task_is_valid]
        one_over_uncertainty_squared = (-valid_log_uncertainty * 2).exp()
        loss_each_task_list = loss_mat.sum(0)[task_is_valid] / valid_task_count_list[task_is_valid]

        loss = (one_over_uncertainty_squared @ loss_each_task_list + valid_log_uncertainty.sum()) / task_is_valid.sum()
        loss.backward()

        optimizer.step()
        #print(log_uncertainty)

        loss_acc += loss.detach().item()

        if args.lr_decay:
            args.lr = args.lr * np.exp(-args.lr_decay)
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr
    lr_scaler = (-log_uncertainty * 2).exp().detach().mean().item()    
    print(f'Loss: {loss_acc / len(loader)}; lr: {args.lr:10.3g}; lr_scaler: {lr_scaler:10.3g}; effective lr: {args.lr * lr_scaler:10.3g}')


def get_initial_task_weight(args, model, readout_model, device, loader, optimizer):
    '''
    Small updates.
    If the loss matrix is too sparse, i.e., we have too many missing labels,
    then we use this amortised-version of loss for each task:
    Go through the whole dataset with current :math:`\theta` and get a loss list.
    '''
    model.eval()
    readout_model.eval()

    global_loss_each_task_list, global_valid_task_count_list = 0, 0

    with torch.no_grad():
        for step, batch in enumerate(loader):
            batch = batch.to(device)
            graph_repr = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            y_pred = readout_model(graph_repr)
            y_true = batch.y.view(y_pred.shape).to(torch.float64)
            optimizer.zero_grad()

            is_valid = y_true**2 > 0
            loss_mat = criterion(y_pred.double(), (y_true+1)/2)
            loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))
            valid_task_count_list = is_valid.sum(0).detach()
            loss_each_task_list = loss_mat.sum(0).detach()

            global_loss_each_task_list += loss_each_task_list
            global_valid_task_count_list += valid_task_count_list

    task_loss_each_task_list = global_loss_each_task_list / (global_valid_task_count_list + EPS)
    print('valid task num\t', torch.sum(global_valid_task_count_list>0))

    return task_loss_each_task_list


def backward_grad_extraction_hook(module, grad_input, grad_output):
    '''
    grad_input[0]: T
    grad_input[1]: B * emb_dim
    grad_input[2]: emb_dim * T
    grad_output[0]: B * T
    '''

    ########## Some simple output ##########
    # print('len of grad input\t', len(grad_input), grad_input[0].size(), grad_input[1].size(), grad_input[2].size())
    # print('len of grad output\t', len(grad_output), grad_output[0].size())

    ########## For simplicity, we just use a global variable to store gradient, but this cannot fit into parallel operation ##########
    global global_readout_grad
    global_readout_grad = grad_input[2]
    return


def train_GradNorm(args, model, readout_model, device, loader, optimizer):
    alpha = args.alpha
    initial_loss_each_task_list = get_initial_task_weight(args, model, readout_model, device, loader, optimizer)
    model.train()
    readout_model.train()

    loss_acc = 0
    for step, batch in enumerate(loader):
        batch = batch.to(device)
        graph_repr = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        y_pred = readout_model(graph_repr)
        y_true = batch.y.view(y_pred.shape).to(torch.float64)
        optimizer.zero_grad()

        is_valid = y_true**2 > 0
        loss_mat = criterion(y_pred.double(), (y_true+1)/2)
        loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))
        valid_task_count_list = is_valid.sum(0)
        loss_each_task_list = (loss_mat.sum(0)) / (valid_task_count_list+EPS)
        valid_task_slice = torch.nonzero(valid_task_count_list > 0).squeeze()

        optimizer.zero_grad()
        ########## w_i L_i ##########
        weighted_loss_each_task_list = loss_each_task_list * gradnorm_model.weights
        ########## \sum_i w_i L_i ##########
        loss = weighted_loss_each_task_list.sum() / (valid_task_count_list > 0).sum()

        loss.backward(retain_graph=True)

        gradnorm_model.weights.grad.data.fill_(0)

        global global_readout_grad

        valid_gradient = global_readout_grad[:, valid_task_slice].transpose(0, 1)

        ########## G_W^i ##########
        grad_norm_weights = gradnorm_model.weights
        valid_grad_norm_weights = grad_norm_weights[valid_task_slice]
        GW_norm_list = torch.norm(valid_gradient, dim=1)
        weighted_GW_norm_list = valid_grad_norm_weights * GW_norm_list

        ########## \bar G_W ##########
        expected_weighted_GW_norm = weighted_GW_norm_list.mean().detach()

        ########## \tilde L_i ##########
        loss_ratio = loss_each_task_list.detach() / (initial_loss_each_task_list + EPS) # If all tasks have >0 labels, then we don't need to add EPS
        loss_ratio = loss_ratio[valid_task_slice]
        ########## r_i ##########
        inverse_training_rate = loss_ratio / loss_ratio.mean()

        ########## Constant ##########
        target = expected_weighted_GW_norm * torch.pow(inverse_training_rate, alpha)
        ########## L_grad ##########
        L = torch.mean(torch.abs(weighted_GW_norm_list - target))

        ########## Update gradients of GradNorm ##########
        gradnorm_model.weights.grad = torch.autograd.grad(L, gradnorm_model.weights)[0]

        optimizer.step()

        gradnorm_model.renormalize()

        loss_acc += loss.detach().item()
    print('Loss: ', loss_acc / len(loader))
    return


def train_LBTW(args, model, readout_model, device, loader, optimizer):
    alpha = args.alpha
    initial_loss_each_task_list = get_initial_task_weight(args, model, readout_model, device, loader, optimizer)
    model.train()
    readout_model.train()

    loss_acc = 0
    for step, batch in enumerate(loader):
        batch = batch.to(device)
        graph_repr = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        y_pred = readout_model(graph_repr)
        y_true = batch.y.view(y_pred.shape).to(torch.float64)
        optimizer.zero_grad()

        is_valid = y_true**2 > 0
        loss_mat = criterion(y_pred.double(), (y_true+1)/2)
        loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))
        valid_task_count_list = is_valid.sum(0)
        loss_each_task_list = (loss_mat.sum(0)) / (valid_task_count_list+EPS)

        # get loss ratio
        loss_ratio = loss_each_task_list.detach() / (initial_loss_each_task_list + EPS) # If all tasks have >0 labels, then we don't need to add EPS
        task_weights = loss_ratio.pow(alpha)

        loss_each_task_list = torch.mul(loss_each_task_list, task_weights)
        loss = loss_each_task_list.sum() / (valid_task_count_list > 0).sum()
        loss.backward()
        optimizer.step()

        loss_acc += loss.detach().item()
    print('Loss: ', loss_acc / len(loader))
    return


def train_DWA(args, model, readout_model, device, loader, optimizer):
    initial_loss_each_task_list = get_initial_task_weight(args, model, readout_model, device, loader, optimizer)
    model.train()
    readout_model.train()

    loss_acc = 0
    for step, batch in enumerate(loader):
        batch = batch.to(device)
        graph_repr = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        y_pred = readout_model(graph_repr)
        y_true = batch.y.view(y_pred.shape).to(torch.float64)
        optimizer.zero_grad()

        is_valid = y_true**2 > 0
        loss_mat = criterion(y_pred.double(), (y_true+1)/2)
        loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))
        valid_task_count_list = is_valid.sum(0)
        loss_each_task_list = (loss_mat.sum(0)) / (valid_task_count_list+EPS)

        # get loss ratio
        loss_ratio = loss_each_task_list.detach() / (initial_loss_each_task_list + EPS) # If all tasks have >0 labels, then we don't need to add EPS
        loss_ratio /= args.dwa_T
        loss_ratio = softmax_smoothing(loss_ratio) * args.num_tasks
        task_weights = loss_ratio

        loss_each_task_list = torch.mul(loss_each_task_list, task_weights)
        loss = loss_each_task_list.sum() / (valid_task_count_list > 0).sum()
        loss.backward()
        optimizer.step()

        loss_acc += loss.detach().item()
    print('Loss: ', loss_acc / len(loader))
    return


def eval(args, model, readout_model, device, loader, evaluation_mode):
    model.eval()
    readout_model.eval()
    y_true = []
    y_scores = []

    with torch.no_grad():
        for step, batch in enumerate(loader):
            batch = batch.to(device)
            graph_repr = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            y_pred = readout_model(graph_repr)

            y_true.append(batch.y.view(y_pred.shape).cpu())
            y_scores.append(y_pred.cpu())

    y_true = torch.cat(y_true, dim=0).numpy()
    y_scores = torch.cat(y_scores, dim=0).numpy()

    if args.output_y_score_file and evaluation_mode == 'all':
        import pickle
        with open(args.output_y_score_file, 'wb') as output_file:
            pickle.dump(y_scores, output_file)

    roc_list = []
    invalid_count = 0
    for i in range(y_true.shape[1]):
        #AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == -1) > 0:
            is_valid = y_true[:,i]**2 > 0
            roc_list.append(roc_auc_score((y_true[is_valid,i] + 1)/2, y_scores[is_valid,i]))
        else:
            invalid_count += 1

    print('Invalid task count:\t', invalid_count)

    if len(roc_list) < y_true.shape[1]:
        print('Some target is missing!')
        print('Missing ratio: %f' %(1 - float(len(roc_list))/y_true.shape[1]))

    roc_list = np.array(roc_list)
    roc_value = np.mean(roc_list)
    print(f'{evaluation_mode}\tROC\t{roc_value}')
    return roc_value


def save_model(save_best=False):
    if not args.output_model_file == '':
        saved_model_dict = {
            'model': model.state_dict(),
            'readout_model': readout_model.state_dict()
        }

        ########## For GradNorm ##########
        if gradnorm_model is not None:
            saved_model_dict.update({'gradnorm_model': gradnorm_model.state_dict()})

        ########## For GradKG ##########
        if gradKG_model is not None:
            saved_model_dict.update({'gradKG_model': gradKG_model.state_dict()})
        if task_relation_matching_model is not None:
            saved_model_dict.update({'task_relation_matching_model': task_relation_matching_model.state_dict()})

        if save_best:
            print('saving best model...')
            torch.save(saved_model_dict, args.output_model_file + '_best.pth')
        else:
            torch.save(saved_model_dict, args.output_model_file + '.pth')
    return


if __name__ == '__main__':
    print(args.num_tasks)
    print('arguments\t', args)
    
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device('cuda:' + str(args.device)) if torch.cuda.is_available() else torch.device('cpu')

    root = '../datasets/' + args.dataset
    dataset = MoleculeDataset(root=root, dataset=args.dataset)
    print('len of ChEMBL\t', len(dataset))

    start_time = time.time()
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
    print('splitting took\t', time.time() - start_time)

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

    #set up model
    model = GNN_graphpred(args.num_layer, args.emb_dim, args.num_tasks, JK=args.JK, drop_ratio=args.dropout_ratio, graph_pooling=args.graph_pooling, gnn_type=args.gnn_type)
    if not args.input_model_file == '':
        model.from_pretrained(args.input_model_file + '.pth')
    model.to(device)
    readout_model = torch.nn.Linear(args.emb_dim, args.num_tasks).to(device)

    gradnorm_model = None
    gradKG_model, gradKG_optimizer, task_relation_matching_model, gnn_model = None, None, None, None
    global_readout_grad = None
    softmax_smoothing = nn.Softmax(dim=0)
    log_uncertainty = None

    if args.mtl_method == 'mtl':
        train_function = train_MTL

    elif args.mtl_method == 'uw':
        train_function = train_UW
        log_uncertainty = nn.Parameter(torch.randn(args.num_tasks, dtype=torch.float64, device=device) * 0.05)

    elif args.mtl_method == 'gradnorm':
        train_function = train_GradNorm
        readout_model.register_backward_hook(backward_grad_extraction_hook)
        gradnorm_model = GradNormModel(args.num_tasks, args).to(device)

    elif args.mtl_method == 'lbtw':
        train_function = train_LBTW

    elif args.mtl_method == 'dwa':
        train_function = train_DWA

    else:
        raise ValueError('Training Function {} not included.'.format(args.mtl_method))

    #set up optimizer
    parameters = list(model.parameters()) + list(readout_model.parameters())
    if gradnorm_model is not None:
        parameters += list(gradnorm_model.parameters())
    if log_uncertainty is not None:
        parameters += [log_uncertainty]
    optimizer = optim.Adam(parameters, lr=args.lr, weight_decay=args.decay)
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    task_relation_criterion = nn.BCEWithLogitsLoss()

    best_valid_roc, best_test_roc = 0, 0
    for epoch in range(1, args.epochs+1):
        print('====epoch ' + str(epoch))

        start_time = time.time()
        train_function(args, model, readout_model, device, train_dataloader, optimizer)

        if (epoch % args.eval_every_n_epochs == 0 or epoch == args.epochs) and valid_dataloader is not None and test_dataloader is not None:
            valid_roc = eval(args, model, readout_model, device, valid_dataloader, 'valid')
            test_roc = eval(args, model, readout_model, device, test_dataloader, 'test')
            if valid_roc > best_valid_roc:
                best_valid_roc, best_test_roc = valid_roc, test_roc
                save_model(save_best=True)
                
        print('took {:.3f}s'.format(time.time() - start_time))
        print()

    save_model(save_best=False)
    print(f'Best valid_roc: {best_valid_roc:10.6f}; Best test_roc: {best_test_roc:10.6f}')

    if args.output_y_score_file:
        eval(args, model, readout_model, device, DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers), 'all')
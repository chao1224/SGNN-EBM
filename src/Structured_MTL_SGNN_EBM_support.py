import torch
import torch.nn as nn
from torch_scatter import scatter_add, scatter_mean

from configure import *
from Structured_MTL_EBM_support import *

bce_criterion = nn.BCEWithLogitsLoss()
ce_criterion = nn.CrossEntropyLoss()
softmax_opt = nn.Softmax(-1)
EPS = 1e-8


# ----------------------------------------------------------------------------------------------------------------------
#
# Prediction Function
#
# ----------------------------------------------------------------------------------------------------------------------

# get_GNN_prediction_second_order
def get_GNN_prediction_1st_order_prediction(
        first_order_prediction_model, second_order_prediction_model, GNN_energy_model,
        graph_repr, task_repr, task_edge, args):
    B = len(graph_repr)
    T = task_repr.size()[0]
    M = task_edge.size()[1]

    ########## Get 1st-order prediction ##########
    graph_repr_1st_order = graph_repr.unsqueeze(1).expand(-1, args.num_tasks, -1)  # B, T, d_mol
    task_repr_1st_order = task_repr.unsqueeze(0).expand(B, -1, -1)  # B, T, d_task
    y_pred_1st_order = first_order_prediction_model(graph_repr_1st_order, task_repr_1st_order)  # B, T, 2*d_ebm
    y_pred_1st_order = y_pred_1st_order.view(B, T, 2, -1)  # B, T, 2, d_ebm

    ########## Get 2nd-order prediction ##########
    graph_repr_2nd_order = graph_repr.unsqueeze(1).expand(-1, M, -1)  # B, M, d_mol
    task_repr_2nd_order_node1, task_repr_2nd_order_node2 = mapping_task_repr(task_repr, task_edge, B)  # (B, M, d_task), (B, M, d_task)
    y_pred_2nd_order = second_order_prediction_model(graph_repr_2nd_order, task_repr_2nd_order_node1, task_repr_2nd_order_node2)  # B, M, 4*d_ebm
    y_pred_2nd_order = y_pred_2nd_order.view(B, M, 4, -1)  # B, M, 4, d_ebm

    y_pred = GNN_energy_model(y_pred_1st_order, y_pred_2nd_order, task_edge)  # B, T, 2

    return y_pred


# get_GNN_prediction_second_order_2nd_Order_Prediction
def get_GNN_prediction_2nd_order_prediction(
        first_order_prediction_model, second_order_prediction_model, GNN_energy_model,
        graph_repr, task_repr, task_edge, args):
    B = len(graph_repr)
    T = task_repr.size()[0]
    M = task_edge.size()[1]

    ########## Get 1st-order prediction ##########
    graph_repr_1st_order = graph_repr.unsqueeze(1).expand(-1, args.num_tasks, -1)  # B, T, d_mol
    task_repr_1st_order = task_repr.unsqueeze(0).expand(B, -1, -1)  # B, T, d_task
    y_pred_1st_order = first_order_prediction_model(graph_repr_1st_order, task_repr_1st_order)  # B, T, 2*d_ebm
    y_pred_1st_order = y_pred_1st_order.view(B, T, 2, -1)  # B, T, 2, d_ebm

    ########## Get 2nd-order prediction ##########
    graph_repr_2nd_order = graph_repr.unsqueeze(1).expand(-1, M, -1)  # B, M, d_mol
    task_repr_2nd_order_node1, task_repr_2nd_order_node2 = mapping_task_repr(task_repr, task_edge, B)  # (B, M, d_task), (B, M, d_task)
    y_pred_2nd_order = second_order_prediction_model(graph_repr_2nd_order, task_repr_2nd_order_node1, task_repr_2nd_order_node2)  # B, M, 4*d_ebm
    y_pred_2nd_order = y_pred_2nd_order.view(B, M, 4, -1)  # B, M, 4, d_ebm

    y_pred_1st_order, y_pred_2nd_order = GNN_energy_model(y_pred_1st_order, y_pred_2nd_order, task_edge)  # (B, T, 2), (B, M, 4)

    # print(y_pred_2nd_order[0, :10, 1])
    if torch.isnan(y_pred_2nd_order).any():
        import pdb; pdb.set_trace()

    if args.softmax_energy:
        y_pred_1st_order = softmax_opt(y_pred_1st_order)
        if torch.isnan(y_pred_1st_order).any():
            import pdb; pdb.set_trace()
        y_pred_2nd_order = softmax_opt(y_pred_2nd_order)

    return y_pred_1st_order, y_pred_2nd_order


def get_GNN_prediction_Binary_Task_Embedding_CE(
        first_order_prediction_model, second_order_prediction_model,
        GNN_energy_model,
        graph_repr, task_repr, task_edge, args):
    B = len(graph_repr)
    T = task_repr.size()[0]
    M = task_edge.size()[1]

    # graph_repr: B, T, d_mol
    # task_repr: T, 2, d_task

    ########## Get 1st-order prediction ##########
    graph_repr_1st_order = graph_repr.unsqueeze(1).expand(-1, args.num_tasks, -1)  # B, T, d_mol
    task_repr_1st_order_neg = task_repr[:, 0].unsqueeze(0).expand(B, -1, -1)  # B, T, d_task
    task_repr_1st_order_pos = task_repr[:, 1].unsqueeze(0).expand(B, -1, -1)  # B, T, d_task
    y_pred_1st_order_neg = first_order_prediction_model(graph_repr_1st_order, task_repr_1st_order_neg)  # B, T, d_ebm
    y_pred_1st_order_pos = first_order_prediction_model(graph_repr_1st_order, task_repr_1st_order_pos)  # B, T, d_ebm
    y_pred_1st_order = torch.stack([y_pred_1st_order_neg, y_pred_1st_order_pos], dim=2)  # B, T, 2, d_ebm

    ########## Get 2nd-order prediction ##########
    graph_repr_2nd_order = graph_repr.unsqueeze(1).expand(-1, M, -1)  # B, M, d_mol
    task_repr_2nd_order_node1_neg, task_repr_2nd_order_node2_neg = mapping_task_repr(task_repr[:, 0], task_edge, B)  # (B, M, d_task), (B, M, d_task)
    task_repr_2nd_order_node1_pos, task_repr_2nd_order_node2_pos = mapping_task_repr(task_repr[:, 1], task_edge, B)  # (B, M, d_task), (B, M, d_task)
    y_pred_2nd_order_neg_neg = second_order_prediction_model(graph_repr_2nd_order, task_repr_2nd_order_node1_neg, task_repr_2nd_order_node2_neg)  # B, M, d_ebm
    y_pred_2nd_order_neg_pos = second_order_prediction_model(graph_repr_2nd_order, task_repr_2nd_order_node1_neg, task_repr_2nd_order_node2_pos)  # B, M, d_ebm
    y_pred_2nd_order_pos_neg = second_order_prediction_model(graph_repr_2nd_order, task_repr_2nd_order_node1_pos, task_repr_2nd_order_node2_neg)  # B, M, d_ebm
    y_pred_2nd_order_pos_pos = second_order_prediction_model(graph_repr_2nd_order, task_repr_2nd_order_node1_pos, task_repr_2nd_order_node2_pos)  # B, M, d_ebm
    y_pred_2nd_order = torch.stack(
        [y_pred_2nd_order_neg_neg, y_pred_2nd_order_neg_pos, y_pred_2nd_order_pos_neg, y_pred_2nd_order_pos_pos], dim=2
    ) # B, M, 4, d_ebm

    y_pred_1st_order, y_pred_2nd_order = GNN_energy_model(y_pred_1st_order, y_pred_2nd_order, task_edge)  # (B, T, 2), (B, M, 4)

    if args.softmax_energy:
        y_pred_1st_order = softmax_opt(y_pred_1st_order)
        y_pred_2nd_order = softmax_opt(y_pred_2nd_order)

    return y_pred_1st_order, y_pred_2nd_order


# ----------------------------------------------------------------------------------------------------------------------
#
# Energy Function
#
# ----------------------------------------------------------------------------------------------------------------------

def energy_function_GNN_CE_1st_order(
        first_order_prediction_model, second_order_prediction_model, GNN_energy_model, prediction_function,
        graph_repr, task_repr, y_true, task_edge, args, **kwargs):
    y_pred = prediction_function(
        first_order_prediction_model=first_order_prediction_model,
        second_order_prediction_model=second_order_prediction_model,
        GNN_energy_model=GNN_energy_model,
        graph_repr=graph_repr, task_repr=task_repr, task_edge=task_edge,
        args=args
    )
    y_true = y_true.unsqueeze(2)  # B, T, 1
    y_valid = y_true ** 2 > 0
    y_true = ((1 + y_true) / 2)  # B, T, 1

    masked_y_true = torch.masked_select(y_true, y_valid)
    masked_y_pred = torch.masked_select(y_pred, y_valid)

    energy_loss = bce_criterion(masked_y_pred, masked_y_true)

    return energy_loss


def energy_function_GNN_CE_2nd_order(
        first_order_prediction_model, second_order_prediction_model, GNN_energy_model, prediction_function,
        graph_repr, task_repr, y_true, task_edge,
        first_order_label_weights, second_order_label_weights, args, **kwargs):
    y_pred_1st_order, y_pred_2nd_order = prediction_function(
        first_order_prediction_model=first_order_prediction_model,
        second_order_prediction_model=second_order_prediction_model,
        GNN_energy_model=GNN_energy_model,
        graph_repr=graph_repr, task_repr=task_repr, task_edge=task_edge,
        args=args
    )
    y_valid = y_true ** 2 > 0

    ########## Prepare for 1st-order energy ##########
    y_true_1st_order = ((1+y_true) / 2).long().unsqueeze(2)  # B, T, 1
    y_valid_1st_order = y_valid.unsqueeze(2)  # B, T, 1
    masked_y_true_1st_order = torch.masked_select(y_true_1st_order, y_valid_1st_order)
    masked_y_pred_1st_order = torch.masked_select(y_pred_1st_order, y_valid_1st_order).view(-1, 2)

    ########## Prepare for 2nd-order energy ##########
    y_true_2nd_order = mapping_label(y_true, task_edge)  # B, M, 1
    y_valid_2nd_order = mapping_valid_label(y_valid, task_edge)  # B, M, 1
    masked_y_true_2nd_order = torch.masked_select(y_true_2nd_order, y_valid_2nd_order)
    masked_y_pred_2nd_order = torch.masked_select(y_pred_2nd_order, y_valid_2nd_order).view(-1, 4)

    first_order_energy_loss = ce_criterion(masked_y_pred_1st_order, masked_y_true_1st_order)
    second_order_energy_loss = ce_criterion(masked_y_pred_2nd_order, masked_y_true_2nd_order)

    return y_pred_1st_order, y_pred_2nd_order, first_order_energy_loss, second_order_energy_loss


def extract_log_prob_for_1st_order(y_pred_1st_order, y_pred_2nd_order, y_true, y_valid, task_edge, args):
    '''
    :param y_pred_1st_order: (B, T, 2)
    :param y_pred_2nd_order: (B, M, 4)
    :param y_true: B, T
    :param y_valid: B, T
    :return:
    '''
    y_true_1st_order = y_true.unsqueeze(2)  # B, T, 1
    y_valid_1st_order = y_valid.unsqueeze(2)  # B, T, 1
    y_pred_1st_order = torch.gather(y_pred_1st_order, 2, y_true_1st_order)  # (B, T, 2) => (B, T, 1)
    y_pred_1st_order = torch.log(y_pred_1st_order + EPS)
    energy_1st_order = torch.sum(y_pred_1st_order*y_valid_1st_order, dim=1)  # B, 1

    energy = energy_1st_order.squeeze(1)

    return energy


def extract_log_prob_for_energy(y_pred_1st_order, y_pred_2nd_order, y_true, y_valid, task_edge, args):
    '''
    :param y_pred_1st_order: (B, T, 2)
    :param y_pred_2nd_order: (B, M, 4)
    :param y_true: B, T
    :param y_valid: B, T
    :return:
    '''
    y_true_1st_order = y_true.unsqueeze(2)  # B, T, 1
    y_valid_1st_order = y_valid.unsqueeze(2)  # B, T, 1
    y_pred_1st_order = torch.gather(y_pred_1st_order, 2, y_true_1st_order)  # (B, T, 2) => (B, T, 1)
    y_pred_1st_order = torch.log(y_pred_1st_order + EPS)
    energy_1st_order = torch.sum(y_pred_1st_order * y_valid_1st_order, dim=1)  # B, 1

    y_true_2nd_order = mapping_label_02(y_true, task_edge)  # B, M, 1
    y_valid_2nd_order = mapping_valid_label(y_valid, task_edge)  # B, M, 1
    y_pred_2nd_order = torch.gather(y_pred_2nd_order, 2, y_true_2nd_order)  # (B, M, 4) => (B, M, 1)
    y_pred_2nd_order = torch.log(y_pred_2nd_order + EPS)
    energy_2nd_order = torch.sum(y_pred_2nd_order * y_valid_2nd_order, dim=1)  # B, 1

    energy = energy_1st_order + args.structured_lambda * energy_2nd_order
    energy = energy.squeeze(1)

    return energy


def energy_function_GNN_EBM_NCE(
        first_order_prediction_model, second_order_prediction_model, GNN_energy_model, prediction_function,
        graph_repr, task_repr, y_true, task_edge, NCE_C_param, args,
        prior_prediction=None, prior_prediction_logits=None, id=None, **kwargs):
    y_pred_1st_order, y_pred_2nd_order = prediction_function(
        first_order_prediction_model=first_order_prediction_model,
        second_order_prediction_model=second_order_prediction_model,
        GNN_energy_model=GNN_energy_model,
        graph_repr=graph_repr, task_repr=task_repr, task_edge=task_edge,
        args=args
    )
    B = y_pred_1st_order.size()[0]
    M = y_pred_2nd_order.size()[1]
    y_valid = y_true ** 2 > 0  # B, T
    y_true = ((1+y_true) / 2).long()  # B, T

    if args.NCE_mode == 'uniform':
        # TODO: filling 0 for missing labels
        y_valid = (y_valid >= -1).long()

        x_noise_1st_order = torch.ones_like(y_pred_1st_order) * 0.5  # B, M, 2
        x_noise_2nd_order = torch.ones_like(y_pred_2nd_order) * 0.25  # B, M, 4

        log_p_x = extract_log_prob_for_energy(y_pred_1st_order, y_pred_2nd_order, y_true, y_valid, task_edge, args)  # B
        log_q_x = extract_log_prob_for_1st_order(x_noise_1st_order, x_noise_2nd_order, y_true, y_valid, task_edge, args)  # B, -T log(2)
        # # TODO:
        # import math
        # T = y_pred_1st_order.size()[1]
        # print('log_q_x\t', log_q_x, '\t', T * math.log(2))
        loss_data = log_p_x - torch.logsumexp(torch.stack([log_p_x, log_q_x], dim=1), dim=1, keepdim=True)

        y_noise = softmax_opt(torch.rand_like(y_pred_1st_order))  # B, T, 2
        y_noise = (y_noise[..., 1] >= 0.5).long()  # B, T
        log_p_noise = extract_log_prob_for_energy(y_pred_1st_order, y_pred_2nd_order, y_noise, y_valid, task_edge, args)  # B
        log_q_noise = extract_log_prob_for_1st_order(x_noise_1st_order, x_noise_2nd_order, y_noise, y_valid, task_edge, args)  # B, -T log(2)
        loss_noise = log_q_noise - torch.logsumexp(torch.stack([log_p_noise, log_q_noise], dim=1), dim=1, keepdim=True)

        loss = - (loss_data.mean() + loss_noise.mean())

    elif args.NCE_mode == 'gs':
        x_noise_1st_order = GNN_EBM_GS_inference(
            y_pred_1st_order=y_pred_1st_order, y_pred_2nd_order=y_pred_2nd_order, return_full_prob=True,
            first_order_prediction_model=first_order_prediction_model,
            second_order_prediction_model=second_order_prediction_model,
            GNN_energy_model=GNN_energy_model, prediction_function=prediction_function,
            prior_prediction=prior_prediction, prior_prediction_logits=prior_prediction_logits, id=id,
            graph_repr=graph_repr, task_repr=task_repr, task_edge=task_edge, args=args,
        )  # B, T, 2
        x_noise_1st_order_u = x_noise_1st_order[:, task_edge[0], :]  # B, M, 2
        x_noise_1st_order_v = x_noise_1st_order[:, task_edge[1], :]  # B, M, 2
        x_noise_2nd_order = torch.einsum('bmxy,bmyz->bmxz',
            x_noise_1st_order_u.unsqueeze(3), x_noise_1st_order_v.unsqueeze(2)).view(B, M, 4)  # B, M, 4

        y_prob = torch.rand_like(x_noise_1st_order)  # B, T, 2
        y_noise = (x_noise_1st_order >= y_prob).long()[..., 1]  # B, T
        if args.filling_missing_data_mode is not 'no_filling':
            y_true = torch.where(y_valid, y_true, y_noise)
            y_valid.fill_(1)

        log_p_x = extract_log_prob_for_energy(y_pred_1st_order, y_pred_2nd_order, y_true, y_valid, task_edge, args)  # B
        log_q_x = extract_log_prob_for_1st_order(x_noise_1st_order, x_noise_2nd_order, y_true, y_valid, task_edge, args)  # B
        loss_data = log_p_x - torch.logsumexp(torch.stack([log_p_x, log_q_x], dim=1), dim=1, keepdim=True)

        log_p_noise = extract_log_prob_for_energy(y_pred_1st_order, y_pred_2nd_order, y_noise, y_valid, task_edge, args)  # B
        log_q_noise = extract_log_prob_for_1st_order(x_noise_1st_order, x_noise_2nd_order, y_noise, y_valid, task_edge, args)  # B
        loss_noise = log_q_noise - torch.logsumexp(torch.stack([log_p_noise, log_q_noise], dim=1), dim=1, keepdim=True)

        loss = - (loss_data.mean() + loss_noise.mean())

    elif args.NCE_mode == 'mean_field':
        x_noise_1st_order = GNN_EBM_mean_field_variational_inference(
            y_pred_1st_order=y_pred_1st_order, y_pred_2nd_order=y_pred_2nd_order, return_full_prob=True,
            first_order_prediction_model=first_order_prediction_model,
            second_order_prediction_model=second_order_prediction_model,
            GNN_energy_model=GNN_energy_model, prediction_function=prediction_function,
            prior_prediction=prior_prediction, prior_prediction_logits=prior_prediction_logits, id=id,
            graph_repr=graph_repr, task_repr=task_repr, task_edge=task_edge, args=args,
        )  # B, T, 2
        x_noise_1st_order_u = x_noise_1st_order[:, task_edge[0], :]  # B, M, 2
        x_noise_1st_order_v = x_noise_1st_order[:, task_edge[1], :]  # B, M, 2
        x_noise_2nd_order = torch.einsum('bmxy,bmyz->bmxz',
            x_noise_1st_order_u.unsqueeze(3), x_noise_1st_order_v.unsqueeze(2)).view(B, M, 4)  # B, M, 4

        y_prob = torch.rand_like(x_noise_1st_order)  # B, T, 2
        y_noise = (x_noise_1st_order >= y_prob).long()[..., 1]  # B, T
        if args.filling_missing_data_mode is not 'no_filling':
            y_true = torch.where(y_valid, y_true, y_noise)
            y_valid.fill_(1)

        log_p_x = extract_log_prob_for_energy(y_pred_1st_order, y_pred_2nd_order, y_true, y_valid, task_edge, args)  # B
        log_q_x = extract_log_prob_for_1st_order(x_noise_1st_order, x_noise_2nd_order, y_true, y_valid, task_edge, args)  # B
        loss_data = log_p_x - torch.logsumexp(torch.stack([log_p_x, log_q_x], dim=1), dim=1, keepdim=True)

        log_p_noise = extract_log_prob_for_energy(y_pred_1st_order, y_pred_2nd_order, y_noise, y_valid, task_edge, args)  # B
        log_q_noise = extract_log_prob_for_1st_order(x_noise_1st_order, x_noise_2nd_order, y_noise, y_valid, task_edge, args)  # B
        loss_noise = log_q_noise - torch.logsumexp(torch.stack([log_p_noise, log_q_noise], dim=1), dim=1, keepdim=True)

        loss = - (loss_data.mean() + loss_noise.mean())

    elif args.NCE_mode == 'ce':
        ########## Prepare for 1st-order energy ##########
        y_true_1st_order = y_true.unsqueeze(2)  # B, T, 1
        y_valid_1st_order = y_valid.unsqueeze(2)  # B, T, 1
        masked_y_true_1st_order = torch.masked_select(y_true_1st_order, y_valid_1st_order)
        masked_y_pred_1st_order = torch.masked_select(y_pred_1st_order, y_valid_1st_order).view(-1, 2)

        first_order_energy_loss = ce_criterion(masked_y_pred_1st_order, masked_y_true_1st_order)

        loss = first_order_energy_loss
        print('1st loss: {}\t\t2nd loss: {}'.format(first_order_energy_loss.item(), second_order_energy_loss.item()))

    elif args.NCE_mode == 'statistics':
        T = y_pred_1st_order.size()[1]
        total_N = B * T
        valid_N = 1. * torch.sum(y_valid.detach())
        pos_N = 1. * torch.sum(y_true.detach())
        neg_N = valid_N - pos_N
        print('B: {}\tT: {}\tM: {}'.format(B, T, M))
        print('total: {}\nvalid: {} ({})'.format(total_N, valid_N, valid_N/total_N))
        print('pos: {} ({})\tneg: {} ({})'.format(pos_N, pos_N/total_N, neg_N, neg_N/total_N))

        y_true_1st_order = y_true.unsqueeze(2)  # B, T, 1
        y_valid_1st_order = y_valid.unsqueeze(2)  # B, T, 1
        masked_y_true_1st_order = torch.masked_select(y_true_1st_order, y_valid_1st_order)
        masked_y_pred_1st_order = torch.masked_select(y_pred_1st_order, y_valid_1st_order).view(-1, 2)
        loss = ce_criterion(masked_y_pred_1st_order, masked_y_true_1st_order)

    else:
        raise ValueError('NCE mode {} not included.'.format(args.NCE_mode))

    return loss


def energy_function_GNN_EBM_CD_GS(
        first_order_prediction_model, second_order_prediction_model, GNN_energy_model, prediction_function,
        graph_repr, task_repr, y_true, task_edge,
        first_order_label_weights, second_order_label_weights, args, **kwargs):
    y_pred_1st_order, y_pred_2nd_order = prediction_function(
        first_order_prediction_model=first_order_prediction_model,
        second_order_prediction_model=second_order_prediction_model,
        GNN_energy_model=GNN_energy_model,
        graph_repr=graph_repr, task_repr=task_repr, task_edge=task_edge,
        args=args
    )
    y_valid = y_true ** 2 > 0
    y_true_1st_order = ((1 + y_true) / 2).long()  # B, T

    ########## Get negative phase from Gibbs Sampling ##########
    T = y_pred_1st_order.size()[1]
    if args.filling_missing_data_mode == 'no_filling':
        ########## No filling missing value, 0 for missing by default ##########
        y_sample = y_true_1st_order.clone()  # B, T
    elif args.filling_missing_data_mode in ['mtl', 'gradnorm', 'dwa', 'lbtw', 'mtl_task', 'mtl_task_KG', 'gnn']:
        prior_prediction = kwargs['prior_prediction']
        y_sample = prior_prediction.detach().clone()
        y_sample = y_sample[kwargs['id']]
    else:
        raise ValueError('Filling missing data {} not included.'.format(args.filling_missing_data_mode))
    p = torch.rand_like(y_pred_1st_order)  # B, T, 2
    y_accum = []

    pos_first_order_energy_loss, pos_second_order_energy_loss = phase_energy_calculation(y_true_1st_order, y_pred_1st_order, y_pred_2nd_order, y_valid, task_edge)
    neg_first_energy, neg_second_energy = 0, 0
    neg_first_order_energy_loss, neg_second_order_energy_loss = 0, 0

    for layer_idx in range(args.GS_iteration):
        y_sample, p = Gibbs_sampling(
            T=T, y_sample=y_sample, p=p, y_pred_1st_order=y_pred_1st_order, y_pred_2nd_order=y_pred_2nd_order,
            task_edge=task_edge, args=args)
        y_accum.append(y_sample.clone())

        neg_first_energy, neg_second_energy = phase_energy_calculation(y_sample.detach(), y_pred_1st_order, y_pred_2nd_order, y_valid, task_edge)

        neg_first_order_energy_loss += neg_first_energy
        neg_second_order_energy_loss += neg_second_energy

    if args.GS_learning == 'last':
        neg_first_order_energy_loss = neg_first_energy
        neg_second_order_energy_loss = neg_second_energy
    elif args.GS_learning == 'average':
        neg_first_order_energy_loss /= args.GS_iteration
        neg_second_order_energy_loss /= args.GS_iteration
    else:
        raise ValueError('GS learning {} not included.'.format(args.GS_learning))

    first_order_energy_loss = pos_first_order_energy_loss - neg_first_order_energy_loss
    second_order_energy_loss = pos_second_order_energy_loss - neg_second_order_energy_loss
    return y_pred_1st_order, y_pred_2nd_order, first_order_energy_loss, second_order_energy_loss


def energy_function_GNN_EBM_CE_2nd_order_Binary_Task(
        first_order_prediction_model, second_order_prediction_model, GNN_energy_model,
        graph_repr, task_repr, y_true, task_edge,
        first_order_label_weights, second_order_label_weights, args, **kwargs):
    y_pred_1st_order, y_pred_2nd_order = get_GNN_prediction_Binary_Task_Embedding_CE(
        first_order_prediction_model=first_order_prediction_model,
        second_order_prediction_model=second_order_prediction_model,
        GNN_energy_model=GNN_energy_model,
        graph_repr=graph_repr, task_repr=task_repr, task_edge=task_edge,
        args=args
    )
    y_valid = y_true ** 2 > 0

    ########## Prepare for 1st-order energy ##########
    y_true_1st_order = ((1+y_true) / 2).long().unsqueeze(2)  # B, T, 1
    y_valid_1st_order = y_valid.unsqueeze(2)  # B, T, 1
    masked_y_true_1st_order = torch.masked_select(y_true_1st_order, y_valid_1st_order)
    masked_y_pred_1st_order = torch.masked_select(y_pred_1st_order, y_valid_1st_order).view(-1, 2)

    ########## Prepare for 2nd-order energy ##########
    y_true_2nd_order = mapping_label(y_true, task_edge)  # B, M, 1
    y_valid_2nd_order = mapping_valid_label(y_valid, task_edge)  # B, M, 1
    masked_y_true_2nd_order = torch.masked_select(y_true_2nd_order, y_valid_2nd_order)
    masked_y_pred_2nd_order = torch.masked_select(y_pred_2nd_order, y_valid_2nd_order).view(-1, 4)

    first_order_energy_loss = ce_criterion(masked_y_pred_1st_order, masked_y_true_1st_order)
    second_order_energy_loss = ce_criterion(masked_y_pred_2nd_order, masked_y_true_2nd_order)

    return y_pred_1st_order, y_pred_2nd_order, first_order_energy_loss, second_order_energy_loss

# ----------------------------------------------------------------------------------------------------------------------
#
# Inference Function
#
# ----------------------------------------------------------------------------------------------------------------------

def GNN_1st_order_inference(
        first_order_prediction_model, second_order_prediction_model, GNN_energy_model, prediction_function,
        graph_repr, task_repr, task_edge, args, **kwargs):
    if args.energy_function == 'energy_function_GNN_CE_1st_order':
        y_pred_1st = prediction_function(
            first_order_prediction_model=first_order_prediction_model,
            second_order_prediction_model=second_order_prediction_model,
            GNN_energy_model=GNN_energy_model,
            graph_repr=graph_repr, task_repr=task_repr, task_edge=task_edge,
            args=args
        )
        y_pred = y_pred_1st.squeeze(2)  # B, T
    else:
        y_pred_1st, _ = prediction_function(
            first_order_prediction_model=first_order_prediction_model,
            second_order_prediction_model=second_order_prediction_model,
            GNN_energy_model=GNN_energy_model,
            graph_repr=graph_repr, task_repr=task_repr, task_edge=task_edge,
            args=args
        )
        y_pred = y_pred_1st[..., 1]  # (B, T, 2) ===> (B, T)

    return y_pred


def GNN_EBM_mean_field_variational_inference(
        first_order_prediction_model, second_order_prediction_model, GNN_energy_model, prediction_function,
        graph_repr, task_repr, task_edge, args,
        y_pred_1st_order=None, y_pred_2nd_order=None, return_full_prob=False, **kwargs):
    '''
    :return: logits or confidence: B, T
    '''
    if y_pred_1st_order is None:
        y_pred_1st_order, y_pred_2nd_order = prediction_function(
            first_order_prediction_model=first_order_prediction_model,
            second_order_prediction_model=second_order_prediction_model,
            GNN_energy_model=GNN_energy_model,
            graph_repr=graph_repr, task_repr=task_repr, task_edge=task_edge,
            args=args)
    B = y_pred_1st_order.size()[0]
    T = y_pred_1st_order.size()[1]
    M = y_pred_2nd_order.size()[1]

    T = y_pred_1st_order.size()[1]
    # if args.filling_missing_data_mode == 'no_filling':
    #     y_sample = ((softmax_opt(y_pred_1st_order)[..., 1]) > 0.5).long()  # B, T
    # elif args.filling_missing_data_mode in ['mtl', 'mtl_uncertainty', 'gradnorm', 'dwa', 'lbtw', 'mtl_task', 'mtl_task_KG', 'gnn', 'ebm']:
    #     prior_prediction = kwargs['prior_prediction']
    #     y_sample = prior_prediction.detach().clone()
    #     y_sample = y_sample[kwargs['id']]  # B, T
    # else:
    #     raise ValueError('{} not implemented.'.format(args.filling_missing_data_mode))

    y_prior = None
    if args.filling_missing_data_mode in ['mtl', 'mtl_uncertainty', 'gradnorm', 'dwa', 'lbtw', 'mtl_task', 'mtl_task_KG', 'gnn', 'ebm'] and args.ebm_as_tilting:
        prior_prediction_logits = kwargs['prior_prediction_logits']
        y_prior = prior_prediction_logits.detach().clone()
        y_prior = y_prior[kwargs['id']]
    # p = torch.rand_like(y_pred_1st_order)  # B, T, 2
    q = torch.abs(torch.randn_like(y_pred_1st_order))  # B, T, 2
    q = softmax_opt(q)  # B, T, 2

    aggregate_indice = torch.LongTensor([0, 0, 1, 1]).to(args.device)

    for _ in range(args.MFVI_iteration):
        q_i, q_j = q[:, task_edge[0]], q[:, task_edge[1]]  # (B,T,2) => (B,M,2)
        q_second_order = torch.einsum('bmxy,bmyz->bmxz', q_i.unsqueeze(3), q_j.unsqueeze(2))  # B, M, 2, 2
        q_second_order = q_second_order.view(B, M, 4)  # B, M, 4

        aggregated_y_pred_2nd_order = scatter_add(q_second_order*y_pred_2nd_order, task_edge[0], dim=1, dim_size=T)  # B, T, 4
        y_pred_2nd_order_ = scatter_add(aggregated_y_pred_2nd_order, aggregate_indice, dim=2)  # B, T, 2

        if y_prior is not None and args.ebm_as_tilting:
            prior = F.sigmoid(torch.stack([-y_prior, y_prior], dim=2))
        else:
            prior = 1
        q = y_pred_1st_order + args.structured_lambda * y_pred_2nd_order_
        q = softmax_opt(q) * prior

    if return_full_prob:
        return q
    return q[..., 1]


def GNN_EBM_GS_inference(
        first_order_prediction_model, second_order_prediction_model, GNN_energy_model, prediction_function,
        graph_repr, task_repr, task_edge, args,
        y_pred_1st_order=None, y_pred_2nd_order=None, return_full_prob=False, **kwargs):
    '''
    :return: logits or confidence: B, T
    '''
    if y_pred_1st_order is None:
        y_pred_1st_order, y_pred_2nd_order = prediction_function(
            first_order_prediction_model=first_order_prediction_model,
            second_order_prediction_model=second_order_prediction_model,
            GNN_energy_model=GNN_energy_model,
            graph_repr=graph_repr, task_repr=task_repr, task_edge=task_edge,
            args=args
        )

    T = y_pred_1st_order.size()[1]
    if args.filling_missing_data_mode == 'no_filling':
        y_sample = ((softmax_opt(y_pred_1st_order)[..., 1]) > 0.5).long()  # B, T
    elif args.filling_missing_data_mode in ['mtl', 'mtl_uncertainty', 'gradnorm', 'dwa', 'lbtw', 'mtl_task', 'mtl_task_KG', 'gnn', 'ebm']:
        prior_prediction = kwargs['prior_prediction']
        y_sample = prior_prediction.detach().clone()
        y_sample = y_sample[kwargs['id']]
    else:
        raise ValueError('{} not implemented.'.format(args.filling_missing_data_mode))

    y_prior = None
    if args.filling_missing_data_mode in ['mtl', 'mtl_uncertainty', 'gradnorm', 'dwa', 'lbtw', 'mtl_task', 'mtl_task_KG', 'gnn', 'ebm'] and args.ebm_as_tilting:
        prior_prediction_logits = kwargs['prior_prediction_logits']
        y_prior = prior_prediction_logits.detach().clone()
        y_prior = y_prior[kwargs['id']]
    p = torch.rand_like(y_pred_1st_order)  # B, T, 2
    p_accum = 0

    for layer_idx in range(args.GS_iteration):
        y_sample, p = Gibbs_sampling(
            T=T, y_sample=y_sample, p=p, y_pred_1st_order=y_pred_1st_order, y_pred_2nd_order=y_pred_2nd_order,
            task_edge=task_edge, args=args, y_prior=y_prior)

        p_accum += p.clone()

    if args.GS_inference == 'last':
        p_accum = p
    elif args.GS_inference == 'average':
        p_accum /= args.GS_iteration

    if return_full_prob:
        return p_accum
    return p_accum[..., 1]


def GNN_EBM_1st_order_inference_Binary_Task(
        first_order_prediction_model, second_order_prediction_model, GNN_energy_model, prediction_function,
        graph_repr, task_repr, y_true, task_edge, args, **kwargs):
    y_pred, _ = prediction_function(
        first_order_prediction_model=first_order_prediction_model,
        second_order_prediction_model=second_order_prediction_model,
        GNN_energy_model=GNN_energy_model,
        graph_repr=graph_repr, task_repr=task_repr, task_edge=task_edge,
        args=args
    )
    y_pred = y_pred[..., 1]  # (B, T, 2) ===> (B, T)
    return y_pred

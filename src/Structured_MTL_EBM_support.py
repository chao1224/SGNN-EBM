import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add, scatter_mean

from configure import *

bce_criterion = nn.BCEWithLogitsLoss(reduction='none')
ce_criterion = nn.CrossEntropyLoss()
softmax_opt = nn.Softmax(-1)
sigmoid_opt = nn.Sigmoid()


def mapping_task_repr(task_repr, task_edge, B):
    '''
    Mapping task_repr to node1_task_repr and node2_task_repr, w.r.t. task_edge
    :param task_repr: T, d_task
    :param task_edge:  2, M
    :return: (T, M, d_task), (T, M, d_task)
    '''
    task_repr_2nd_order_node1 = torch.index_select(task_repr, 0, task_edge[0])
    task_repr_2nd_order_node2 = torch.index_select(task_repr, 0, task_edge[1])
    task_repr_2nd_order_node1 = task_repr_2nd_order_node1.unsqueeze(0).expand(B, -1, -1)  # B, M, d_task
    task_repr_2nd_order_node2 = task_repr_2nd_order_node2.unsqueeze(0).expand(B, -1, -1)  # B, M, d_task
    return task_repr_2nd_order_node1, task_repr_2nd_order_node2


def mapping_label(y_true, task_edge):
    '''
    [-1, -1] => 0
    [-1,  1] => 1
    [ 1, -1] => 2
    [ 1,  1] => 3
    '''
    y_true_node1 = torch.index_select(y_true, 1, task_edge[0])
    y_true_node2 = torch.index_select(y_true, 1, task_edge[1])

    y_true_2nd_order = ((2 * y_true_node1 + y_true_node2 + 3)/2).long().unsqueeze(2)
    return y_true_2nd_order


def mapping_label_02(y_true, task_edge):
    '''
    [0, 0] => 0
    [0, 1] => 1
    [1, 0] => 2
    [1, 1] => 3
    '''
    y_true_node1 = torch.index_select(y_true, 1, task_edge[0])
    y_true_node2 = torch.index_select(y_true, 1, task_edge[1])

    y_true_2nd_order = (2 * y_true_node1 + y_true_node2).long().unsqueeze(2)

    # count = {}
    # for a in y_true_2nd_order:
    #     for b in a:
    #         i = b.detach().item()
    #         if i not in count:
    #             count[i] = 0
    #         count[i] += 1
    # print('count\t', count)
    return y_true_2nd_order


def mapping_valid_label(y_valid, task_edge):
    y_valid_node1 = torch.index_select(y_valid, 1, task_edge[0])
    y_valid_node2 = torch.index_select(y_valid, 1, task_edge[1])

    y_valid_2nd_order = torch.logical_and(y_valid_node1, y_valid_node2).unsqueeze(2)
    return y_valid_2nd_order


def extract_amortized_task_label_weights(dataloader, task_edge, device, args):
    '''
    :return:
    first_order_label_weights (T, 2)
    second_order_label_weights (M, 4)
    '''
    T = args.num_tasks
    M = len(task_edge[0])
    print('T={} tasks, M={} task edges.'.format(T, M))
    first_order_label_weights = torch.zeros((T, 2), device=device)
    second_order_label_weights = torch.zeros((M, 4), device=device)
    first_order_valid_counts = torch.zeros(T, device=device)
    second_order_valid_counts = torch.zeros(M, device=device)

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            B = len(batch.id)
            y_true = batch.y.view(B, args.num_tasks).float()  # B, T
            y_true_first_order = ((1 + y_true) / 2).long()  # B, T
            y_valid_first_order = y_true ** 2 > 0  # B, T

            y_true_second_order = mapping_label(y_true, task_edge).squeeze()  # B, M
            y_valid_second_order = mapping_valid_label(y_valid_first_order, task_edge).squeeze()  # B, M

            for label in [0, 1]:
                batch_statistics = torch.logical_and(y_true_first_order == label, y_valid_first_order)
                first_order_label_weights[:, label] += batch_statistics.sum(0)

            for label in [0, 1, 2, 3]:
                batch_statistics = torch.logical_and(y_true_second_order == label, y_valid_second_order)
                second_order_label_weights[:, label] += batch_statistics.sum(0)

            first_order_valid_counts += y_valid_first_order.sum(0)
            second_order_valid_counts += y_valid_second_order.sum(0)

    for i in range(T):
        assert first_order_label_weights[i].sum(0) == first_order_valid_counts[i]

    for i in range(M):
        assert second_order_label_weights[i].sum(0) == second_order_valid_counts[i]

    first_order_label_weights /= first_order_valid_counts.unsqueeze(1)
    second_order_label_weights /= second_order_valid_counts.unsqueeze(1)

    return first_order_label_weights, second_order_label_weights


def get_EBM_prediction(
        first_order_prediction_model, second_order_prediction_model,
        graph_repr, task_repr, task_edge, args):
    B = len(graph_repr)
    M = len(task_edge[0])

    graph_repr_1st_order = graph_repr.unsqueeze(1).expand(-1, args.num_tasks, -1)  # B, T, d_mol
    task_repr_1st_order = task_repr.unsqueeze(0).expand(B, -1, -1)  # B, T, d_task
    y_pred_1st_order = first_order_prediction_model(graph_repr_1st_order, task_repr_1st_order)  # B, T, 2

    graph_repr_2nd_order = graph_repr.unsqueeze(1).expand(-1, M, -1)  # B, M, d_mol
    task_repr_2nd_order_node1, task_repr_2nd_order_node2 = mapping_task_repr(task_repr, task_edge, B)  # (B, M, d_task), (B, M, d_task)
    y_pred_2nd_order = second_order_prediction_model(graph_repr_2nd_order, task_repr_2nd_order_node1, task_repr_2nd_order_node2)  # B, M, 4

    return y_pred_1st_order, y_pred_2nd_order


def customized_loss(y_pred, y_true, args, weights=None):
    if args.energy_CD_loss == 'ce':
        loss = ce_criterion(y_pred, y_true)
        # y_true = y_true.unsqueeze(1)
        # pos_term = - torch.gather(y_pred, 1, y_true).squeeze(1)
        # neg_term = torch.log(torch.exp(y_pred).sum(-1))
        # ce_loss = pos_term + neg_term
        # ce_loss = ce_loss.mean()
    elif args.energy_CD_loss == 'weighted_ce':
        y_true = y_true.unsqueeze(1)
        pos_term = - torch.gather(y_pred, 1, y_true).squeeze(1)
        neg_term = torch.log((torch.exp(y_pred) * weights).sum(-1))
        loss = pos_term + neg_term
        loss = loss.mean()
    elif args.energy_CD_loss == 'raw':
        y_true = y_true.unsqueeze(1)
        pos_term = - torch.gather(y_pred, 1, y_true).squeeze(1)
        neg_term = (y_pred * weights).sum(1)
        loss = pos_term + neg_term
        loss = loss.mean()
    elif args.energy_CD_loss == 'smoothing':
        y_true = y_true.unsqueeze(1)
        pos_term = - torch.gather(y_pred, 1, y_true).squeeze(1)
        neg_term = (torch.exp(y_pred) * weights).sum(1)
        loss = pos_term + neg_term
        loss = loss.mean()
    else:
        raise ValueError('CD loss not included {}.'.format(args.energy_CD_loss))

    return loss


def energy_function_CD_AA(
        first_order_prediction_model, second_order_prediction_model,
        graph_repr, task_repr, y_true, task_edge,
        first_order_label_weights, second_order_label_weights, args, **kwargs):

    y_pred_1st_order, y_pred_2nd_order = get_EBM_prediction(
        first_order_prediction_model=first_order_prediction_model,
        second_order_prediction_model=second_order_prediction_model,
        graph_repr=graph_repr, task_repr=task_repr, task_edge=task_edge,
        args=args)

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

    B = len(graph_repr)
    first_order_label_weights = first_order_label_weights.unsqueeze(0).expand(B, -1, -1)
    second_order_label_weights = second_order_label_weights.unsqueeze(0).expand(B, -1, -1)
    masked_label_weights_1st_order = torch.masked_select(first_order_label_weights, y_valid_1st_order).view(-1, 2)
    masked_second_order_label_weights = torch.masked_select(second_order_label_weights, y_valid_2nd_order).view(-1, 4)

    first_order_energy_loss = customized_loss(masked_y_pred_1st_order, masked_y_true_1st_order, args, masked_label_weights_1st_order)
    second_order_energy_loss = customized_loss(masked_y_pred_2nd_order, masked_y_true_2nd_order, args, masked_second_order_label_weights)

    return y_pred_1st_order, y_pred_2nd_order, first_order_energy_loss, second_order_energy_loss


def Gibbs_sampling(T, y_sample, p, y_pred_1st_order, y_pred_2nd_order, task_edge, args, y_prior=None, **kwargs):
    def gather(y_sample_edge, filtered_y_pred_2nd_order):
        '''
        :param y_sample_edge: (B, k, 1)
        :param filtered_y_pred_2nd_order: (B, k, 4)
        :return:
        '''
        y_pred_2nd_order_ = torch.gather(filtered_y_pred_2nd_order, 2, y_sample_edge)  # (B, k, 4) (B, k, 1) => (B, k, 1)
        y_pred_2nd_order_ = y_pred_2nd_order_.sum(dim=1)  # B, 1
        return y_pred_2nd_order_

    with torch.no_grad():
        task_idx_list = torch.randperm(T)
        for i in task_idx_list:
            filtered_task_edge_index = (task_edge[0] == i).nonzero()  # k, 1
            filtered_y_pred_2nd_order = y_pred_2nd_order[:, filtered_task_edge_index.squeeze(1)]  # (B, T, 4) => (B, k, 4)

            y_sample_j_index = task_edge[1][filtered_task_edge_index].squeeze(1)  # k
            y_sample_j_label = (y_sample[:, y_sample_j_index]).unsqueeze(2)  # B, k, 1

            y_sample_edge_label_0 = y_sample_j_label  # B, k, 1
            y_sample_edge_label_1 = 2 + y_sample_j_label  # B, k, 1

            y_sample_pred_edge_0j = gather(y_sample_edge_label_0, filtered_y_pred_2nd_order)  # B, 1
            y_sample_pred_label_1j = gather(y_sample_edge_label_1, filtered_y_pred_2nd_order)  # B, 1
            y_pred_2nd_order_ = torch.cat([y_sample_pred_edge_0j, y_sample_pred_label_1j], dim=1)  # B, 2

            temp = y_pred_1st_order[:, i] + args.structured_lambda * y_pred_2nd_order_  # B, 2
            temp = softmax_opt(temp)  # B, 2
            # print('temp norm:', temp[0, 0])
            if y_prior is not None and args.ebm_as_tilting:
                prior = F.sigmoid(torch.stack([-y_prior[:, i], y_prior[:, i]], dim=1))
                temp *= prior
            # print('prior norm:', prior[0, 0])

            p[:, i] = temp
            y_sample[:, i] = (temp[..., 1] > torch.rand_like(temp[..., 1])).long()  # B

    return y_sample, p


def phase_energy_calculation(y_true, y_pred_1st_order, y_pred_2nd_order, y_valid, task_edge):
    def customized_energy_single_phase(y_pred, y_true, args):
        energy = -torch.gather(y_pred, 2, y_true).squeeze(1)
        energy = energy.sum()
        return energy
    
    y_true_1st_order = y_true.unsqueeze(2)  # B, T, 1
    y_true_2nd_order = mapping_label_02(y_true, task_edge)  # B, M, 1

    first_order_energy = customized_energy_single_phase(y_pred_1st_order, y_true_1st_order, args)
    second_order_energy = customized_energy_single_phase(y_pred_2nd_order, y_true_2nd_order, args)
    return first_order_energy, second_order_energy


def energy_function_CD_GS(
        first_order_prediction_model, second_order_prediction_model,
        graph_repr, task_repr, y_true, task_edge,
        first_order_label_weights, second_order_label_weights, args, **kwargs):

    y_pred_1st_order, y_pred_2nd_order = get_EBM_prediction(
        first_order_prediction_model=first_order_prediction_model,
        second_order_prediction_model=second_order_prediction_model,
        graph_repr=graph_repr, task_repr=task_repr, task_edge=task_edge,
        args=args)

    y_valid = y_true ** 2 > 0
    y_true_1st_order = ((1 + y_true) / 2).long()  # B, T, 1

    ########## Get negative phase from Gibbs Sampling ##########
    T = y_pred_1st_order.size()[1]
    if args.filling_missing_data_mode == 'no_filling':
        ########## No filling missing value, 0 for missing by default ##########
        # y_sample = (softmax_opt(y_pred_1st_order)[..., 1] > 0.5).long()  # B, T
        y_sample = ((1 + y_true) / 2).long()  # B, T
    elif args.filling_missing_data_mode in ['mtl_task', 'mtl_task_KG', 'gnn']:
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


def amortized_mean_field_inference_first_order(
        first_order_prediction_model, second_order_prediction_model, energy_function,
        graph_repr, task_repr, y_true, task_edge,
        first_order_label_weights, second_order_label_weights, args):
    '''
    :return: logits or confidence: B, T
    '''
    y_pred_1st_order, _ = get_EBM_prediction(
        first_order_prediction_model=first_order_prediction_model,
        second_order_prediction_model=second_order_prediction_model,
        graph_repr=graph_repr, task_repr=task_repr, task_edge=task_edge,
        args=args)
    y_pred_1st_order = y_pred_1st_order[..., 1]
    if args.amortized_logits_transform_to_confidence:
        y_pred_1st_order = torch.exp(y_pred_1st_order)
    y_pred = y_pred_1st_order
    return y_pred


def amortized_mean_field_inference_second_order(
        first_order_prediction_model, second_order_prediction_model, energy_function,
        graph_repr, task_repr, y_true, task_edge,
        first_order_label_weights, second_order_label_weights, args):
    '''
    :return: logits or confidence: B, T
    '''
    y_pred_1st_order, y_pred_2nd_order = get_EBM_prediction(
        first_order_prediction_model=first_order_prediction_model,
        second_order_prediction_model=second_order_prediction_model,
        graph_repr=graph_repr, task_repr=task_repr, task_edge=task_edge,
        args=args)

    T = y_pred_1st_order.size()[1]
    y_pred_1st_order = y_pred_1st_order[..., 1]
    if args.amortized_logits_transform_to_confidence:
        y_pred_1st_order = torch.exp(y_pred_1st_order)
        y_pred_2nd_order = torch.exp(y_pred_2nd_order)

    amortized_y_pred_2nd_order = y_pred_2nd_order[..., 2:].mean(2)  # B, M
    amortized_y_pred_2nd_order = scatter_mean(amortized_y_pred_2nd_order, task_edge[1], dim=1, dim_size=T)  # B, T
    y_pred = (y_pred_1st_order + args.structured_lambda * amortized_y_pred_2nd_order) / (1 + args.structured_lambda)
    return y_pred


def amortized_mean_field_inference_label_propagation_first_order(
        first_order_prediction_model, second_order_prediction_model, energy_function,
        graph_repr, task_repr, y_true, task_edge,
        first_order_label_weights, second_order_label_weights, args):
    '''
    :return: logits or confidence: B, T
    '''
    y_pred_1st_order, _ = get_EBM_prediction(
        first_order_prediction_model=first_order_prediction_model,
        second_order_prediction_model=second_order_prediction_model,
        graph_repr=graph_repr, task_repr=task_repr, task_edge=task_edge,
        args=args)

    node_in, node_out = task_edge[0], task_edge[1]
    T = y_pred_1st_order.size()[1]

    for _ in range(args.MFVI_iteration):
        message = y_pred_1st_order[:, node_in, :]  # B, M, 2
        update = scatter_mean(message, node_out, dim=1, dim_size=T)  # B, T, 2
        y_pred_1st_order = update + y_pred_1st_order

    if args.amortized_logits_transform_to_confidence:
        y_pred_1st_order = torch.exp(y_pred_1st_order)

    y_pred_1st_order = y_pred_1st_order[..., 1]
    y_pred = y_pred_1st_order
    return y_pred


def mean_field_variational_inference(
        first_order_prediction_model, second_order_prediction_model, energy_function,
        graph_repr, task_repr, y_true, task_edge,
        first_order_label_weights, second_order_label_weights, args):
    '''
    :return: logits or confidence: B, T
    '''
    # This is very similar to Mean-Field Variational Inference function, check that one
    y_pred_1st_order, y_pred_2nd_order = get_EBM_prediction(
        first_order_prediction_model=first_order_prediction_model,
        second_order_prediction_model=second_order_prediction_model,
        graph_repr=graph_repr, task_repr=task_repr, task_edge=task_edge,
        args=args)
    B = y_pred_1st_order.size()[0]
    T = y_pred_1st_order.size()[1]
    M = y_pred_2nd_order.size()[1]
    q = torch.abs(torch.randn_like(y_pred_1st_order))  # B, T, 2
    q = softmax_opt(q)  # B, T, 2

    aggregate_indice = torch.LongTensor([0, 0, 1, 1]).to(args.device)

    for _ in range(args.MFVI_iteration):
        q_i, q_j = q[:, task_edge[0]], q[:, task_edge[1]]  # (B,T,2) => (B,M,2)
        q_second_order = torch.einsum('bmxy,bmyz->bmxz', q_i.unsqueeze(3), q_j.unsqueeze(2))  # B, M, 2, 2
        q_second_order = q_second_order.view(B, M, 4)  # B, M, 4

        aggregated_y_pred_2nd_order = scatter_add(q_second_order*y_pred_2nd_order, task_edge[0], dim=1, dim_size=T)  # B, T, 4
        y_pred_2nd_order_ = scatter_add(aggregated_y_pred_2nd_order, aggregate_indice, dim=2)  # B, T, 2

        q = y_pred_1st_order + args.structured_lambda * y_pred_2nd_order_
        q = softmax_opt(q)

    return q[..., 1]


def GS_inference(
        first_order_prediction_model, second_order_prediction_model, energy_function,
        graph_repr, task_repr, y_true, task_edge,
        first_order_label_weights, second_order_label_weights, args):
    '''
    :return: logits or confidence: B, T
    '''

    y_pred_1st_order, y_pred_2nd_order = get_EBM_prediction(
        first_order_prediction_model=first_order_prediction_model,
        second_order_prediction_model=second_order_prediction_model,
        graph_repr=graph_repr, task_repr=task_repr, task_edge=task_edge,
        args=args)

    T = y_pred_1st_order.size()[1]
    if args.filling_missing_data_mode == 'no_filling':
        y_sample = ((softmax_opt(y_pred_1st_order)[..., 1]) > 0.5).long()  # B, T
    else:
        raise ValueError('{} not implemented.'.format(args.filling_missing_data_mode))
    p = torch.rand_like(y_pred_1st_order)  # B, T, 2
    p_accum = 0

    for layer_idx in range(args.GS_iteration):
        y_sample, p = Gibbs_sampling(
            T=T, y_sample=y_sample, p=p, y_pred_1st_order=y_pred_1st_order, y_pred_2nd_order=y_pred_2nd_order,
            task_edge=task_edge, args=args)

        p_accum += p.clone()

    if args.GS_inference == 'last':
        p_accum = p
    elif args.GS_inference == 'average':
        p_accum /= args.GS_iteration
    return p_accum[..., 1]


def GS_inference_legacy(
        first_order_prediction_model, second_order_prediction_model, energy_function,
        graph_repr, task_repr, y_true, task_edge,
        first_order_label_weights, second_order_label_weights, args):
    '''
    :return: logits or confidence: B, T
    '''

    def gather(y_sample_edge):
        y_sample_2nd_order_ = torch.gather(y_pred_2nd_order, 2, y_sample_edge.unsqueeze(2))  # (B, M, 4) => (B, M, 1)
        y_pred_2nd_order_ = scatter_add(y_sample_2nd_order_, task_edge[0], dim=1, dim_size=T)  # (B, M, 1) => (B, T, 1)
        return y_pred_2nd_order_

    y_pred_1st_order, y_pred_2nd_order = get_EBM_prediction(
        first_order_prediction_model=first_order_prediction_model,
        second_order_prediction_model=second_order_prediction_model,
        graph_repr=graph_repr, task_repr=task_repr, task_edge=task_edge,
        args=args)

    T = y_pred_1st_order.size()[1]
    y_sample = (softmax_opt(y_pred_1st_order)[..., 1] > 0.5).long()  # B, T
    p = torch.ones_like(y_pred_1st_order)  # B, T, 2
    p_accum = 0

    for _ in range(args.GS_iteration):
        task_idx_list = torch.randperm(T)
        for i in task_idx_list:
            y_sample_pred_i, y_sample_pred_j = y_sample[:, task_edge[0]], y_sample[:, task_edge[1]]  # B, M
            y_sample_edge_label_0 = y_sample_pred_j  # B, M
            y_sample_edge_label_1 = 2 + y_sample_pred_j  # B, M

            y_pred_2nd_order_0 = gather(y_sample_edge_label_0)  # B, T, 1
            y_pred_2nd_order_1 = gather(y_sample_edge_label_1)  # B, T, 1

            y_pred_2nd_order_ = torch.cat([y_pred_2nd_order_0, y_pred_2nd_order_1], dim=2)  # B, T, 2

            temp = y_pred_1st_order[:, i] + args.structured_lambda * y_pred_2nd_order_[:, i]  # B, 2
            p[:, i] = softmax_opt(temp)
            y_sample[:, i] = (p[:, i, 1] > torch.rand_like(p[:, i, 1])).long()  # B

        p_accum += p.clone()

    if args.GS_inference == 'last':
        p_accum = p
    elif args.GS_inference == 'average':
        p_accum /= args.GS_iteration
    else:
        raise ValueError('GS inference {} not included.'.format(args.GS_inference))
    return p_accum[..., 1]


def mean_field_variation_inference_test(
        first_order_prediction_model, second_order_prediction_model, energy_function,
        graph_repr, task_repr, y_true, task_edge,
        first_order_label_weights, second_order_label_weights, args):
    '''
    :return: logits or confidence: B, T
    '''
    # This is very similar to Mean-Field Variational Inference function, check that one
    y_pred_1st_order, y_pred_2nd_order = get_EBM_prediction(
        first_order_prediction_model=first_order_prediction_model,
        second_order_prediction_model=second_order_prediction_model,
        graph_repr=graph_repr, task_repr=task_repr, task_edge=task_edge,
        args=args)
    B = y_pred_1st_order.size()[0]
    T = y_pred_1st_order.size()[1]
    M = y_pred_2nd_order.size()[1]
    q = torch.abs(torch.randn_like(y_pred_1st_order))  # B, T, 2
    q = softmax_opt(q)  # B, T, 2

    aggregate_indice = torch.LongTensor([0, 0, 1, 1]).to(args.device)

    # aggregated_y_pred_2nd_order = scatter_add(y_pred_2nd_order, task_edge[1], dim=1, dim_size=T)  # B, T, 4

    for i in range(M):
        if task_edge[0,i] == 0:
            print('=====\t', i, task_edge[0,i], task_edge[1,i])

    for _ in range(args.MFVI_iteration):
        q_i, q_j = q[:, task_edge[0]], q[:, task_edge[1]]  # (B,T,2) => (B,M,2)
        q_second_order = torch.einsum('bmxy,bmyz->bmxz', q_i.unsqueeze(3), q_j.unsqueeze(2))  # B, M, 2, 2
        q_second_order = q_second_order.view(B, M, 4)  # B, M, 4
        print('q_second_order\t', q_second_order.size())

        ######### For debugging #########
        print('0,0\t', q[0, 0])
        print('0,46\t', q[0, 46])
        print('0,129\t', q[0, 129])
        print('0-46', q_second_order[0, 0])
        # print('46-0', q_second_order[0, 1])
        print('0-129', q_second_order[0, 2])
        # print('129-0', q_second_order[0, 3])

        temp = scatter_add(q_second_order, task_edge[0], dim=1, dim_size=T)  # B, T, 4
        print('agg 0 (46+129)', temp[0, 0])
        temp = scatter_add(temp, aggregate_indice, dim=2)
        print('agg 0 (46+129)', temp[0, 0])

        aggregated_y_pred_2nd_order = scatter_add(q_second_order*y_pred_2nd_order, task_edge[0], dim=1, dim_size=T)  # B, T, 4
        print('aggregated_y_pred_2nd_order\t', aggregated_y_pred_2nd_order.size())
        y_pred_2nd_order_ = scatter_add(aggregated_y_pred_2nd_order, aggregate_indice, dim=2)  # B, T, 2
        print('y_pred_2nd_order_\t', y_pred_2nd_order_.size())

        q = y_pred_1st_order + args.structured_lambda * y_pred_2nd_order_
        q = softmax_opt(q)

    return q[..., 1]


def GS_inference_match_testing(
        first_order_prediction_model, second_order_prediction_model, energy_function,
        graph_repr, task_repr, y_true, task_edge,
        first_order_label_weights, second_order_label_weights, args):
    '''
    :return: logits or confidence: B, T
    '''

    def gather(y_sample_edge, filtered_y_pred_2nd_order):
        '''
        :param y_sample_edge: (B, k, 1)
        :param filtered_y_pred_2nd_order: (B, k, 4)
        :return:
        '''
        y_pred_2nd_order_ = torch.gather(filtered_y_pred_2nd_order, 2, y_sample_edge)  # (B, k, 4) (B, k, 1) => (B, k, 1)
        y_pred_2nd_order_ = y_pred_2nd_order_.sum(dim=1)  # B, 1
        return y_pred_2nd_order_

    y_pred_1st_order, y_pred_2nd_order = get_EBM_prediction(
        first_order_prediction_model=first_order_prediction_model,
        second_order_prediction_model=second_order_prediction_model,
        graph_repr=graph_repr, task_repr=task_repr, task_edge=task_edge,
        args=args)

    T = y_pred_1st_order.size()[1]
    y_sample = (softmax_opt(y_pred_1st_order)[..., 1] > 0.5).long()  # B, T
    p = torch.rand_like(y_pred_1st_order)  # B, T, 2
    aggregated_y_pred_2nd_order = scatter_add(y_pred_2nd_order, task_edge[0], dim=1, dim_size=T)  # B, T, 4
    p_accum = 0

    def test():
        def gather(y_sample_edge):
            y_sample_2nd_order_ = torch.gather(y_pred_2nd_order, 2, y_sample_edge.unsqueeze(2))  # (B, M, 4) => (B, M, 1)
            y_pred_2nd_order_ = scatter_add(y_sample_2nd_order_, task_edge[0], dim=1, dim_size=T)  # (B, M, 1) => (B, T, 1)
            return y_pred_2nd_order_

        print('Scatter')
        print('y_pred (0,46), (46,0), (0,129), (129,0)\n', y_pred_2nd_order[0, 0:4])

        y_sample_pred_i, y_sample_pred_j = y_sample[:, task_edge[0]], y_sample[:, task_edge[1]]  # B, M
        y_sample_edge_label_0 = y_sample_pred_j  # B, M
        y_sample_edge_label_1 = 2 + y_sample_pred_j  # B, M
        print('y_sample_edge_label_0 (0,46), (46,0), (0,129), (129,0)\t', y_sample_edge_label_0[0, 0:4])
        print('y_sample_edge_label_1 (0,46), (46,0), (0,129), (129,0)\t', y_sample_edge_label_1[0, 0:4])

        y_pred_2nd_order_0 = gather(y_sample_edge_label_0)  # B, T, 1
        y_pred_2nd_order_1 = gather(y_sample_edge_label_1)  # B, T, 1
        print('y_pred_2nd_order_0\ty0=0 (46+129)\t', y_pred_2nd_order_0[0, 0])
        print('y_pred_2nd_order_0\ty0=1 (46+129)\t', y_pred_2nd_order_1[0, 0])

        y_pred_2nd_order_ = torch.cat([y_pred_2nd_order_0, y_pred_2nd_order_1], dim=2)  # B, T, 2
        print('scatter\ty_pred_2nd_order_ [0,0]\t', y_pred_2nd_order_[0, 0])

        temp = y_pred_1st_order[:, i] + args.structured_lambda * y_pred_2nd_order_[:, i]
        temp = softmax_opt(temp)

        return temp

    for layer_idx in range(args.GS_iteration):
        task_idx_list = torch.randperm(T)
        for i in task_idx_list:
            i = 0
            print('Exact')
            filtered_task_edge_index = (task_edge[0] == i).nonzero()  # k, 1
            filtered_y_pred_2nd_order = y_pred_2nd_order[:, filtered_task_edge_index.squeeze(1)]  # (B, T, 4) => (B, k, 4)
            print('filtered_task_edge_index\t', filtered_task_edge_index.squeeze())
            print('filtered_y_pred_2nd_order\n', filtered_y_pred_2nd_order[0])
            y_sample_j_index = task_edge[1][filtered_task_edge_index].squeeze(1)  # k
            y_sample_j_label = (y_sample[:, y_sample_j_index]).unsqueeze(2)  # B, k, 1

            y_sample_edge_label_0 = y_sample_j_label  # B, k, 1
            y_sample_edge_label_1 = 2 + y_sample_j_label  # B, k, 1

            y_sample_pred_edge_0j = gather(y_sample_edge_label_0, filtered_y_pred_2nd_order)  # B, 1
            y_sample_pred_label_1j = gather(y_sample_edge_label_1, filtered_y_pred_2nd_order)  # B, 1

            print('y_sample_edge_label_0\t', y_sample_edge_label_0[0].squeeze(), '\ty_sample_pred_edge_0j\t', y_sample_pred_edge_0j[0])
            print('y_sample_edge_label_1\t', y_sample_edge_label_1[0].squeeze(), '\ty_sample_pred_label_1j\t', y_sample_pred_label_1j[0])

            y_pred_2nd_order_ = torch.cat([y_sample_pred_edge_0j, y_sample_pred_label_1j], dim=1)  # B, 2
            print('exact\ty_pred_2nd_order_ [0]\t', y_pred_2nd_order_[0])
            print()

            temp = y_pred_1st_order[:, i] + args.structured_lambda * y_pred_2nd_order_  # B, 2
            temp = softmax_opt(temp)  # B, 2
            tempp = test()
            print()
            print('Check\t', temp[0], '\t', tempp[0])
            print()

            p[:, i] = temp
            y_sample[:, i] = (temp[..., 1] > torch.rand_like(temp[..., 1])).long()  # B

            break

        if layer_idx > 0:
            p_accum += p

    p_accum /= (args.GS_iteration - 1)
    return p_accum[..., 1]


def SGLD_inference():
    return


if __name__ == '__main__':
    edge = torch.LongTensor(
        [
            [0, 1, 1, 2, 0, 3],
            [1, 0, 2, 1, 3, 0]
        ]
    )
    x = torch.LongTensor([[0, 1, 2, 3, 4, 5]])
    x = torch.LongTensor([[0, 0, 2, 2, 4, 4]])
    print(x, '\n')

    print(edge[0])
    print(edge[1])

    ans = scatter_add(x, edge[0], dim=1)
    print('scatter along 0\t', ans, '\n')

    ans = scatter_add(x, edge[1], dim=1)
    print('scatter along 1\t', ans, '\n')

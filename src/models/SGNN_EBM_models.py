import torch
from torch import nn
import torch.nn.functional as F
from torch_scatter import scatter_add


class NCE_C_Parameter(torch.nn.Module):
    def __init__(self, N):
        super(NCE_C_Parameter, self).__init__()
        self.NCE_C = nn.Parameter(torch.zeros(N, requires_grad=True))


class GNN_EBM_Layer_01(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GNN_EBM_Layer_01, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.edge_layer = torch.nn.Linear(input_dim, output_dim)
        self.node_layer = torch.nn.Linear(input_dim, output_dim)
        self.mlp = torch.nn.Linear(input_dim, output_dim)

    def node_message_passing(self, x, x_2nd_agg, edge):
        T = x.size()[1]
        node_in, node_out = edge[0], edge[1]  # M, M

        update = (scatter_add(x_2nd_agg, node_out, dim=1, dim_size=T) +
                  scatter_add(x_2nd_agg, node_in, dim=1, dim_size=T)) / 2  # B, T, d
        x = x + update  # B, T, d

        return x

    def forward(self, x_1st, x_2nd, edge):
        '''
        :param x: (B, T, 2, d)
        :param x_2nd: (B, M, 4, d)
        :param edge: (M, 2)
        :return: (B, T, 2, d_out)
        '''
        aggregate_indice = torch.LongTensor([0, 0, 1, 1]).to(x_1st.device)
        node_i_indice = torch.LongTensor([0, 0, 1, 1]).to(x_1st.device)
        node_j_indice = torch.LongTensor([0, 1, 0, 1]).to(x_1st.device)

        x_1st_neg = x_1st[:, :, 0, :]  # B, T, d
        x_1st_pos = x_1st[:, :, 1, :]  # B, T, d

        x_2nd_agg = scatter_add(x_2nd, aggregate_indice, dim=2)  # B, T, 2, d
        x_2nd_neg = x_2nd_agg[:, :, 0, :]  # B, M, d
        x_2nd_pos = x_2nd_agg[:, :, 1, :]  # B, M, d

        x_neg = self.node_message_passing(x_1st_neg, x_2nd_neg, edge)  # B, T, d
        x_pos = self.node_message_passing(x_1st_pos, x_2nd_pos, edge)  # B, T, d
        x = torch.stack([x_neg, x_pos], dim=2)  # B, T, 2, d
        x = self.node_layer(x)  # B, T, 2, d

        edge_i = torch.index_select(x_1st, 1, edge[0])  # B, M, 2, dim
        edge_i = torch.index_select(edge_i, 2, node_i_indice)  # B, M, 4, dim

        edge_j = torch.index_select(x_1st, 1, edge[1])  # B, M, 2, dim
        edge_j = torch.index_select(edge_j, 2, node_j_indice)  # B, M, 4, dim

        edge = x_2nd + self.mlp(edge_i + edge_j)  # B, M, 4, d
        edge = self.edge_layer(edge)

        return x, edge


class GNN_Energy_Model_1st_Order_01(torch.nn.Module):
    def __init__(self, ebm_GNN_dim, ebm_GNN_layer_num, output_dim, dropout=0, concat=False):
        super(GNN_Energy_Model_1st_Order_01, self).__init__()
        self.ebm_GNN_dim = ebm_GNN_dim
        self.ebm_GNN_layer_num = ebm_GNN_layer_num - 1
        self.dropout = dropout
        self.output_dim = output_dim
        self.concat = concat

        hidden_layers_dim = [ebm_GNN_dim] * ebm_GNN_layer_num

        self.hidden_layers = torch.nn.ModuleList()
        for in_, out_ in zip(hidden_layers_dim[:-1], hidden_layers_dim[1:]):
            self.hidden_layers.append(GNN_EBM_Layer_01(in_, out_))

        if self.concat:
            hidden_dim_sum = sum(hidden_layers_dim)
        else:
            hidden_dim_sum = ebm_GNN_dim
        self.node_readout = torch.nn.Sequential(
            torch.nn.Linear(2 * hidden_dim_sum, 2 * hidden_dim_sum),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * hidden_dim_sum, hidden_dim_sum),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim_sum, output_dim)
        )
        return

    def forward(self, x_1st, x_2nd, edge):
        '''
        :param x_1st: B,T,2,dim
        :param x_2nd: B,M,4,dim
        :param edge: 2,M
        :return: B,T,1
        '''
        B, T = x_1st.size()[:2]
        h_node_list = [x_1st]
        x_node, x_edge = x_1st, x_2nd

        for i in range(self.ebm_GNN_layer_num):
            x_node, x_edge = self.hidden_layers[i](x_node, x_edge, edge)
            if i < self.ebm_GNN_layer_num - 1:
                x_node = F.relu(x_node)
                # x_edge = F.relu(x_edge)
            x_node = F.dropout(x_node, self.dropout, training=self.training)
            # x_edge = F.dropout(x_edge, self.dropout, training=self.training)
            h_node_list.append(x_node)

        if self.concat:
            h = torch.cat(h_node_list, dim=3).view(B, T, -1)  # B, T, 2*layer_num*d
        else:
            h = x_node.view(B, T, -1)  # B, T, 2*d
        h = self.node_readout(h)  # B, T, 1
        return h


class GNN_Energy_Model_1st_Order_02(torch.nn.Module):
    def __init__(self, ebm_GNN_dim, ebm_GNN_layer_num, output_dim, dropout=0, concat=False):
        super(GNN_Energy_Model_1st_Order_02, self).__init__()
        self.ebm_GNN_dim = ebm_GNN_dim
        self.ebm_GNN_layer_num = ebm_GNN_layer_num - 1
        self.dropout = dropout
        self.output_dim = output_dim
        self.concat = concat

        hidden_layers_dim = [ebm_GNN_dim] * ebm_GNN_layer_num

        self.hidden_layers = torch.nn.ModuleList()
        for in_, out_ in zip(hidden_layers_dim[:-1], hidden_layers_dim[1:]):
            self.hidden_layers.append(GNN_EBM_Layer_01(in_, out_))

        if self.concat:
            hidden_dim_sum = sum(hidden_layers_dim)
        else:
            hidden_dim_sum = ebm_GNN_dim
        self.node_readout = torch.nn.Linear(2 * hidden_dim_sum, output_dim)
        return

    def forward(self, x_1st, x_2nd, edge):
        '''
        :param x_1st: B,T,2,dim
        :param x_2nd: B,M,4,dim
        :param edge: 2,M
        :return: B,T,1
        '''
        B, T = x_1st.size()[:2]
        h_node_list = [x_1st]
        x_node, x_edge = x_1st, x_2nd

        for i in range(self.ebm_GNN_layer_num):
            x_node, x_edge = self.hidden_layers[i](x_node, x_edge, edge)
            if i < self.ebm_GNN_layer_num - 1:
                x_node = F.relu(x_node)
                # x_edge = F.relu(x_edge)
            x_node = F.dropout(x_node, self.dropout, training=self.training)
            # x_edge = F.dropout(x_edge, self.dropout, training=self.training)
            h_node_list.append(x_node)

        if self.concat:
            h = torch.cat(h_node_list, dim=3).view(B, T, -1)  # B, T, 2*layer_num*d
        else:
            h = x_node.view(B, T, -1)  # B, T, 2*d
        h = self.node_readout(h)  # B, T, 1
        return h


class GNN_Energy_Model_2nd_Order_01(torch.nn.Module):
    def __init__(self, ebm_GNN_dim, ebm_GNN_layer_num, dropout=0, concat=False):
        super(GNN_Energy_Model_2nd_Order_01, self).__init__()
        self.ebm_GNN_dim = ebm_GNN_dim
        self.ebm_GNN_layer_num = ebm_GNN_layer_num - 1
        self.dropout = dropout
        self.concat = concat

        hidden_layers_dim = [ebm_GNN_dim] * ebm_GNN_layer_num

        self.hidden_layers = torch.nn.ModuleList()
        for in_, out_ in zip(hidden_layers_dim[:-1], hidden_layers_dim[1:]):
            self.hidden_layers.append(GNN_EBM_Layer_01(in_, out_))

        if self.concat:
            hidden_dim_sum = sum(hidden_layers_dim)
        else:
            hidden_dim_sum = ebm_GNN_dim
        self.node_readout = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim_sum, 2 * hidden_dim_sum),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * hidden_dim_sum, hidden_dim_sum),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim_sum, 1)
        )
        self.edge_readout = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim_sum, 2 * hidden_dim_sum),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * hidden_dim_sum, hidden_dim_sum),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim_sum, 1)
        )
        return

    def forward(self, x_1st, x_2nd, edge):
        '''
        :param x_1st: B,T,2,dim
        :param x_2nd: B,M,4,dim
        :param edge: 2,M
        :return: (B,T,2), (B,M,4)
        '''
        B, T = x_1st.size()[:2]
        M = edge.size()[1]
        h_node_list = [x_1st]
        h_edge_list = [x_2nd]
        x_node, x_edge = x_1st, x_2nd

        for i in range(self.ebm_GNN_layer_num):
            x_node, x_edge = self.hidden_layers[i](x_node, x_edge, edge)
            if i < self.ebm_GNN_layer_num - 1:
                x_node = F.relu(x_node)
                # x_edge = F.relu(x_edge)
            x_node = F.dropout(x_node, self.dropout, training=self.training)
            # x_edge = F.dropout(x_edge, self.dropout, training=self.training)
            h_node_list.append(x_node)
            h_edge_list.append(x_edge)

        if self.concat:
            h_node = torch.cat(h_node_list, dim=3)  # B, T, 2, layer_num*d
            h_edge = torch.cat(h_edge_list, dim=3)  # B, M, 4, layer_num*d
        else:
            h_node = x_node  # B, T, 2, d
            h_edge = x_edge  # B, M, 4, d
        h_node = self.node_readout(h_node)  # B, T, 2, 1
        h_edge = self.edge_readout(h_edge)  # B, M, 4, 1
        h_node = h_node.squeeze(3)  # B, T, 2
        h_edge = h_edge.squeeze(3)  # B, M, 4
        return h_node, h_edge


class GNN_Energy_Model_2nd_Order_02(torch.nn.Module):
    def __init__(self, ebm_GNN_dim, ebm_GNN_layer_num, dropout=0, concat=False):
        super(GNN_Energy_Model_2nd_Order_02, self).__init__()
        self.ebm_GNN_dim = ebm_GNN_dim
        self.ebm_GNN_layer_num = ebm_GNN_layer_num - 1
        self.dropout = dropout
        self.concat = concat

        hidden_layers_dim = [ebm_GNN_dim] * ebm_GNN_layer_num

        self.hidden_layers = torch.nn.ModuleList()
        for in_, out_ in zip(hidden_layers_dim[:-1], hidden_layers_dim[1:]):
            self.hidden_layers.append(GNN_EBM_Layer_01(in_, out_))

        if self.concat:
            hidden_dim_sum = sum(hidden_layers_dim)
        else:
            hidden_dim_sum = ebm_GNN_dim
        self.node_readout = torch.nn.Sequential(
            torch.nn.Linear(2 * hidden_dim_sum, 2 * hidden_dim_sum),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * hidden_dim_sum, hidden_dim_sum),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim_sum, 2)
        )
        self.edge_readout = torch.nn.Sequential(
            torch.nn.Linear(4 * hidden_dim_sum, 2 * hidden_dim_sum),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * hidden_dim_sum, hidden_dim_sum),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim_sum, 4)
        )
        return

    def forward(self, x_1st, x_2nd, edge):
        '''
        :param x_1st: B,T,2,dim
        :param x_2nd: B,M,4,dim
        :param edge: 2,M
        :return: (B,T,2), (B,M,4)
        '''
        B, T = x_1st.size()[:2]
        M = x_2nd.size()[1]
        h_node_list = [x_1st]
        h_edge_list = [x_2nd]
        x_node, x_edge = x_1st, x_2nd

        for i in range(self.ebm_GNN_layer_num):
            x_node, x_edge = self.hidden_layers[i](x_node, x_edge, edge)
            if i < self.ebm_GNN_layer_num - 1:
                x_node = F.relu(x_node)
                # x_edge = F.relu(x_edge)
            x_node = F.dropout(x_node, self.dropout, training=self.training)
            # x_edge = F.dropout(x_edge, self.dropout, training=self.training)
            h_node_list.append(x_node)
            h_edge_list.append(x_edge)

        if self.concat:
            h_node = torch.cat(h_node_list, dim=3).view(B, T, -1)  # B, T, 2*layer_num*d
            h_edge = torch.cat(h_edge_list, dim=3).view(B, M, -1)  # B, M, 4*layer_num*d
        else:
            h_node = x_node.view(B, T, -1)  # B, T, 2*d
            h_edge = x_edge.view(B, M, -1)  # B, M, 4*d
        h_node = self.node_readout(h_node)  # B, T, 2
        h_edge = self.edge_readout(h_edge)  # B, M, 4
        return h_node, h_edge


class GNN_Energy_Model_2nd_Order_03(torch.nn.Module):
    def __init__(self, ebm_GNN_dim, ebm_GNN_layer_num, dropout=0, concat=False):
        super(GNN_Energy_Model_2nd_Order_03, self).__init__()
        self.ebm_GNN_dim = ebm_GNN_dim
        self.ebm_GNN_layer_num = ebm_GNN_layer_num - 1
        self.dropout = dropout
        self.concat = concat

        hidden_layers_dim = [ebm_GNN_dim] * ebm_GNN_layer_num

        self.hidden_layers = torch.nn.ModuleList()
        for in_, out_ in zip(hidden_layers_dim[:-1], hidden_layers_dim[1:]):
            self.hidden_layers.append(GNN_EBM_Layer_01(in_, out_))

        if self.concat:
            hidden_dim_sum = sum(hidden_layers_dim)
        else:
            hidden_dim_sum = ebm_GNN_dim

        self.node_readout = nn.Linear(2 * hidden_dim_sum, 2)
        self.edge_readout = nn.Linear(4 * hidden_dim_sum, 4)
        return

    def forward(self, x_1st, x_2nd, edge):
        '''
        :param x_1st: B,T,2,dim
        :param x_2nd: B,M,4,dim
        :param edge: 2,M
        :return: (B,T,2), (B,M,4)
        '''
        B, T = x_1st.size()[:2]
        M = edge.size()[1]
        h_node_list = [x_1st]
        h_edge_list = [x_2nd]
        x_node, x_edge = x_1st, x_2nd

        for i in range(self.ebm_GNN_layer_num):
            x_node, x_edge = self.hidden_layers[i](x_node, x_edge, edge)
            if i < self.ebm_GNN_layer_num - 1:
                x_node = F.relu(x_node)
                # x_edge = F.relu(x_edge)
            x_node = F.dropout(x_node, self.dropout, training=self.training)
            # x_edge = F.dropout(x_edge, self.dropout, training=self.training)
            h_node_list.append(x_node)
            h_edge_list.append(x_edge)

        if self.concat:
            h_node = torch.cat(h_node_list, dim=3)  # B, T, 2, layer_num*d
            h_edge = torch.cat(h_edge_list, dim=3)  # B, M, 4, layer_num*d
        else:
            h_node = x_node  # B, T, 2, d
            h_edge = x_edge  # B, M, 4, d

        h_node = h_node.view(B, T, -1)  # B, T, 2*d
        h_edge = h_edge.view(B, M, -1)  # B, M, 4*d

        h_node = self.node_readout(h_node)  # B, T, 2
        h_edge = self.edge_readout(h_edge)  # B, M, 4
        return h_node, h_edge


# class GATNet(torch.nn.Module):
#     def __init__(self, embedding_dim=10, hidden_dim=10, num_head=8):
#         super(GATNet, self).__init__()
#         self.conv1 = GATConv(embedding_dim, hidden_dim, heads=num_head, dropout=0.6)
#         self.conv2 = GATConv(hidden_dim * num_head, hidden_dim, heads=1, concat=False, dropout=0.6)

#     def forward(self, data):
#         x = data.x
#         x = F.dropout(x, p=0.6, training=self.training)
#         x = F.elu(self.conv1(x, data.edge_index))
#         x = F.dropout(x, p=0.6, training=self.training)
#         x = self.conv2(x, data.edge_index)
#         return x


# class MLP(nn.Sequential):
#     def __init__(self, input_dim, output_dim, hidden_dims=[1024, 512], dropout=0.1, use_batch_norm=False):
#         super(MLP, self).__init__()

#         self.input_dim = input_dim
#         self.output_dim = output_dim
#         self.hidden_dims = hidden_dims
#         self.use_batch_norm = use_batch_norm
#         self.dropout = nn.Dropout(0.1)

#         self.layer_size = len(self.hidden_dims) + 1
#         dims = [self.input_dim] + self.hidden_dims + [self.output_dim]

#         self.predictor = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(self.layer_size)])
#         if use_batch_norm:
#             self.batch_norms = nn.ModuleList([nn.BatchNorm1d(dims[i + 1]) for i in range(self.layer_size)])
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.xavier_uniform_(m.weight.data)
#                 if m.bias is not None:
#                     m.bias.data.fill_(0.0)

#     def norm(self):
#         with torch.no_grad():
#             norm = 0
#             for m in self.modules():
#                 if isinstance(m, nn.Linear):
#                     norm += torch.norm(m.weight.data).item()
#         return norm

#     def forward(self, v):
#         '''
#             : params x: (batch_size, *, input_dim)
#             : output : (batch_size, *, output_dim)
#         '''
#         B, t, _ = v.size()
#         v = v.flatten(0, -2)
#         # print('input norm: %.5f' % (torch.norm(v).item()))
#         for i, l in enumerate(self.predictor):
#             v = l(v)
#             if i != self.layer_size - 1:
#                 if self.use_batch_norm:
#                     v = self.batch_norms[i](v)
#                 v = F.relu(v)
#                 v = self.dropout(v)
#             # print('layer %d norm: %.5f' % (i, torch.norm(v).item()))
#         v = v.reshape(B, t, -1)
#         return v


# class GradKnowledgeGraphModel(nn.Module):
#     def __init__(self, num_tasks, args):
#         super(GradKnowledgeGraphModel, self).__init__()

#         self.num_tasks = num_tasks

#         self.weights = nn.Parameter(torch.ones(self.num_tasks, 1), requires_grad=True)
#         self.register_parameter('grad_KG', self.weights)
#         self.softmax = nn.Softmax(dim=0)
#         self.normalize_method = args.grad_KG_normalize_method

#     def forward(self, task_repr):
#         # ########## This won't train ##########
#         # task_repr = task_repr * self.weights.data
#         task_repr = task_repr * self.weights
#         return task_repr

#     def renormalize(self):
#         if self.normalize_method == 'sum':
#             ########## TODO: there might be negatives after backward ##########
#             normalize_coeff = self.num_tasks / self.weights.data.sum()
#             self.weights.data *= normalize_coeff
#         elif self.normalize_method == 'softmax':
#             self.weights.data = self.softmax(self.weights.data) * self.num_tasks
#         return

#     def reset_param(self):
#         self.weights.data.fill_(1)
#         return

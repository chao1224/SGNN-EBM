import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCNNet(torch.nn.Module):
    def __init__(self, embedding_dim=10, hidden_dim=10, dropout=0):
        super(GCNNet, self).__init__()
        self.dropout = dropout
        self.conv1 = GCNConv(embedding_dim, hidden_dim, cached=True, normalize=True)
        self.conv2 = GCNConv(hidden_dim, hidden_dim, cached=True, normalize=True)

    def forward(self, x, edge_index):
        x = x + self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = x + self.conv2(x, edge_index)
        return x


class TaskEmbeddingModel(nn.Sequential):
    def __init__(self, num_task, embedding_dim):
        super(TaskEmbeddingModel, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=num_task, embedding_dim=embedding_dim)
        self.linear = nn.Linear(embedding_dim, embedding_dim)
        return

    def forward(self, x):
        x = self.embedding(x)
        x = self.linear(x)
        return x


class TaskEmbeddingModel_BinaryEmbedding(nn.Sequential):
    def __init__(self, num_task, embedding_dim):
        super(TaskEmbeddingModel_BinaryEmbedding, self).__init__()
        self.embedding_dim = embedding_dim
        self.embedding_neg = nn.Embedding(num_embeddings=num_task, embedding_dim=embedding_dim)
        self.embedding_pos = nn.Embedding(num_embeddings=num_task, embedding_dim=embedding_dim)
        self.linear = nn.Linear(embedding_dim, embedding_dim)
        return

    def forward(self, x):
        x_neg = self.embedding_neg(x)
        x_pos = self.embedding_pos(x)
        x = torch.stack([x_neg, x_pos], dim=1)  # T, 2, dim
        x = self.linear(x)  # T, 2, dim
        return x


class MoleculeTaskPredictionModel(nn.Sequential):
    def __init__(self, data_emb_dim, task_emb_dim, dropout=0.1, output_dim=1, batch_norm=False):
        super(MoleculeTaskPredictionModel, self).__init__()

        self.input_dim = data_emb_dim + task_emb_dim
        self.output_dim = output_dim
        self.dropout = nn.Dropout(dropout)
        self.batch_norm = batch_norm

        self.hidden_dims = [1024, 512]
        self.layer_size = len(self.hidden_dims) + 1
        dims = [self.input_dim] + self.hidden_dims + [self.output_dim]

        self.predictor = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(self.layer_size)])
        if self.batch_norm:
            self.batch_norms = nn.ModuleList([nn.BatchNorm1d(dims[i + 1]) for i in range(self.layer_size)])
        return

    def forward(self, emb_0, emb_1):
        v = torch.cat([emb_0, emb_1], -1)
        B, T, _ = v.size()
        v = v.reshape(B * T, -1)
        for i, l in enumerate(self.predictor):
            if i == self.layer_size - 1:
                v = l(v)
            else:
                v = l(v)
                if self.batch_norm:
                    v = self.batch_norms[i](v)
                v = F.relu(v)
                v = self.dropout(v)

        v = v.reshape(B, T, self.output_dim)
        return v


class MoleculeTaskTaskPredictionModel(nn.Sequential):
    def __init__(self, data_emb_dim, task_emb_dim, dropout=0.1, output_dim=4, batch_norm=False):
        super(MoleculeTaskTaskPredictionModel, self).__init__()

        self.input_dim = data_emb_dim + task_emb_dim + task_emb_dim
        self.output_dim = output_dim
        self.dropout = nn.Dropout(dropout)
        self.batch_norm = batch_norm

        self.hidden_dims = [1024, 512]
        self.layer_size = len(self.hidden_dims) + 1
        dims = [self.input_dim] + self.hidden_dims + [output_dim]

        self.predictor = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(self.layer_size)])
        if self.batch_norm:
            self.batch_norms = nn.ModuleList([nn.BatchNorm1d(dims[i + 1]) for i in range(self.layer_size)])
        return

    def forward(self, emb_0, emb_1, emb_2):
        '''
        :param emb_0: B, M, d_data
        :param emb_1: B, M, 2*d_task
        :return: B * M * 4
        '''
        v = torch.cat([emb_0, emb_1, emb_2], -1)
        B, M, _ = v.size()
        v = v.reshape(B * M, -1)
        for i, l in enumerate(self.predictor):
            if i == self.layer_size - 1:
                v = l(v)
            else:
                v = l(v)
                if self.batch_norm:
                    v = self.batch_norms[i](v)
                v = F.relu(v)
                v = self.dropout(v)

        v = v.squeeze()
        v = v.reshape(B, M, self.output_dim)
        return v


class MoleculeTaskTaskPredictionModelDense(nn.Sequential):
    def __init__(self, data_emb_dim, task_emb_dim):
        super(MoleculeTaskTaskPredictionModelDense, self).__init__()

        self.input_dim = data_emb_dim + task_emb_dim + task_emb_dim

        self.dropout = nn.Dropout(0.1)

        self.hidden_dims = [1024, 512]
        self.layer_size = len(self.hidden_dims) + 1
        dims = [self.input_dim] + self.hidden_dims + [4]

        self.predictor = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(self.layer_size)])
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(dims[i + 1]) for i in range(self.layer_size)])
        return

    def forward(self, emb_0, emb_1):
        '''
        :param emb_0: B, T, T, d_data
        :param emb_1: B, T, T, 2*d_task
        :return: B * T * T * 4
        '''
        v = torch.cat([emb_0, emb_1], -1)
        B, t, t, _ = v.size()
        v = v.reshape(B * t * t, -1)
        for i, l in enumerate(self.predictor):
            if i == self.layer_size - 1:
                v = l(v)
            else:
                v = l(v)
                v = self.batch_norms[i](v)
                v = F.relu(v)
                v = self.dropout(v)

        v = v.squeeze()
        v = v.reshape(B, t, t, 4)
        return v


class PairwiseTaskPredictionModel(nn.Sequential):
    def __init__(self, task_emb_dim):
        super(PairwiseTaskPredictionModel, self).__init__()
        self.input_dim = task_emb_dim + task_emb_dim

        self.dropout = nn.Dropout(0.1)

        self.hidden_dims = [1024, 512]
        self.layer_size = len(self.hidden_dims) + 1
        dims = [self.input_dim] + self.hidden_dims + [1]

        self.predictor = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(self.layer_size)])
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(dims[i + 1]) for i in range(self.layer_size)])
        return

    def forward(self, emb_0, emb_1):
        v = torch.cat([emb_0, emb_1], -1)
        for i, l in enumerate(self.predictor):
            if i == self.layer_size - 1:
                v = l(v)
            else:
                v = l(v)
                v = self.batch_norms[i](v)
                v = F.relu(v)
                v = self.dropout(v)

        v = v.squeeze()
        return v
import torch
from torch_geometric.data import InMemoryDataset
# from ../models import AliasTable
import torch
from torch import nn


# def transform_sparse_edge_to_dense(edge_list, num_tasks):
#     M = len(edge_list)
#     task_edge_masking = torch.sparse.FloatTensor(
#         edge_list.t(),
#         torch.ones(M),
#         torch.Size([num_tasks, num_tasks])
#     ).to_dense().bool()
#     return task_edge_masking


class AliasTable(nn.Module):
    def __init__(self, probs):
        super(AliasTable, self).__init__()
        self.num_element = len(probs)
        probs, alias = self.build(probs)
        self.register_buffer("probs", probs)
        self.register_buffer("alias", alias)
        self.device = 'cpu'

    def build(self, probs):
        with torch.no_grad():
            probs = probs / probs.mean()
            alias = torch.zeros_like(probs, dtype=torch.long)

            index = torch.arange(len(probs))
            is_available = probs < 1
            small_index = index[is_available]
            large_index = index[~is_available]
            while len(small_index) and len(large_index):
                count = min(len(small_index), len(large_index))
                small, small_index = small_index.split((count, len(small_index) - count))
                large, large_index = large_index.split((count, len(large_index) - count))

                alias[small] = large
                probs[large] += probs[small] - 1

                is_available = probs[large] < 1
                small_index_new = large[is_available]
                large_index_new = large[~is_available]
                small_index = torch.cat([small_index, small_index_new])
                large_index = torch.cat([large_index, large_index_new])

            alias[small_index] = small_index
            alias[large_index] = large_index

        return probs, alias

    def sample(self, sample_shape):
        with torch.no_grad():
            index = torch.randint(self.num_element, sample_shape, device=self.device)
            prob = torch.rand(sample_shape, device=self.device)
            samples = torch.where(prob < self.probs[index], index, self.alias[index])

        return samples



class PPI_dataset(InMemoryDataset):
    def __init__(self, args, string_threshold, neg_sample_size, neg_sample_exponent):
        super(InMemoryDataset, self).__init__()
        f = open('../datasets/{}/filtered_task_score.tsv'.format(args.dataset), 'r')
        edge_list = []
        self.neg_sample_size = neg_sample_size
        self.neg_sample_exponent = neg_sample_exponent

        num_tasks, num_edge = args.num_tasks, 0
        for line in f:
            line = line.strip().split('\t')
            t1, t2, score = int(line[0]), int(line[1]), float(line[2])
            if score < string_threshold:
                continue
            edge_list.append([t1, t2])
            edge_list.append([t2, t1])
            num_edge += 1
        self.edge_list = torch.LongTensor(edge_list)
        print('num task: {}\tnum edge: {}'.format(num_tasks, num_edge))

        degree = [0 for _ in range(num_tasks)]
        for u in self.edge_list[:, 0]:
            degree[u] += 1
        self.degree = torch.FloatTensor(degree)
        self.drug_table = AliasTable(self.degree ** self.neg_sample_exponent)
        return

    def __len__(self):
        return len(self.edge_list)

    def __getitem__(self, idx):
        edge = self.edge_list[idx]
        node0, node1 = edge
        node0_neg = self.drug_table.sample((self.neg_sample_size,))
        node1_neg = self.drug_table.sample((self.neg_sample_size,))

        return node0, node1, node0_neg, node1_neg
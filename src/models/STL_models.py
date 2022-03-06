import torch
from torch import nn
from copy import deepcopy


class SingleTaskModel(torch.nn.Module):
    def __init__(self, model, num_tasks):
        super(SingleTaskModel, self).__init__()
        self.num_tasks = num_tasks
        self.models = nn.ModuleList([deepcopy(model) for _ in range(num_tasks)])

    def forward(self, *args):
        """
            :params kwargs: (batch_size, model_input)
            :output : (batch_size, num_tasks, model_out)
        """
        x = torch.stack([model(*args) for model in self.models], dim=1)
        return x


# class SingleTaskReadout(torch.nn.Module):
#     def __init__(self, emb_dim, num_tasks, init_weight=None, init_bias=None):
#         super(SingleTaskReadout, self).__init__()
#         self.emb_dim = emb_dim
#         self.num_tasks = num_tasks
#         if init_weight is not None and init_bias is not None:
#             weight = torch.cat([deepcopy(init_weight) for _ in range(num_tasks)], dim=0)
#             bias = torch.cat([deepcopy(init_bias) for _ in range(num_tasks)], dim=0)
#         else:
#             weight = torch.randn(num_tasks, emb_dim) / 100
#             bias = torch.randn(num_tasks) / 10
#         self.weight = nn.Parameter(weight)
#         self.bias = nn.Parameter(bias)

#     def forward(self, x):
#         '''
#             :params x: (batch_size, num_tasks, emb_dim)
#             :output : (batch_size, num_tasks)
#         '''
#         x = torch.einsum('ijk, jk -> ij', x, self.weight)
#         x = x + self.bias.expand_as(x)
#         return x


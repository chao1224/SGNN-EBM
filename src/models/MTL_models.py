import torch
from torch import nn


class GradNormModel(nn.Module):
    def __init__(self, num_tasks, args):
        super(GradNormModel, self).__init__()

        self.num_tasks = num_tasks

        self.weights = nn.Parameter(torch.ones(self.num_tasks), requires_grad=True)
        self.register_parameter('grad_norm', self.weights)

    def renormalize(self):
        normalize_coeff = self.num_tasks / self.weights.data.sum()
        self.weights.data *= normalize_coeff
        return


import torch
from torch import nn
import torch.optim as optim
from configure import *


def backward_grad_update_hook(module, grad_input, grad_output):
    '''
    grad_input[0]: T
    grad_input[1]: B * emb_dim
    grad_input[2]: emb_dim * T
    grad_output[0]: B * T
    '''

    x = grad_input[2]

    x.fill_(0)

    grad_input = (grad_input[0], grad_input[1], x)

    global count
    count += 1
    print('calling \t', count)
    print()
    return grad_input


if __name__ == '__main__':
    count = 0
    B = 128
    device = torch.device('cuda:' + str(args.device)) if torch.cuda.is_available() else torch.device('cpu')
    readout_model = torch.nn.Linear(10, 3).to(device)
    optimizer = optim.Adam(readout_model.parameters(), lr=args.lr, weight_decay=args.decay)
    criterion = nn.BCEWithLogitsLoss()

    x = torch.randn((B, 10)).to(device)
    y_label = (torch.randn(B, 3) > 0.5).float().to(device)

    for _ in range(5):
        optimizer.zero_grad()
        y_pred = readout_model(x)
        print(y_pred.size(), '\t', y_label.size())
        loss = criterion(y_pred.view(-1), y_label.view(-1))
        loss.backward()
        print('before\t', readout_model.weight.grad[:5, :5])
        print()

    for _ in range(3):
        handle = readout_model.register_backward_hook(backward_grad_update_hook)
        optimizer.zero_grad()
        y_pred = readout_model(x)
        print(y_pred.size(), '\t', y_label.size())
        loss = criterion(y_pred.view(-1), y_label.view(-1))
        loss.backward()
        print('after\t', readout_model.weight.grad[:5, :5])
        print()
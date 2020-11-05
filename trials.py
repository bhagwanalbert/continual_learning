from models import BigGAN
import numpy as np
import torch
from torch import nn

n_class = 50
# Root of weights
weight_root = './weights'


G = BigGAN.Generator(n_classes = n_class)
D = BigGAN.Discriminator(n_classes = n_class)

print(G)

for name, param in G.named_parameters():
    print(name)

print(G.parameters())

print(D)

for name, param in D.named_parameters():
    print(name)

print(D.parameters())

# G.optim.load_state_dict(
#       torch.load('%s/%s.pth' % (weight_root, 'G_optim')))
#
# for param_group in G.optim.state_dict():
#     print(param_group)
#
# print(G.optim.state_dict()['param_groups'])
#
# D.optim.load_state_dict(
#       torch.load('%s/%s.pth' % (weight_root, 'D_optim')))
#
# for param_group in D.optim.state_dict():
#     print(param_group)
#
# print(D.optim.state_dict()['param_groups'])
#
# for param_group in D.optim.state_dict()['state']:
#     for param in D.optim.state_dict()['state'][param_group]:
#         print(D.optim.state_dict()['state'][param_group][param])

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

print(D)

for name, param in D.named_parameters():
    print(name)

emb_params = {}
lin_params = {}
bn_params = {}
conv_params = {}
for name, param in G.named_parameters():
    param.requires_grad = True
    if ("shared" in name):
        emb_params[name] = param
    elif ("linear" in name):
        lin_params[name] = param
    elif ("bn" in name):
        bn_params[name] = param
    else:
        conv_params[name] = param

print(emb_params)
print(lin_params)
print(bn_params)
print(conv_params)

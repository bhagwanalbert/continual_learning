from models import BigGAN
import numpy as np
import torch
from torch import nn

n_class = 50


G = BigGAN.Generator(n_classes = n_class)
D = BigGAN.Discriminator(n_classes = n_class)

print(G)

for name, param in G.named_parameters():
    # tells whether we want to use gradients for a given parameter
    print(name)

print(D)

for name, param in D.named_parameters():
    # tells whether we want to use gradients for a given parameter
    print(name)

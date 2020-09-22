from models import generator
import numpy as np
import torch
from torch import nn

nz = 110
batch_size = 100

gen = generator.generator(nz)

eval_noise = torch.FloatTensor(batch_size, nz, 1, 1).normal_(0, 1)
eval_noise_ = np.random.normal(0, 1, (batch_size, nz))
eval_label = np.random.randint(0, 10, batch_size)
eval_onehot = np.zeros((batch_size, 10))
eval_onehot[np.arange(batch_size), eval_label] = 1
eval_noise_[np.arange(batch_size), :10] = eval_onehot[np.arange(batch_size)]
eval_noise_ = (torch.from_numpy(eval_noise_))
eval_noise.data.copy_(eval_noise_.view(batch_size, nz, 1, 1))

print(gen(eval_noise).shape)

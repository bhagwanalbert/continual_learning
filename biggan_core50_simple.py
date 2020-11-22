# -*- coding: utf-8 -*-

from __future__ import print_function
#%matplotlib inline
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
from torch.utils.tensorboard import SummaryWriter
from utils import *
from models import BigGAN
import losses

from data_loader import CORE50

# Create tensorboard writer object
writer = SummaryWriter('logs/biggan2')

# Root directory for dataset
dataset = CORE50(root='/home/abhagwan/datasets/core50', scenario="nicv2_391")

# Number of workers for dataloader
workers = 2

# Batch size during training
batch_size = 20

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 128

# Number of channels in the training images. For color images this is 3
nc = 3

# Number of training epochs
num_epochs = 100

# Number of classes of dataset
n_class = 50

# Generator input size
nz = 120

# Learning rate for optimizers
att_lr = 0.0001
conv_lr = 0.0001
lin_lr = 0.0000001
emb_lr = 0.01
bn_lr = 0.0005
out_lr = 0.0001
eps = 1e-8

# Beta1 hyperparam for Adam optimizers
beta1 = 0.0

# Images to view per class to test generator
n_imag = 5

# Additional parameters for BigGAN
num_D_steps = 1
num_D_accumulations = 8
num_G_accumulations = 8


# Root of weights
weight_root = './weights'

# Set cuda device (based on your hardware)
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4"

# Use cuda or not
use_cuda = True

state_dict = {'itr': 0, 'epoch': 0, 'save_num': 0, 'save_best_num': 0,
              'best_IS': 0, 'best_FID': 999999}
## Load pretrained weights with original structure
G = maybe_cuda(BigGAN.Generator(), use_cuda = use_cuda)
G.load_state_dict(
      torch.load('%s/%s.pth' % (weight_root, 'G')), strict=True)

D = maybe_cuda(BigGAN.Discriminator(), use_cuda = use_cuda)
D.load_state_dict(
      torch.load('%s/%s.pth' % (weight_root, 'D')), strict=True)

G_ema = maybe_cuda(BigGAN.Generator(skip_init=True,no_optim=True), use_cuda = use_cuda)
G_ema.load_state_dict(
      torch.load('%s/%s.pth' % (weight_root, 'G_ema')), strict=True)

for item in state_dict:
  state_dict[item] = torch.load('%s/%s.pth' % (weight_root, 'state_dict'))[item]

## Change structures to match dataset number of classes
new_state_dict = G.state_dict()
new_state_dict['shared.weight'] = new_state_dict['shared.weight'][500:n_class+500]

G = BigGAN.Generator(n_classes = n_class).to('cuda:0')
G.load_state_dict(new_state_dict, strict=True)

new_state_dict = D.state_dict()
new_state_dict['embed.weight'] = new_state_dict['embed.weight'][500:n_class+500]
new_state_dict['embed.u0'] = new_state_dict['embed.u0'][:,500:n_class+500]

D = BigGAN.Discriminator(n_classes = n_class).to('cuda:0')
D.load_state_dict(new_state_dict, strict=True)

new_state_dict = G_ema.state_dict()
new_state_dict['shared.weight'] = new_state_dict['shared.weight'][500:n_class+500]

G_ema = BigGAN.Generator(n_classes = n_class, skip_init=True, no_optim=True).to('cuda:0')
G_ema.load_state_dict(new_state_dict, strict=True)
ema = ema(G, G_ema,start_itr = 20000)

## Load optimizer state dict and adapt it
# G.optim.load_state_dict(
#       torch.load('%s/%s.pth' % (weight_root, 'G_optim')))
#
# for param_group in G.optim.state_dict()['state']:
#     for param in G.optim.state_dict()['state'][param_group]:
#         if param == 'step':
#             pass
#         else:
#             if G.optim.state_dict()['state'][param_group][param].shape == torch.Size([]):
#                 pass
#             elif G.optim.state_dict()['state'][param_group][param].shape[0] == 1000:
#                 G.optim.state_dict()['state'][param_group][param] = \
#                     G.optim.state_dict()['state'][param_group][param][500:n_class+500]
#
# D.optim.load_state_dict(
#       torch.load('%s/%s.pth' % (weight_root, 'D_optim')))
#
# for param_group in D.optim.state_dict()['state']:
#     for param in D.optim.state_dict()['state'][param_group]:
#         if param == 'step':
#             pass
#         else:
#             if D.optim.state_dict()['state'][param_group][param].shape == torch.Size([]):
#                 pass
#             elif D.optim.state_dict()['state'][param_group][param].shape[0] == 1000:
#                 D.optim.state_dict()['state'][param_group][param] = \
#                     D.optim.state_dict()['state'][param_group][param][500:n_class+500]

## Use a fresh optimizer
emb_params = {}
lin_params = {}
bn_params = {}
conv_params = {}
out_params = {}
att_params = {}
for name, param in G.named_parameters():
    param.requires_grad = True
    if ("shared" in name):
        emb_params[name] = param
    elif ("linear" in name):
        lin_params[name] = param
    elif ("bn" in name):
        bn_params[name] = param
    elif ("conv" in name):
        conv_params[name] = param
    elif ("output" in name):
        out_params[name] = param
    else:
        att_params[name] = param


params = []
params.append({"params":list(emb_params.values()), "lr":emb_lr})
params.append({"params":list(lin_params.values()), "lr":lin_lr})
params.append({"params":list(bn_params.values()), "lr":bn_lr})
params.append({"params":list(conv_params.values()), "lr":conv_lr})
params.append({"params":list(out_params.values()), "lr":out_lr})
params.append({"params":list(att_params.values()), "lr":att_lr})

G.optim = torch.optim.Adam(params, lr=0, betas=(beta1, 0.999), eps=eps)

emb_params = {}
lin_params = {}
att_params = {}
conv_params = {}
for name, param in D.named_parameters():
    param.requires_grad = True
    if ("embed" in name):
        emb_params[name] = param
    elif ("linear" in name):
        lin_params[name] = param
    elif ("conv" in name):
        conv_params[name] = param
    else:
        att_params[name] = param

params = []
params.append({"params":list(emb_params.values()), "lr":emb_lr})
params.append({"params":list(lin_params.values()), "lr":lin_lr})
params.append({"params":list(att_params.values()), "lr":att_lr})
params.append({"params":list(conv_params.values()), "lr":conv_lr})

D.optim = torch.optim.Adam(params, lr=0, betas=(beta1, 0.999), eps=eps)

print(G.optim)
print(D.optim)

GD = BigGAN.G_D(G, D)
GD = nn.DataParallel(GD, device_ids=[0, 1, 2, 3, 4])

## Test current BigGAN
eval_z = torch.FloatTensor(n_imag*n_class, nz).normal_(0, 1)
eval_z_ = np.random.normal(0, 1, (n_imag*n_class, nz))
eval_z_ = (torch.from_numpy(eval_z_))
eval_z.data.copy_(eval_z_.view(n_imag*n_class, nz))
eval_z = eval_z.to('cuda:0')

eval_y = np.zeros((n_imag*n_class))
for c in range(n_class):
    eval_y[np.arange(n_imag*c,n_imag*(c+1))] = c
eval_y = (torch.from_numpy(eval_y))
eval_y = eval_y.to('cpu', torch.int64)
eval_y = eval_y.to('cuda:0')

# with torch.no_grad():
#     fake = nn.parallel.data_parallel(G, (eval_z, G.shared(eval_y)), device_ids=[2, 3, 0, 1])
# writer.add_image("Generated images", vutils.make_grid(fake, nrow=n_imag, padding=2, normalize=True))
# writer.close()

# vutils.save_image(fake.float(),
#                              'random_samples.jpg',
#                              nrow=n_imag,
#                              normalize=True)

## out = D(fake, y)

## Load dataset
test_x, test_y = dataset.get_test_set()
test_x = preprocess_imgs(test_x, norm=False, symmetric = True)

train_x, train_y = next(iter(dataset))
train_x = preprocess_imgs(train_x, norm=False, symmetric = True)

train_x = torch.from_numpy(train_x).type(torch.FloatTensor)
train_y = torch.from_numpy(train_y).type(torch.LongTensor)

indexes = np.random.permutation(train_y.size(0))

# Shuffle train dataset
train_x = train_x[indexes]
train_y = train_y[indexes]

training_examples = None

for c in range(n_class):
    if training_examples == None:
        training_examples = train_x[train_y.numpy() == c][:5]
    else:
        training_examples = torch.cat((training_examples, train_x[train_y.numpy() == c][:5]))

writer.add_image("Training images", vutils.make_grid(training_examples, nrow=n_imag, padding=2, normalize=True).cpu())
writer.close()

# Training Loop
print("Starting Training Loop...")

tot_it_step = 0
it_x_ep = train_x.size(0) // batch_size

train_x = train_x.to('cuda:0')
train_y = train_y.to('cuda:0')

x_mb = torch.split(train_x, batch_size)
y_mb = torch.split(train_y, batch_size)

print(x_mb[0].shape)
print(y_mb[0].shape)
D_fake, D_real = GD(eval_z, eval_y,
                    x_mb[0], y_mb[0], train_G=False,
                    split_D=False)

tot_it_step = 0

x_mb = torch.split(train_x, batch_size)
y_mb = torch.split(train_y, batch_size)

print(len(x_mb))

num_iter = len(x_mb)//(num_D_steps*num_D_accumulations)

for ep in range(num_epochs):
    print("training ep: ", ep)

    counter = 0

    G.train()
    D.train()
    G_ema.train()

    for it in range(num_iter):

        G.optim.zero_grad()
        D.optim.zero_grad()

        toggle_grad(D, True)
        toggle_grad(G, False)

        for step_index in range(num_D_steps):
          # If accumulating gradients, loop multiple times before an optimizer step
          D.optim.zero_grad()
          for accumulation_index in range(num_D_accumulations):
            z = torch.FloatTensor(y_mb[counter].shape[0], nz).normal_(0, 1)
            z_ = np.random.normal(0, 1, (y_mb[counter].shape[0], nz))
            z_ = (torch.from_numpy(z_))
            z.data.copy_(z_.view(y_mb[counter].shape[0], nz))
            z = z.to('cuda:0')

            y = np.random.randint(0, n_class, y_mb[counter].shape[0])
            y = (torch.from_numpy(y))
            y = y.to('cpu', torch.int64)
            y = y.to('cuda:0')

            print(z.shape)
            print(y.shape)
            print(x_mb[counter].shape)
            print(y_mb[counter].shape)
            print(it)
            print(accumulation_index)
            print(counter)

            D_fake, D_real = GD(z, y,
                                x_mb[counter], y_mb[counter], train_G=False,
                                split_D=False)

            # Compute components of D's loss, average them, and divide by
            # the number of gradient accumulations
            D_loss_real, D_loss_fake = losses.discriminator_loss(D_fake, D_real)
            D_loss = (D_loss_real + D_loss_fake) / float(num_D_accumulations)
            D_loss.backward()
            counter += 1

          D.optim.step()

        toggle_grad(D, False)
        toggle_grad(G, True)

        # Zero G's gradients by default before training G, for safety
        G.optim.zero_grad()

        # If accumulating gradients, loop multiple times
        for accumulation_index in range(num_G_accumulations):
          z = torch.FloatTensor(batch_size, nz).normal_(0, 1)
          z_ = np.random.normal(0, 1, (batch_size, nz))
          z_ = (torch.from_numpy(z_))
          z.data.copy_(z_.view(batch_size, nz))
          z = z.to('cuda:0')

          y = np.random.randint(0, n_class, batch_size)
          y = (torch.from_numpy(y))
          y = y.to('cpu', torch.int64)
          y = y.to('cuda:0')

          D_fake = GD(z, y, train_G=True, split_D=False)
          G_loss = losses.generator_loss(D_fake) / float(num_G_accumulations)
          G_loss.backward()

        G.optim.step()

        ema.update(state_dict['itr'])

        tot_it_step += 1

        writer.add_scalar('G_loss', float(G_loss.item()), tot_it_step)
        writer.add_scalar('D_loss_real', float(D_loss_real.item()), tot_it_step)
        writer.add_scalar('D_loss_fake', float(D_loss_fake.item()), tot_it_step)
        writer.close()

    with torch.no_grad():
        fake = nn.parallel.data_parallel(G, (eval_z, G.shared(eval_y)), device_ids=[0, 1, 2, 3, 4])
    writer.add_image("Generated images", vutils.make_grid(fake, nrow=n_imag, padding=2, normalize=True))
    writer.close()

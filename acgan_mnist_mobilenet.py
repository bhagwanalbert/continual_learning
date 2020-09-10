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
from models.mobilenet import MyMobilenetV1
from models.generator import generator
from utils import *

# Create tensorboard writer object
writer = SummaryWriter('logs/mnist3')

# Root directory for dataset
dataroot = "/home/abhagwan/datasets/MNIST"
#dataroot = "/home/deepak/datasets/MNIST"

# Number of workers for dataloader
workers = 2

# Batch size during training
batch_size = 128

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 128

# Number of channels in the training images. For color images this is 3
nc = 3

# Number of training epochs
num_epochs = 100

# Number of classes of dataset
n_class = 10

# Generator input size
nz = 100 + n_class

# Learning rate for optimizers
lr_disc = 0.0004
lr_gen = 0.0001

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# Images to view per class to test generator
n_imag = 5

# Freezed layers details
#freeze_below_layer = "lat_features.19.bn.beta"
freeze_below_layer = "lat_features.19.bn.bias"
latent_layer_num = 19

# Set cuda device (based on your hardware)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Use cuda or not
#use_cuda = False
use_cuda = True


train_dataset = dset.MNIST(root = dataroot, download = True, train = True,
                            transform = transforms.Compose([
                                        transforms.Resize(image_size),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                        (0.0,), (1.0,))]))

test_dataset = dset.MNIST(root = dataroot, download = True, train = False,
                            transform = transforms.Compose([
                                        transforms.Resize(image_size),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                        (0.0,), (1.0,))]))

# Create the dataloader
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)

test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)

# Plot some training images
real_batch = next(iter(train_dataloader))

writer.add_image("Training images", vutils.make_grid(real_batch[0][:64], padding=2, normalize=True).cpu())
writer.close()

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# Discriminator + classifier
model = MyMobilenetV1(pretrained=True, latent_layer_num=latent_layer_num, num_classes=n_class, softmax=True, discriminator=True)
gen = generator(nz)

gen.apply(weights_init)

# init_update_rate = 0.01
# inc_update_rate = 0.00005
# max_r_max = 1.25
# max_d_max = 0.5
# inc_step = 4.1e-05
# momentum = 0.9
# l2 = 0.0005
# replace_bn_with_brn(
#     model, momentum=init_update_rate, r_d_max_inc_step=inc_step,
#     max_r_max=max_r_max, max_d_max=max_d_max
# )

# Optimizer setup
optimizer = torch.optim.Adam(model.parameters(), lr=lr_disc, betas=(beta1, 0.999))
optimG = torch.optim.Adam(gen.parameters(), lr=lr_gen, betas=(beta1, 0.999))

criterion = torch.nn.NLLLoss()
criterion_source = torch.nn.BCELoss()

# Fix noise to view generated images
eval_noise = torch.FloatTensor(n_imag*n_class, nz, 1, 1).normal_(0, 1)
eval_noise_ = np.random.normal(0, 1, (n_imag*n_class, nz))
eval_onehot = np.zeros((n_imag*n_class, n_class))

for c in range(n_class):
    eval_onehot[np.arange(n_imag*c,n_imag*(c+1)), c] = 1

eval_noise_[np.arange(n_imag*n_class), :n_class] = eval_onehot[np.arange(n_imag*n_class)]

eval_noise_ = (torch.from_numpy(eval_noise_))
eval_noise.data.copy_(eval_noise_.view(n_imag*n_class, nz, 1, 1))
eval_noise = maybe_cuda(eval_noise, use_cuda=use_cuda)
print(eval_noise.size())

# Training Loop
print("Starting Training Loop...")

tot_it_step = 0

for ep in range(num_epochs):
    print("training ep: ", ep)
    for i, data in enumerate(train_dataloader):

        freeze_up_to(model, freeze_below_layer, only_conv=False)

        train_x, train_y = data

        model.train()
        model.lat_features.eval()

        model = maybe_cuda(model, use_cuda=use_cuda)
        gen = maybe_cuda(gen, use_cuda=use_cuda)
        acc = None
        ave_loss = 0

        train_x = maybe_cuda(torch.cat((train_x,train_x,train_x), 1).type(torch.FloatTensor), use_cuda=use_cuda)
        train_y = maybe_cuda(train_y.type(torch.LongTensor), use_cuda=use_cuda)

        correct_cnt, ave_loss, correct_src, correct_src_fake, ave_loss_gen = 0, 0, 0, 0, 0
        data_encountered = 0

        optimizer.zero_grad()

        classes, source = model(train_x, latent_input=None, return_lat_acts=False)

        # Labels indicating source of the image
        real_label = maybe_cuda(torch.FloatTensor(train_y.size(0)), use_cuda=use_cuda)
        real_label.fill_(0.9)

        fake_label = maybe_cuda(torch.FloatTensor(train_y.size(0)), use_cuda=use_cuda)
        fake_label.fill_(0.1)

        _, pred_label = torch.max(classes, 1)
        correct_cnt += (pred_label == train_y).sum()

        pred_source = torch.round(source)
        correct_src += (pred_source == 1).sum()

        loss = criterion(classes, train_y) + criterion_source(source, real_label)

        loss.backward()
        optimizer.step()

        ave_loss += loss.item()
        data_encountered += train_y.size(0)

        acc = correct_cnt.item() / data_encountered
        ave_loss /= data_encountered
        source_acc = correct_src.item() / data_encountered

        ## Training with fake data now
        noise = torch.FloatTensor(train_y.size(0), nz, 1, 1).normal_(0, 1)
        noise_ = np.random.normal(0, 1, (train_y.size(0), nz))
        label = np.random.randint(0, n_class, train_y.size(0))
        onehot = np.zeros((train_y.size(0), n_class))
        onehot[np.arange(train_y.size(0)), label] = 1
        noise_[np.arange(train_y.size(0)), :n_class] = onehot[np.arange(train_y.size(0))]
        noise_ = (torch.from_numpy(noise_))
        noise.data.copy_(noise_.view(train_y.size(0), nz, 1, 1))
        noise = maybe_cuda(noise, use_cuda=use_cuda)

        label = ((torch.from_numpy(label)).long())
        label = maybe_cuda(label, use_cuda=use_cuda)

        noise_image = gen(noise)

        classes, source = model(
               noise_image.detach(), latent_input=None, return_lat_acts=False)

        pred_source = torch.round(source)
        correct_src_fake += (pred_source == 0).sum()

        loss_fake = criterion_source(source, fake_label) + criterion(classes, label)

        loss_fake.backward()
        optimizer.step()

        source_acc_fake = correct_src_fake.item() / data_encountered

        ## Train the generator
        optimG.zero_grad()

        classes, source = model(noise_image)

        source_loss = criterion_source(source, real_label) #The generator tries to pass its images as real---so we pass the images as real to the cost function
        class_loss = criterion(classes, label)

        loss_gen = source_loss + class_loss

        loss_gen.backward()
        optimG.step()

        ave_loss_gen += loss_gen.item()
        ave_loss_gen /= data_encountered

        # Output training stats
        if i % 50 == 0:
            print(
                '==>>> it: {}, avg. loss: {:.6f}, '
                'running train acc: {:.3f}, '
                'running source acc: {:.3f}, '
                'running source acc fake: {:.3f}, '
                'running avg. loss gen: {:.3f}'
                    .format(i, ave_loss, acc, source_acc, source_acc_fake, ave_loss_gen)
            )
            with torch.no_grad():
                fake = gen(eval_noise).detach().cpu()
            writer.add_image("Generated images", vutils.make_grid(fake, padding=2, normalize=True))

        tot_it_step +=1

        writer.add_scalar('train_loss', ave_loss, tot_it_step)
        writer.add_scalar('train_accuracy', acc, tot_it_step)
        writer.add_scalar('source_accuracy', source_acc, tot_it_step)
        writer.add_scalar('source_fake_accuracy', source_acc_fake, tot_it_step)
        writer.add_scalar('gen_loss', ave_loss_gen, tot_it_step)

        writer.close()

        # Check how the generator is doing by saving G's output on fixed_noise
        # if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(train_dataloader)-1)):
        #     with torch.no_grad():
        #         fake = netG(fixed_noise).detach().cpu()
        #     img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
        #     writer.add_image("Generated images", vutils.make_grid(fake, padding=2, normalize=True))
        #     writer.close()
        #
        # iters += 1

    model.eval()
    correct_cnt, ave_loss, correct_src = 0, 0, 0
    data_encountered = 0

    model = maybe_cuda(model, use_cuda=use_cuda)

    for i, data in  enumerate(test_dataloader):

        test_x, test_y = data

        test_x = maybe_cuda(torch.cat((test_x,test_x,test_x), 1), use_cuda=use_cuda)
        test_y = maybe_cuda(test_y, use_cuda=use_cuda)

        classes, source = model(test_x)

        loss = criterion(classes, test_y)
        _, pred_label = torch.max(classes.data, 1)
        correct_cnt += (pred_label == test_y.data).sum()
        ave_loss += loss.item()

        pred_source = torch.round(source)
        correct_src += (pred_source == 1).sum()

        data_encountered += test_y.size(0)

    acc = correct_cnt.item() * 1.0 / data_encountered
    source_acc = correct_src.item() * 1.0 / data_encountered
    ave_loss /= data_encountered

    # Log scalar values (scalar summary) to TB
    writer.add_scalar('test_loss', ave_loss, ep)
    writer.add_scalar('test_accuracy', acc, ep)
    writer.add_scalar('test_source', source_acc, ep)

    writer.close()

    print("---------------------------------")
    print("Accuracy: ", acc)
    print("---------------------------------")

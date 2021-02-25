#!/usr/bin/env python
# -*- coding: utf-8 -*-

################################################################################
# Copyright (c) 2020. Vincenzo Lomonaco, Gabriele Graffieti, Lorenzo           #
# Pellegrini, Davide Maltoni. All rights reserved.                             #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 01-04-2020                                                             #
# Authors: Vincenzo Lomonaco, Gabriele Graffieti, Lorenzo Pellegrini, Davide   #
# Maltoni.                                                                     #
# E-mail: vincenzo.lomonaco@unibo.it                                           #
# Website: vincenzolomonaco.com                                                #
################################################################################

""" Simple AR1* implementation in PyTorch with Latent Replay """

# Python 2-3 compatible
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

from data_loader import CORE50
import copy
import os
import json
from models.mobilenet import MyMobilenetV1
from utils import *
import configparser
import argparse
from pprint import pprint
from torch.utils.tensorboard import SummaryWriter
from models.generator import generator_feat
from models.discriminator import conditioned_discriminator_feat
import matplotlib.pyplot as plt
import io
import PIL.Image
import random
import numpy as np
from torchvision import transforms

def histogram(x):
    figure = plt.figure(figsize=(10,7))
    x_flat = x.view(x.shape[0]*x.shape[1]*x.shape[2]*x.shape[3])
    print("max feat: ", torch.max(x_flat).item())
    print("min feat: ", torch.min(x_flat).item())
    print("sparse measurement: ", torch.sum(x_flat==-1)/float(x_flat.shape[0]))

    plt.hist(np.array(x_flat.cpu().detach().numpy()), bins=100)

    plt.show()
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    image = PIL.Image.open(buf)
    image = transforms.ToTensor()(image)

    return image

# --------------------------------- Setup --------------------------------------

# recover exp configuration name
parser = argparse.ArgumentParser(description='Run CL experiments')
parser.add_argument('--name', dest='exp_name',  default='DEFAULT',
                    help='name of the experiment you want to run.')
args = parser.parse_args()

# set cuda device (based on your hardware)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "9"

# recover config file for the experiment
config = configparser.ConfigParser()
config.read("params.cfg")
exp_config = config[args.exp_name]
print("Experiment name:", args.exp_name)
pprint(dict(exp_config))

# recover parameters from the cfg file and compute the dependent ones.
exp_name = eval(exp_config['exp_name'])
comment = eval(exp_config['comment'])
use_cuda = eval(exp_config['use_cuda'])
init_lr = eval(exp_config['init_lr'])
inc_lr = eval(exp_config['inc_lr'])
mb_size = eval(exp_config['mb_size'])
init_train_ep = eval(exp_config['init_train_ep'])
inc_train_ep = eval(exp_config['inc_train_ep'])
init_update_rate = eval(exp_config['init_update_rate'])
inc_update_rate = eval(exp_config['inc_update_rate'])
max_r_max = eval(exp_config['max_r_max'])
max_d_max = eval(exp_config['max_d_max'])
inc_step = eval(exp_config['inc_step'])
rm_sz = eval(exp_config['rm_sz'])
momentum = eval(exp_config['momentum'])
l2 = eval(exp_config['l2'])
freeze_below_layer = eval(exp_config['freeze_below_layer'])
latent_layer_num = eval(exp_config['latent_layer_num'])
reg_lambda = eval(exp_config['reg_lambda'])

# setting up log dir for tensorboard
log_dir = 'logs/' + exp_name
writer = SummaryWriter(log_dir)

# Saving params
hyper = json.dumps(dict(exp_config))
writer.add_text("parameters", hyper, 0)

# Other variables init
tot_it_step = 0
rm = None

# Create the dataset object
dataset = CORE50(root='/home/abhagwan/datasets/core50', scenario="nicv2_391")
n_class = 50
preproc = preprocess_imgs

# Get the fixed test set
test_x, test_y = dataset.get_test_set()

# Model setup
model = MyMobilenetV1(pretrained=True, latent_layer_num=latent_layer_num)
# we replace BN layers with Batch Renormalization layers
replace_bn_with_brn(
    model, momentum=init_update_rate, r_d_max_inc_step=inc_step,
    max_r_max=max_r_max, max_d_max=max_d_max
)
model.saved_weights = {}
model.past_j = {i:0 for i in range(n_class)}
model.cur_j = {i:0 for i in range(n_class)}
if reg_lambda != 0:
    # the regularization is based on Synaptic Intelligence as described in the
    # paper. ewcData is a list of two elements (best parametes, importance)
    # while synData is a dictionary with all the trajectory data needed by SI
    ewcData, synData = create_syn_data(model)

# Optimizer setup
optimizer = torch.optim.SGD(
    model.parameters(), lr=init_lr, momentum=momentum, weight_decay=l2
)
criterion = torch.nn.CrossEntropyLoss()

# GAN stuff
gan_lr = 0.00005
nz = 100 + n_class
gan_train_ep = 100
gan_tot_it = 0
n_imag = 10

normalize = transforms.Normalize(mean=[2.5, 2.5, 2.5], std=[2.5, 2.5, 2.5])
unnormalize = transforms.Normalize(mean=[-1.0, -1.0, -1.0], std=[0.4, 0.4, 0.4])

disc = conditioned_discriminator_feat(num_classes=n_class)
gen = generator_feat(nz)

disc = maybe_cuda(disc, use_cuda=use_cuda)
gen = maybe_cuda(gen, use_cuda=use_cuda)

disc.apply(weights_init)
gen.apply(weights_init)

optimD = torch.optim.Adam(disc.parameters(), lr=gan_lr, betas=(0.5, 0.999))
optimG = torch.optim.Adam(gen.parameters(), lr=gan_lr, betas=(0.5, 0.999))

criterion_class = torch.nn.NLLLoss()
criterion_source = torch.nn.BCELoss()

fixed_noise = {}

for c in range(n_class):
    eval_noise = torch.FloatTensor(n_imag, nz, 1, 1).normal_(0, 1)
    eval_noise_ = np.random.normal(0, 1, (n_imag, nz))
    eval_onehot = np.zeros((n_imag, n_class))
    eval_onehot[:, c] = 1

    eval_noise_[np.arange(n_imag), :n_class] = eval_onehot[np.arange(n_imag)]

    eval_noise_ = (torch.from_numpy(eval_noise_))
    eval_noise.data.copy_(eval_noise_.view(n_imag, nz, 1, 1))
    fixed_noise[str(c)] = maybe_cuda(eval_noise, use_cuda=use_cuda)

# --------------------------------- Training -----------------------------------

# loop over the training incremental batches
for i, train_batch in enumerate(dataset):

    # ABB: for the moment only train one batch
    if i > 0:
        break

    if reg_lambda != 0:
        init_batch(model, ewcData, synData)

    # we freeze the layer below the replay layer since the first batch
    freeze_up_to(model, freeze_below_layer, only_conv=False)

    if i == 1:
        change_brn_pars(
            model, momentum=inc_update_rate, r_d_max_inc_step=0,
            r_max=max_r_max, d_max=max_d_max)
        optimizer = torch.optim.SGD(
            model.parameters(), lr=inc_lr, momentum=momentum, weight_decay=l2
        )

    train_x, train_y = train_batch
    train_x = preproc(train_x)

    if i == 0:
        cur_class = [int(o) for o in set(train_y)]
        model.cur_j = examples_per_class(train_y)
    else:
        cur_class = [int(o) for o in set(train_y).union(set(rm[1]))]
        model.cur_j = examples_per_class(list(train_y) + list(rm[1]))

    print("----------- batch {0} -------------".format(i))
    print("train_x shape: {}, train_y shape: {}"
          .format(train_x.shape, train_y.shape))

    model.train()
    model.lat_features.eval()

    reset_weights(model, cur_class)
    cur_ep = 0

    if i == 0:
        (train_x, train_y), it_x_ep = pad_data([train_x, train_y], mb_size)
    shuffle_in_unison([train_x, train_y], in_place=True)

    model = maybe_cuda(model, use_cuda=use_cuda)
    acc = None
    ave_loss = 0

    train_x = torch.from_numpy(train_x).type(torch.FloatTensor)
    train_y = torch.from_numpy(train_y).type(torch.LongTensor)

    if i == 0:
        train_ep = init_train_ep
    else:
        train_ep = inc_train_ep

    for ep in range(train_ep):

        print("training ep: ", ep)
        correct_cnt, ave_loss = 0, 0

        # computing how many patterns to inject in the latent replay layer
        if i > 0:
            cur_sz = train_x.size(0) // ((train_x.size(0) + rm_sz) // mb_size)
            it_x_ep = train_x.size(0) // cur_sz
            n2inject = max(0, mb_size - cur_sz)
        else:
            n2inject = 0
        print("total sz:", train_x.size(0) + rm_sz)
        print("n2inject", n2inject)
        print("it x ep: ", it_x_ep)
        if rm != None:
            print("rm sz: ", rm[0].size())

        for it in range(it_x_ep):

            if reg_lambda !=0:
                pre_update(model, synData)

            start = it * (mb_size - n2inject)
            end = (it + 1) * (mb_size - n2inject)

            optimizer.zero_grad()

            x_mb = maybe_cuda(train_x[start:end], use_cuda=use_cuda)

            if i == 0:
                lat_mb_x = None
                y_mb = maybe_cuda(train_y[start:end], use_cuda=use_cuda)

            else:
                lat_mb_x = rm[0][it*n2inject: (it + 1)*n2inject]
                lat_mb_y = rm[1][it*n2inject: (it + 1)*n2inject]
                y_mb = maybe_cuda(
                    torch.cat((train_y[start:end], lat_mb_y), 0),
                    use_cuda=use_cuda)
                lat_mb_x = maybe_cuda(lat_mb_x, use_cuda=use_cuda)

            # if lat_mb_x is not None, this tensor will be concatenated in
            # the forward pass on-the-fly in the latent replay layer
            logits, lat_acts = model(
                x_mb, latent_input=lat_mb_x, return_lat_acts=True)

            # collect latent volumes only for the first ep
            # we need to store them to eventually add them into the external
            # replay memory
            if ep == 0:
                lat_acts = lat_acts.cpu().detach()
                if it == 0:
                    cur_acts = copy.deepcopy(lat_acts)
                else:
                    cur_acts = torch.cat((cur_acts, lat_acts), 0)

            _, pred_label = torch.max(logits, 1)
            correct_cnt += (pred_label == y_mb).sum()

            loss = criterion(logits, y_mb)
            if reg_lambda !=0:
                loss += compute_ewc_loss(model, ewcData, lambd=reg_lambda)
            ave_loss += loss.item()

            loss.backward()
            optimizer.step()

            if reg_lambda !=0:
                post_update(model, synData)

            acc = correct_cnt.item() / \
                  ((it + 1) * y_mb.size(0))
            ave_loss /= ((it + 1) * y_mb.size(0))

            if it % 10 == 0:
                print(
                    '==>>> it: {}, avg. loss: {:.6f}, '
                    'running train acc: {:.3f}'
                        .format(it, ave_loss, acc)
                )

            # Log scalar values (scalar summary) to TB
            tot_it_step +=1
            writer.add_scalar('train_loss', ave_loss, tot_it_step)
            writer.add_scalar('train_accuracy', acc, tot_it_step)

        cur_ep += 1

    consolidate_weights(model, cur_class)
    if reg_lambda != 0:
        update_ewc_data(model, ewcData, synData, 0.001, 1)

    # how many patterns to save for next iter
    h = min(rm_sz // (i + 1), cur_acts.size(0))
    print("h", h)

    print("cur_acts sz:", cur_acts.size(0))
    print("cur_acts sz:", cur_acts.shape)
    writer.add_image("Histogram",histogram(cur_acts),0)
    writer.close()

    idxs_cur = np.random.choice(
        cur_acts.size(0), h, replace=False
    )
    rm_add = [cur_acts[idxs_cur], train_y[idxs_cur]]
    print("rm_add size", rm_add[0].size(0))

    # replace patterns in random memory
    if i == 0:
        rm = copy.deepcopy(rm_add)
    else:
        idxs_2_replace = np.random.choice(
            rm[0].size(0), h, replace=False
        )
        for j, idx in enumerate(idxs_2_replace):
            rm[0][idx] = copy.deepcopy(rm_add[0][j])
            rm[1][idx] = copy.deepcopy(rm_add[1][j])

    set_consolidate_weights(model)
    ave_loss, acc, accs = get_accuracy(
        model, criterion, mb_size, test_x, test_y, preproc=preproc
    )

    for ep in range(gan_train_ep):
        print("GAN training ep: ", ep)

        correct_real_cnt = 0
        correct_fake_cnt = 0
        correct_test_cnt = 0
        correct_src_real_cnt = 0
        correct_src_fake_cnt = 0

        for it in range(it_x_ep):

            start = it * (mb_size - n2inject)
            end = (it + 1) * (mb_size - n2inject)

            x_mb = maybe_cuda(train_x[start:end], use_cuda=use_cuda)

            # if i == 0:
            lat_mb_x = None
            y_mb = maybe_cuda(train_y[start:end], use_cuda=use_cuda)

            # if lat_mb_x is not None, this tensor will be concatenated in
            # the forward pass on-the-fly in the latent replay layer
            with torch.no_grad():
                _, real_feat = model(
                    x_mb, latent_input=lat_mb_x, return_lat_acts=True)

            optimD.zero_grad()
            disc.train()
            gen.eval()

            real_feat = maybe_cuda(normalize(real_feat), use_cuda=use_cuda)

            writer.add_image("Histogram real features",histogram(real_feat), gan_tot_it)

            classes, source = disc(real_feat)
            _, pred_label = torch.max(classes, 1)
            correct_real_cnt += (pred_label == y_mb).sum()
            pred_source = torch.round(source)
            correct_src_real_cnt += (pred_source == 1).sum()

            # Labels indicating source of the image
            real_label = maybe_cuda(torch.FloatTensor(y_mb.size(0)), use_cuda=use_cuda)
            real_label.fill_(0.9)

            fake_label = maybe_cuda(torch.FloatTensor(y_mb.size(0)), use_cuda=use_cuda)
            fake_label.fill_(0.1)

            lossDreal = criterion(classes, y_mb) + criterion_source(source, real_label)

            lossDreal.backward()
            optimD.step()

            noise = torch.FloatTensor(y_mb.size(0), nz, 1, 1).normal_(0, 1)
            noise_ = np.random.normal(0, 1, (y_mb.size(0), nz))
            label = np.random.choice(cur_class, y_mb.size(0))
            onehot = np.zeros((y_mb.size(0), n_class))
            onehot[np.arange(y_mb.size(0)), label] = 1
            noise_[np.arange(y_mb.size(0)), :n_class] = onehot[np.arange(y_mb.size(0))]
            noise_ = (torch.from_numpy(noise_))
            noise.data.copy_(noise_.view(y_mb.size(0), nz, 1, 1))
            noise = maybe_cuda(noise, use_cuda=use_cuda)

            label = ((torch.from_numpy(label)).long())
            label = maybe_cuda(label, use_cuda=use_cuda)

            fake_feat = gen(noise)

            writer.add_image("Histogram fake features",histogram(fake_feat), gan_tot_it)
            writer.close()

            classes, source = disc(fake_feat.detach())
            _, pred_label = torch.max(classes, 1)
            correct_fake_cnt += (pred_label == label).sum()
            pred_source = torch.round(source)
            correct_src_fake_cnt += (pred_source == 0).sum()

            lossDfake = criterion_source(source, fake_label) + criterion(classes, label)

            lossDfake.backward()
            optimD.step()

            disc.eval()
            gen.train()

            optimG.zero_grad()

            classes, source = disc(fake_feat.detach())

            lossG = criterion_source(source, real_label) + criterion(classes, label)

            lossG.backward()
            optimG.step()

            writer.add_scalar('D real training loss', lossDreal, gan_tot_it)
            writer.add_scalar('D fake training loss', lossDfake, gan_tot_it)
            writer.add_scalar('G training loss', lossG, gan_tot_it)

            acc_real = correct_real_cnt.item() / \
                        ((it + 1) * y_mb.size(0))
            acc_fake = correct_fake_cnt.item() / \
                        ((it + 1) * y_mb.size(0))
            acc_src_real = correct_src_real_cnt.item() / \
                        ((it + 1) * y_mb.size(0))
            acc_src_fake = correct_src_fake_cnt.item() / \
                        ((it + 1) * y_mb.size(0))

            gan_tot_it += 1

        with torch.no_grad():
            for c in cur_class:
                test_feat = unnormalize(gen(fixed_noise[str(c)]))
                writer.add_image("Histogram test features",histogram(test_feat), ep)
                classes = model(None, latent_input=test_feat)
                _, pred_label = torch.max(classes, 1)
                correct_test_cnt += (pred_label == c).sum()
                print(pred_label)
                print(c)

            acc_test = correct_test_cnt.item() / (n_imag*len(cur_class))

        writer.add_scalar('GAN real training acc', acc_real, ep)
        writer.add_scalar('GAN fake training acc', acc_fake, ep)
        writer.add_scalar('GAN test acc', acc_test, ep)
        writer.add_scalar('GAN real src acc', acc_src_real, ep)
        writer.add_scalar('GAN fake src acc', acc_src_fake, ep)


    # Log scalar values (scalar summary) to TB
    writer.add_scalar('test_loss', ave_loss, i)
    writer.add_scalar('test_accuracy', acc, i)

    # update number examples encountered over time
    for c, n in model.cur_j.items():
        model.past_j[c] += n

    print("---------------------------------")
    print("Accuracy: ", acc)
    print("---------------------------------")

writer.close()

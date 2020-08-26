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
from models.generator import generator
from utils import *
import configparser
import argparse
from pprint import pprint
from torch.utils.tensorboard import SummaryWriter

# --------------------------------- Setup --------------------------------------

# recover exp configuration name
parser = argparse.ArgumentParser(description='Run CL experiments')
parser.add_argument('--name', dest='exp_name',  default='DEFAULT',
                    help='name of the experiment you want to run.')
args = parser.parse_args()

# set cuda device (based on your hardware)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
nz = eval(exp_config['nz'])

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
preproc = preprocess_imgs

# Get the fixed test set
test_x, test_y = dataset.get_test_set()

# Labels indicating source of the image
real_label = torch.FloatTensor(mb_size).cuda()
real_label.fill_(1)

fake_label = torch.FloatTensor(mb_size).cuda()
fake_label.fill_(0)

# Fix noise to view generated images
eval_noise = torch.FloatTensor(mb_size, nz, 1, 1).normal_(0, 1)
eval_noise_ = np.random.normal(0, 1, (mb_size, nz))
eval_label = np.random.randint(0, 50, mb_size)
eval_onehot = np.zeros((mb_size, 50))
eval_onehot[np.arange(mb_size), eval_label] = 1
eval_noise_[np.arange(mb_size), :50] = eval_onehot[np.arange(mb_size)]
eval_noise_ = (torch.from_numpy(eval_noise_))
eval_noise.data.copy_(eval_noise_.view(mb_size, nz, 1, 1))
eval_noise=eval_noise.cuda()

# Model setup
model = MyMobilenetV1(pretrained=True, latent_layer_num=latent_layer_num)
gen = generator(nz)

gen.apply(weights_init)

# we replace BN layers with Batch Renormalization layers
# ABB: take into account when defining new nets
replace_bn_with_brn(
    model, momentum=init_update_rate, r_d_max_inc_step=inc_step,
    max_r_max=max_r_max, max_d_max=max_d_max
)
model.saved_weights = {}
model.past_j = {i:0 for i in range(50)}
model.cur_j = {i:0 for i in range(50)}
if reg_lambda != 0:
    # the regularization is based on Synaptic Intelligence as described in the
    # paper. ewcData is a list of two elements (best parametes, importance)
    # while synData is a dictionary with all the trajectory data needed by SI
    ewcData, synData = create_syn_data(model)

# Optimizer setup
# ABB: change optimizer to Adam
# optimizer = torch.optim.SGD(
#     model.parameters(), lr=init_lr, momentum=momentum, weight_decay=l2
# )

optimD = torch.optim.Adam(model.parameters(), lr=init_lr, weight_decay=l2)
optimG = torch.optim.Adam(gen.parameters(), lr=init_lr, weight_decay=l2)

# ABB: change loss to BCELoss and NLLLoss
# criterion = torch.nn.CrossEntropyLoss()

source_obj = torch.nn.BCELoss()#source-loss
class_obj = torch.nn.NLLLoss()#class-loss

# --------------------------------- Training -----------------------------------

# loop over the training incremental batches
for i, train_batch in enumerate(dataset):

    if reg_lambda != 0:
        init_batch(model, ewcData, synData)

    # we freeze the layer below the replay layer since the first batch
    freeze_up_to(model, freeze_below_layer, only_conv=False)

    if i == 1:
        # ABB: change accordingly
        change_brn_pars(
            model, momentum=inc_update_rate, r_d_max_inc_step=0,
            r_max=max_r_max, d_max=max_d_max)
        # optimizer = torch.optim.SGD(
        #     model.parameters(), lr=inc_lr, momentum=momentum, weight_decay=l2
        # )
        optimD = torch.optim.Adam(model.parameters(), lr=inc_lr, weight_decay=l2)
        optimG = torch.optim.Adam(gen.parameters(), lr=inc_lr, weight_decay=l2)

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

    # ABB: check if necessary
    reset_weights(model, cur_class)
    cur_ep = 0

    if i == 0:
        (train_x, train_y), it_x_ep = pad_data([train_x, train_y], mb_size)
    shuffle_in_unison([train_x, train_y], in_place=True)

    model = maybe_cuda(model, use_cuda=use_cuda)
    gen = maybe_cuda(gen, use_cuda=use_cuda)
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

            optimD.zero_grad()

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
            logits, source, lat_acts = model(
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

            source_loss = source_obj(source, real_label)
            class_loss = class_obj(logits, y_mb)
            loss_real = source_loss + class_loss

            if reg_lambda !=0:
                loss_real += compute_ewc_loss(model, ewcData, lambd=reg_lambda)
            ave_loss += loss_real.item()

            loss_real.backward()
            optimD.step()

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

            ## Training with fake data now
            noise = torch.FloatTensor(mb_size, nz, 1, 1).normal_(0, 1)
            noise_ = np.random.normal(0, 1, (mb_size, nz))#generating noise by random sampling from a normal distribution

            label = np.random.randint(0,50,mb_size)#generating labels for the entire batch

            noise.data.copy_(eval_noise_.view(mb_size, nz, 1, 1))
            noise = noise.cuda()#converting to tensors in order to work with pytorch

            label = ((torch.from_numpy(label)).long())
            label = label.cuda()#converting to tensors in order to work with pytorch

            noise_image = gen(noise)

            logits, source = model(
                    noise_image.detach(), latent_input=None, return_lat_acts=False)

            print(source.shape)
            print(fake_label.shape)
            print(logits.shape)
            print(label.shape)
            source_loss = source_obj(source, fake_label)
            class_loss = class_obj(logits, label)

            loss_fake = source_loss + class_loss

            loss_fake.backward()
            optimD.step()

            ## Train the generator
            optimG.zero_grad()

            logits, source = model(
                    noise_image, latent_input=None, return_lat_acts=False)

            source_loss = source_obj(source, real_label) #The generator tries to pass its images as real---so we pass the images as real to the cost function
            class_loss = class_obj(logits, label)

            loss_gen = source_loss + class_loss

            loss_gen.backward()
            optimG.step()

        cur_ep += 1

    consolidate_weights(model, cur_class)
    if reg_lambda != 0:
        update_ewc_data(model, ewcData, synData, 0.001, 1)

    ###
    # ABB: adapt this part with output of generator

    # how many patterns to save for next iter
    h = min(rm_sz // (i + 1), cur_acts.size(0))
    print("h", h)

    print("cur_acts sz:", cur_acts.size(0))
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
    ###

    set_consolidate_weights(model)
    ave_loss, acc, accs = get_accuracy(
        model, class_obj, mb_size, test_x, test_y, preproc=preproc
    )

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

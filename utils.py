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

"""
General useful functions for machine learning with Pytorch.
"""

# Python 2-3 compatible
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np
import torch
from models.batch_renorm import BatchRenorm2D

import torch.nn.functional as F
import torch

from scipy.stats import truncnorm

eps = 1e-12


# Convenience utility to switch off requires_grad
def toggle_grad(model, on_or_off):
  for param in model.parameters():
    param.requires_grad = on_or_off

def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

def shuffle_in_unison(dataset, seed=None, in_place=False):
    """
    Shuffle two (or more) list in unison. It's important to shuffle the images
    and the labels maintaining their correspondence.

        Args:
            dataset (dict): list of shuffle with the same order.
            seed (int): set of fixed Cifar parameters.
            in_place (bool): if we want to shuffle the same data or we want
                             to return a new shuffled dataset.
        Returns:
            list: train and test sets composed of images and labels, if in_place
                  is set to False.
    """

    if seed:
        np.random.seed(seed)
    rng_state = np.random.get_state()
    new_dataset = []
    for x in dataset:
        if in_place:
            np.random.shuffle(x)
        else:
            new_dataset.append(np.random.permutation(x))
        np.random.set_state(rng_state)

    if not in_place:
        return new_dataset


def shuffle_in_unison_pytorch(dataset, seed=None):
    """
    Shuffle two (or more) list of torch tensors in unison. It's important to
    shuffle the images and the labels maintaining their correspondence.
    """

    shuffled_dataset = []
    perm = torch.randperm(dataset[0].size(0))
    if seed:
        torch.manual_seed(seed)
    for x in dataset:
        shuffled_dataset.append(x[perm])

    return shuffled_dataset


def pad_data(dataset, mb_size):
    """
    Padding all the matrices contained in dataset to suit the mini-batch
    size. We assume they have the same shape.

        Args:
            dataset (str): sets to pad to reach a multile of mb_size.
            mb_size (int): mini-batch size.
        Returns:
            list: padded data sets
            int: number of iterations needed to cover the entire training set
                 with mb_size mini-batches.
    """

    num_set = len(dataset)
    x = dataset[0]
    # computing test_iters
    n_missing = x.shape[0] % mb_size
    if n_missing > 0:
        surplus = 1
    else:
        surplus = 0
    it = x.shape[0] // mb_size + surplus

    # padding data to fix batch dimentions
    if n_missing > 0:
        n_to_add = mb_size - n_missing
        for i, data in enumerate(dataset):
            dataset[i] = np.concatenate((data[:n_to_add], data))
    if num_set == 1:
        dataset = dataset[0]

    return dataset, it


def get_accuracy(model, criterion, batch_size, test_x, test_y, use_cuda=True,
                 mask=None, preproc=None):
    """
    Test accuracy given a model and the test data.

        Args:
            model (nn.Module): the pytorch model to test.
            criterion (func): loss function.
            batch_size (int): mini-batch size.
            test_x (tensor): test data.
            test_y (tensor): test labels.
            use_cuda (bool): if we want to use gpu or cpu.
            mask (bool): if we want to maks out some classes from the results.
        Returns:
            ave_loss (float): average loss across the test set.
            acc (float): average accuracy.
            accs (list): average accuracy for class.
    """

    model.eval()

    correct_cnt, ave_loss = 0, 0
    model = maybe_cuda(model, use_cuda=use_cuda)

    num_class = int(np.max(test_y) + 1)
    hits_per_class = [0] * num_class
    pattern_per_class = [0] * num_class
    test_it = test_y.shape[0] // batch_size + 1

    test_x = torch.from_numpy(test_x).type(torch.FloatTensor)
    test_y = torch.from_numpy(test_y).type(torch.LongTensor)

    if preproc:
        test_x = preproc(test_x, norm = True)

    for i in range(test_it):
        # indexing
        start = i * batch_size
        end = (i + 1) * batch_size

        x = maybe_cuda(test_x[start:end], use_cuda=use_cuda)
        y = maybe_cuda(test_y[start:end], use_cuda=use_cuda)

        logits = model(x)

        if mask is not None:
            # we put an high negative number so that after softmax that prob
            # will be zero and not contribute to the loss
            idx = (torch.FloatTensor(mask).cuda() == 0).nonzero()
            idx = idx.view(idx.size(0))
            logits[:, idx] = -10e10

        loss = criterion(logits, y)
        _, pred_label = torch.max(logits.data, 1)
        correct_cnt += (pred_label == y.data).sum()
        ave_loss += loss.item()

        for label in y.data:
            pattern_per_class[int(label)] += 1

        for i, pred in enumerate(pred_label):
            if pred == y.data[i]:
                hits_per_class[int(pred)] += 1

    accs = np.asarray(hits_per_class) / \
           np.asarray(pattern_per_class).astype(float)

    acc = correct_cnt.item() * 1.0 / test_y.size(0)

    ave_loss /= test_y.size(0)

    return ave_loss, acc, accs

def get_accuracy_custom(model, criterion, batch_size, test_x, test_y, device,
                 use_cuda):
    """
    Test accuracy given a model and the test data.

        Args:
            model (nn.Module): the pytorch model to test.
            criterion (func): loss function.
            batch_size (int): mini-batch size.
            test_x (tensor): test data.
            test_y (tensor): test labels.
            use_cuda (bool): if we want to use gpu or cpu.
            mask (bool): if we want to maks out some classes from the results.
        Returns:
            ave_loss (float): average loss across the test set.
            acc (float): average accuracy.
            accs (list): average accuracy for class.
    """
    correct_cnt, ave_loss = 0, 0

    num_class = int(torch.max(test_y) + 1)
    hits_per_class = [0] * num_class
    pattern_per_class = [0] * num_class
    test_it = test_y.shape[0] // batch_size + 1

    with torch.no_grad():

        for i in range(test_it):
            # indexing
            start = i * batch_size
            end = (i + 1) * batch_size

            x = maybe_cuda(test_x[start:end], use_cuda=use_cuda).to(device)
            y = maybe_cuda(test_y[start:end], use_cuda=use_cuda).to(device)

            pred, classes = model([im.detach() for im in x], "fake") # actually they are real images

            loss = F.relu( torch.rand_like(pred) * 0.2 + 0.8 + pred).mean()
            conditioned_loss = criterion(torch.log(classes+eps),y)
            loss += conditioned_loss

            _, pred_label = torch.max(classes, 1)
            correct_cnt += (pred_label == y).sum()
            ave_loss += loss.item()

            for label in y.data:
                pattern_per_class[int(label)] += 1

            for i, p in enumerate(pred_label):
                if p == y.data[i]:
                    hits_per_class[int(p)] += 1

        accs = np.asarray(hits_per_class) / \
               np.asarray(pattern_per_class).astype(float)

        acc = correct_cnt.item() * 1.0 / test_y.size(0)

        ave_loss /= test_y.size(0)

        del x
        del y
        torch.cuda.empty_cache()

    return ave_loss, acc, accs

def preprocess_imgs(img_batch, scale=True, norm=True, channel_first=True, symmetric=False):
    """
    Here we get a batch of PIL imgs and we return them normalized as for
    the pytorch pre-trained models.

        Args:
            img_batch (tensor): batch of images.
            scale (bool): if we want to scale the images between 0 an 1.
            channel_first (bool): if the channel dimension is before of after
                                  the other dimensions (width and height).
            norm (bool): if we want to normalize them.
        Returns:
            tensor: pre-processed batch.

    """

    if scale:
        # convert to float in [0, 1]
        img_batch = img_batch / 255
        if symmetric:
            img_batch = (img_batch * 2) - 1

    if norm:
        # normalize
        img_batch[:, :, :, 0] = ((img_batch[:, :, :, 0] - 0.485) / 0.229)
        img_batch[:, :, :, 1] = ((img_batch[:, :, :, 1] - 0.456) / 0.224)
        img_batch[:, :, :, 2] = ((img_batch[:, :, :, 2] - 0.406) / 0.225)

    if channel_first:
        # Swap channel dimension to fit the caffe format (c, w, h)
        img_batch = np.transpose(img_batch, (0, 3, 1, 2))

    return img_batch


def maybe_cuda(what, use_cuda=True, **kw):
    """
    Moves `what` to CUDA and returns it, if `use_cuda` and it's available.

        Args:
            what (object): any object to move to eventually gpu
            use_cuda (bool): if we want to use gpu or cpu.
        Returns
            object: the same object but eventually moved to gpu.
    """

    if use_cuda is not False and torch.cuda.is_available():
        what = what.cuda()
    return what


def replace_bn_with_brn(
        m, name="", momentum=0.1, r_d_max_inc_step=0.0001, r_max=1.0,
        d_max=0.0, max_r_max=3.0, max_d_max=5.0):
    for child_name, child in m.named_children():
        if isinstance(child, torch.nn.BatchNorm2d):
            setattr(m, child_name, BatchRenorm2D(
                child.num_features,
                gamma=child.weight,
                beta=child.bias,
                running_mean=child.running_mean,
                running_var=child.running_var,
                eps=child.eps,
                momentum=momentum,
                r_d_max_inc_step=r_d_max_inc_step,
                r_max=r_max,
                d_max=d_max,
                max_r_max=max_r_max,
                max_d_max=max_d_max
            ))
        else:
            replace_bn_with_brn(child, child_name, momentum, r_d_max_inc_step, r_max, d_max,
                                max_r_max, max_d_max)


def change_brn_pars(
        m, name="", momentum=0.1, r_d_max_inc_step=0.0001, r_max=1.0,
        d_max=0.0):
    for target_name, target_attr in m.named_children():
        if isinstance(target_attr, BatchRenorm2D):
            target_attr.momentum = torch.tensor(momentum, requires_grad=False)
            target_attr.r_max = torch.tensor(r_max, requires_grad=False)
            target_attr.d_max = torch.tensor(d_max, requires_grad=False)
            target_attr.r_d_max_inc_step = r_d_max_inc_step

        else:
            change_brn_pars(target_attr, target_name, momentum, r_d_max_inc_step, r_max, d_max)


def consolidate_weights(model, cur_clas):
    """ Mean-shift for the target layer weights"""

    with torch.no_grad():
        globavg = np.average(model.output.weight.detach()
                             .cpu().numpy()[cur_clas])
        for c in cur_clas:
            w = model.output.weight.detach().cpu().numpy()[c]

            if c in cur_clas:
                new_w = w - globavg
                if c in model.saved_weights.keys():
                    wpast_j = np.sqrt(model.past_j[c] / model.cur_j[c])
                    model.saved_weights[c] = (model.saved_weights[c] * wpast_j
                     + new_w) / (wpast_j + 1)
                else:
                    model.saved_weights[c] = new_w


def set_consolidate_weights(model):
    """ set trained weights """

    with torch.no_grad():
        for c, w in model.saved_weights.items():
            model.output.weight[c].copy_(
                torch.from_numpy(model.saved_weights[c])
            )


def reset_weights(model, cur_clas):
    """ reset weights"""

    with torch.no_grad():
        model.output.weight.fill_(0.0)
        for c, w in model.saved_weights.items():
            if c in cur_clas:
                model.output.weight[c].copy_(
                    torch.from_numpy(model.saved_weights[c])
                )


def examples_per_class(train_y):
    count = {i:0 for i in range(50)}
    for y in train_y:
        count[int(y)] +=1

    return count


def set_brn_to_train(m, name=""):
    for target_name, target_attr in m.named_children():
        if isinstance(target_attr, BatchRenorm2D):
            target_attr.train()
        else:
            set_brn_to_train(target_attr, target_name)


def set_brn_to_eval(m, name=""):
    for target_name, target_attr in m.named_children():
        if isinstance(target_attr, BatchRenorm2D):
            target_attr.eval()
        else:
            set_brn_to_eval(target_attr, target_name)


def set_bn_to(m, name="", phase="train"):
    for target_name, target_attr in m.named_children():
        if isinstance(target_attr, torch.nn.BatchNorm2d):
            if phase == "train":
                target_attr.train()
            else:
                target_attr.eval()
        else:
            set_bn_to(target_attr, target_name, phase)


def freeze_up_to(model, freeze_below_layer, only_conv=False):
    for name, param in model.named_parameters():
        # tells whether we want to use gradients for a given parameter
        if only_conv:
            if "conv" in name:
                param.requires_grad = False
                #print("Freezing parameter " + name)
        else:
            param.requires_grad = False
            #print("Freezing parameter " + name)

        if name == freeze_below_layer:
            break


def create_syn_data(model):
    size = 0
    print('Creating Syn data for Optimal params and their Fisher info')

    for name, param in model.named_parameters():
        if "bn" not in name and "output" not in name:
            print(name, param.flatten().size(0))
            size += param.flatten().size(0)

    # The first array returned is a 2D array: the first component contains
    # the params at loss minimum, the second the parameter importance
    # The second array is a dictionary with the synData
    synData = {}
    synData['old_theta'] = torch.zeros(size, dtype=torch.float32)
    synData['new_theta'] = torch.zeros(size, dtype=torch.float32)
    synData['grad'] = torch.zeros(size, dtype=torch.float32)
    synData['trajectory'] = torch.zeros(size, dtype=torch.float32)
    synData['cum_trajectory'] = torch.zeros(size, dtype=torch.float32)

    return torch.zeros((2, size), dtype=torch.float32), synData


def extract_weights(model, target):

    with torch.no_grad():
        weights_vector= None
        for name, param in model.named_parameters():
            if "bn" not in name and "output" not in name:
                # print(name, param.flatten())
                if weights_vector is None:
                    weights_vector = param.flatten()
                else:
                    weights_vector = torch.cat(
                        (weights_vector, param.flatten()), 0)

        target[...] = weights_vector.cpu()


def extract_grad(model, target):
    # Store the gradients into target
    with torch.no_grad():
        grad_vector= None
        for name, param in model.named_parameters():
            if "bn" not in name and "output" not in name:
                # print(name, param.flatten())
                if grad_vector is None:
                    grad_vector = param.grad.flatten()
                else:
                    grad_vector = torch.cat(
                        (grad_vector, param.grad.flatten()), 0)

        target[...] = grad_vector.cpu()


def init_batch(net, ewcData, synData):
    # Keep initial weights
    extract_weights(net, ewcData[0])
    synData['trajectory'] = 0


def pre_update(net, synData):
    extract_weights(net, synData['old_theta'])


def post_update(net, synData):
    extract_weights(net, synData['new_theta'])
    extract_grad(net, synData['grad'])

    synData['trajectory'] += synData['grad'] * (
                    synData['new_theta'] - synData['old_theta'])


def update_ewc_data(net, ewcData, synData, clip_to, c=0.0015):
    extract_weights(net, synData['new_theta'])
    eps = 0.0000001  # 0.001 in few task - 0.1 used in a more complex setup

    synData['cum_trajectory'] += c * synData['trajectory'] / (
                    np.square(synData['new_theta'] - ewcData[0]) + eps)

    ewcData[1] = torch.empty_like(synData['cum_trajectory'])\
        .copy_(-synData['cum_trajectory'])

    ewcData[1] = torch.clamp(ewcData[1], max=clip_to)
    # (except CWR)
    ewcData[0] = synData['new_theta'].clone().detach()


def compute_ewc_loss(model, ewcData, lambd=0):

    weights_vector = None
    for name, param in model.named_parameters():
        if "bn" not in name and "output" not in name:
            if weights_vector is None:
                weights_vector = param.flatten()
            else:
                weights_vector = torch.cat(
                    (weights_vector, param.flatten()), 0)

    ewcData = maybe_cuda(ewcData, use_cuda=True)
    loss = (lambd / 2) * torch.dot(ewcData[1], (weights_vector - ewcData[0])**2)
    return loss

def weights_init(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		m.weight.data.normal_(0.0, 0.02)
	elif classname.find('BatchNorm') != -1:
		m.weight.data.normal_(1.0, 0.02)
		m.bias.data.fill_(0)

class ema(object):
  def __init__(self, source, target, decay=0.9999, start_itr=0):
    self.source = source
    self.target = target
    self.decay = decay
    # Optional parameter indicating what iteration to start the decay at
    self.start_itr = start_itr
    # Initialize target's params to be source's
    self.source_dict = self.source.state_dict()
    self.target_dict = self.target.state_dict()
    print('Initializing EMA parameters to be source parameters...')
    with torch.no_grad():
      for key in self.source_dict:
        self.target_dict[key].data.copy_(self.source_dict[key].data)
        # target_dict[key].data = source_dict[key].data # Doesn't work!

  def update(self, itr=None):
    # If an iteration counter is provided and itr is less than the start itr,
    # peg the ema weights to the underlying weights.
    if itr and itr < self.start_itr:
      decay = 0.0
    else:
      decay = self.decay
    with torch.no_grad():
      for key in self.source_dict:
        self.target_dict[key].data.copy_(self.target_dict[key].data * decay
                                     + self.source_dict[key].data * (1 - decay))
def train_gan(num_epochs, batch_size, train_x, train_y, use_cuda, disc, gen, optimD, optimG, criterion_class, criterion_source, writer):
    # Training Loop
    print("Starting GAN Training Loop...")

    tot_it_step = 0
    it_x_ep = train_x.size(0) // batch_size

    for ep in range(num_epochs):
        print("training gan ep: ", ep)
        for i in range(it_x_ep):

            start = i * batch_size
            end = (i + 1) * batch_size

            x_mb = maybe_cuda(train_x[start:end], use_cuda=use_cuda)
            y_mb = maybe_cuda(train_y[start:end], use_cuda=use_cuda)

            disc.train()

            disc = maybe_cuda(disc, use_cuda=use_cuda)
            gen = maybe_cuda(gen, use_cuda=use_cuda)
            acc = None
            ave_loss = 0

            correct_cnt, ave_loss, correct_src, correct_src_fake, ave_loss_gen = 0, 0, 0, 0, 0
            data_encountered = 0

            optimD.zero_grad()

            classes, source = disc(x_mb)

            # Labels indicating source of the image
            real_label = maybe_cuda(torch.FloatTensor(y_mb.size(0)), use_cuda=use_cuda)
            real_label.fill_(0.9)

            fake_label = maybe_cuda(torch.FloatTensor(y_mb.size(0)), use_cuda=use_cuda)
            fake_label.fill_(0.1)

            _, pred_label = torch.max(classes, 1)
            correct_cnt += (pred_label == y_mb).sum()

            pred_source = torch.round(source)
            correct_src += (pred_source == 1).sum()

            loss = criterion_class(classes, y_mb) + criterion_source(source, real_label)

            loss.backward()
            optimD.step()

            ave_loss += loss.item()
            data_encountered += y_mb.size(0)

            acc = correct_cnt.item() / data_encountered
            ave_loss /= data_encountered
            source_acc = correct_src.item() / data_encountered

            ## Training with fake data now
            noise = torch.FloatTensor(y_mb.size(0), nz, 1, 1).normal_(0, 1)
            #noise_ = np.random.normal(0, 1, (y_mb.size(0), nz))
            noise_ = trunc_normal3.rvs(y_mb.size(0)*nz, 0)
            noise_ = noise_.reshape(y_mb.size(0),nz)
            label = np.random.randint(0, n_class, y_mb.size(0))
            onehot = np.zeros((y_mb.size(0), n_class))
            onehot[np.arange(y_mb.size(0)), label] = 1
            noise_[np.arange(y_mb.size(0)), :n_class] = onehot[np.arange(y_mb.size(0))]
            noise_ = (torch.from_numpy(noise_))
            noise.data.copy_(noise_.view(y_mb.size(0), nz, 1, 1))
            noise = maybe_cuda(noise, use_cuda=use_cuda)

            label = ((torch.from_numpy(label)).long())
            label = maybe_cuda(label, use_cuda=use_cuda)

            noise_image = gen(noise)

            classes, source = disc(noise_image.detach())

            pred_source = torch.round(source)
            correct_src_fake += (pred_source == 0).sum()

            loss_fake = criterion_source(source, fake_label) + criterion_class(classes, label)

            loss_fake.backward()
            optimD.step()

            source_acc_fake = correct_src_fake.item() / data_encountered

            ## Train the generator
            optimG.zero_grad()

            classes, source = disc(noise_image)

            source_loss = criterion_source(source, real_label) #The generator tries to pass its images as real---so we pass the images as real to the cost function
            class_loss = criterion_class(classes, label)

            loss_gen = source_loss + class_loss

            loss_gen.backward()
            optimG.step()

            ave_loss_gen += loss_gen.item()
            ave_loss_gen /= data_encountered

            # Output training stats
            if i % 5 == 0:
                print(
                    '==>>> it: {}, avg. loss: {:.6f}, '
                    'running train acc: {:.3f}, '
                    'running source acc: {:.3f}, '
                    'running source acc fake: {:.3f}, '
                    'running avg. loss gen: {:.3f}'
                        .format(i, ave_loss, acc, source_acc, source_acc_fake, ave_loss_gen)
                )

            tot_it_step +=1

            writer.add_scalar('train_loss', ave_loss, tot_it_step)
            writer.add_scalar('train_accuracy', acc, tot_it_step)
            writer.add_scalar('source_accuracy', source_acc, tot_it_step)
            writer.add_scalar('source_fake_accuracy', source_acc_fake, tot_it_step)
            writer.add_scalar('gen_loss', ave_loss_gen, tot_it_step)

            writer.close()


if __name__ == "__main__":

    from models.mobilenet import MyMobilenetV1
    model = MyMobilenetV1(pretrained=True)
    replace_bn_with_brn(model, "net")

    ewcData, synData = create_syn_data(model)
    extract_weights(model, ewcData[0])

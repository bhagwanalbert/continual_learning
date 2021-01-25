import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from torchvision import utils as vutils

import argparse
from tqdm import tqdm

from fastgan_models import weights_init, Discriminator, Generator
from operation import copy_G_params, load_params, get_dir
from operation import ImageFolder, InfiniteSamplerWrapper
from diffaug import DiffAugment
from data_loader import CORE50
from torch.utils.tensorboard import SummaryWriter
from utils import *
policy = 'color,translation'
from PerceptualSimilarity import models
import os

import random

import numpy as np

# Set cuda device (based on your hardware)
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"

percept = models.PerceptualLoss(model='net-lin', net='vgg', use_gpu=True, gpu_ids=[5])

class_loss = nn.NLLLoss()
correct_cnt = 0

#torch.backends.cudnn.benchmark = True

# Create tensorboard writer object
writer = SummaryWriter('logs/fastgan5')

def crop_image_by_part(image, part):
    hw = image.shape[2]//2
    if part==0:
        return image[:,:,:hw,:hw]
    if part==1:
        return image[:,:,:hw,hw:]
    if part==2:
        return image[:,:,hw:,:hw]
    if part==3:
        return image[:,:,hw:,hw:]

def train_d(net, data, y, label="real"):
    """Train function of discriminator"""
    global correct_cnt
    part = random.randint(0, 3)
    if label=="real":
        pred, [rec_all, rec_small, rec_part], classes = net(data, label, part)
        err = F.relu(  torch.rand_like(pred) * 0.2 + 0.8 -  pred).mean() + \
            percept( rec_all, F.interpolate(data, rec_all.shape[2]) ).sum() +\
            percept( rec_small, F.interpolate(data, rec_small.shape[2]) ).sum() +\
            percept( rec_part, F.interpolate(crop_image_by_part(data, part), rec_part.shape[2]) ).sum()
        conditioned_loss = class_loss(torch.log(classes),y)
        err += conditioned_loss
        err.backward()
        _, pred_label = torch.max(classes, 1)
        correct_cnt += (pred_label == y).sum()
        return pred.mean().item(), rec_all, rec_small, rec_part, conditioned_loss
    else:
        pred, classes = net(data, label)
        err = F.relu( torch.rand_like(pred) * 0.2 + 0.8 + pred).mean()
        conditioned_loss = class_loss(torch.log(classes),y)
        err += conditioned_loss
        err.backward()
        return pred.mean().item(), conditioned_loss


def train(args):
    global correct_cnt

    checkpoint = args.ckpt
    batch_size = args.batch_size
    im_size = args.im_size
    ndf = 64
    ngf = 64
    n_class = 50
    nz = 256 + n_class
    nlr = 0.0002
    ilr = nlr/4
    nbeta1 = 0.5
    use_cuda = True
    multi_gpu = False
    dataloader_workers = 8
    start_batch = 0
    num_epochs = 100
    inum_epochs = 1
    n_imag = 10
    prev_imag = 10
    n_im_mb = 2

    saved_model_folder, saved_image_folder = get_dir(args)

    transform_list = [
            transforms.ToPILImage(mode='RGB'),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]

    transform_list_aux = [
            transforms.ToPILImage(mode='RGB'),
            transforms.Resize((int(im_size),int(im_size))),
            transforms.ToTensor(),
        ]

    data_transforms = transforms.Compose(transform_list)
    data_transforms_aux = transforms.Compose(transform_list_aux)

    dataset = CORE50(root='/home/abhagwan/datasets/core50', scenario="nicv2_391")

    netG = Generator(ngf=ngf, nz=nz, im_size=im_size)
    netG.apply(weights_init)

    netD = Discriminator(ndf=ndf, im_size=im_size, n_class=n_class)
    netD.apply(weights_init)

    netG = maybe_cuda(netG, use_cuda=use_cuda).to('cuda:5')
    netD = maybe_cuda(netD, use_cuda=use_cuda).to('cuda:5')

    avg_param_G = copy_G_params(netG)

    fixed_noise = torch.FloatTensor(n_imag*n_class, nz).normal_(0, 1)
    fixed_noise_ = np.random.normal(0, 1, (n_imag*n_class, nz))
    eval_onehot = np.zeros((n_imag*n_class, n_class))

    for c in range(n_class):
        eval_onehot[np.arange(n_imag*c,n_imag*(c+1)), c] = 1

    fixed_noise_[np.arange(n_imag*n_class), :n_class] = eval_onehot[np.arange(n_imag*n_class)]

    fixed_noise_ = (torch.from_numpy(fixed_noise_))
    fixed_noise.data.copy_(fixed_noise_.view(n_imag*n_class, nz))
    fixed_noise = maybe_cuda(fixed_noise, use_cuda=use_cuda).to('cuda:5')

    if multi_gpu:
        netG = nn.DataParallel(netG,device_ids=[5, 0, 1, 2, 3, 4, 6, 7, 8, 9])
        netD = nn.DataParallel(netD,device_ids=[5, 0, 1, 2, 3, 4, 6, 7, 8, 9])

    optimizerG = optim.Adam(netG.parameters(), lr=nlr, betas=(nbeta1, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=nlr, betas=(nbeta1, 0.999))

    print(optimizerG)

    enc_classes = {i:0 for i in range(n_class)}
    if checkpoint != 'None':
        ckpt = torch.load(saved_model_folder+"/"+checkpoint)
        if not multi_gpu:
            netG.load_state_dict(dict(zip(netG.state_dict().keys(), ckpt['g'].values())))
            netD.load_state_dict(dict(zip(netD.state_dict().keys(), ckpt['d'].values())))
        else:
            netG.load_state_dict(ckpt['g'])
            netD.load_state_dict(ckpt['d'])
        avg_param_G = ckpt['g_ema']
        optimizerG = optim.Adam(netG.parameters(), lr=ilr, betas=(nbeta1, 0.999))
        optimizerD = optim.Adam(netD.parameters(), lr=ilr, betas=(nbeta1, 0.999))
        start_batch = int(checkpoint.split('_')[-2].split('.')[0])+1
        enc_classes = ckpt['trained_classes']
        del ckpt

    # Training Loop
    print("Starting Training Loop...")
    tot_it_step = 0
    for i, train_batch in enumerate(dataset):

        print("Incremental batch no.: ", i)
        if (i < start_batch):
            print("Skipping batch, already trained in checkpoint")
            continue

        train_x, train_y = train_batch
        train_x = preprocess_imgs(train_x, norm=False, symmetric = False)

        train_x = torch.from_numpy(train_x).type(torch.FloatTensor)
        train_y = torch.from_numpy(train_y).type(torch.LongTensor)

        indexes = np.random.permutation(train_y.size(0))

        train_x = train_x[indexes]
        train_y = train_y[indexes]

        # Show some new training images
        training_examples = None

        for c in range(n_class):
            if training_examples == None:
                training_examples = train_x[train_y.numpy() == c][:n_imag]
            else:
                training_examples = torch.cat((training_examples, train_x[train_y.numpy() == c][:n_imag]))

        writer.add_image("Training images", vutils.make_grid(training_examples, nrow=n_imag, padding=2, normalize=True).cpu())
        writer.close()

        train_x_proc = torch.zeros([train_x.size(0),train_x.size(1),im_size,im_size]).type(torch.FloatTensor)
        for im in range(train_x.shape[0]):
            im_proc = data_transforms_aux((train_x[im]).cpu())
            train_x_proc[im] = im_proc.type(torch.FloatTensor)

        # Add images from previous generator
        if i != 0:
            prev_label = np.array(list({x:enc_classes[x] for x in enc_classes if enc_classes[x]==1}.keys()))
            # Compute noise to generate previous learnt images
            prev_noise = torch.FloatTensor(prev_imag*len(prev_label), nz).normal_(0, 1)
            prev_noise_ = np.random.normal(0, 1, (prev_imag*len(prev_label), nz))
            prev_onehot = np.zeros((prev_imag*len(prev_label), n_class))
            prev_onehot[np.arange(prev_imag*len(prev_label)), np.repeat(prev_label,prev_imag)] = 1
            prev_noise_[np.arange(prev_imag*len(prev_label)), :n_class] = prev_onehot[np.arange(prev_imag*len(prev_label))]
            prev_noise_ = (torch.from_numpy(prev_noise_))
            prev_noise.data.copy_(prev_noise_.view(prev_imag*len(prev_label), nz))
            prev_noise = maybe_cuda(prev_noise, use_cuda=use_cuda).to('cuda:5')

            prev_y = ((torch.from_numpy(np.repeat(prev_label,prev_imag))).long())
            prev_y = maybe_cuda(prev_y, use_cuda=use_cuda).to('cuda:5')

            backup_para = copy_G_params(netG)
            load_params(netG, avg_param_G)
            with torch.no_grad():
                prev_x = netG(prev_noise)[0].add(1).mul(0.5)
                part = random.randint(0, 3)
                pred, [rec_all, rec_small, rec_part], classes = netD(prev_x, "real", part)
                _, pred_label = torch.max(classes, 1)
                filter = pred_label == prev_y
                correct_prev = filter.sum()
                print(correct_prev.item()/prev_y.size(0))
                # writer.add_image("Previous images", vutils.make_grid(prev_x, nrow=prev_imag, padding=2, normalize=True))
                prev_x_filt = torch.zeros([correct_prev.item(),prev_x.size(1),im_size,im_size]).type(torch.FloatTensor)
                prev_x_filt = maybe_cuda(prev_x_filt, use_cuda=use_cuda).to('cuda:5')
                prev_y_filt = torch.zeros([correct_prev.item()]).type(torch.LongTensor)
                prev_y_filt = maybe_cuda(prev_y_filt, use_cuda=use_cuda).to('cuda:5')
                idx = 0
                for f in range(filter.size(0)):
                    prev_x[f] = filter[f]*prev_x[f]
                    if filter[f] == 1:
                        prev_x_filt[idx] = prev_x[f]
                        prev_y_filt[idx] = prev_y[f]
                        idx += 1
                prev_x = prev_x_filt
                prev_y = prev_y_filt
                del prev_x_filt
                del prev_y_filt
                # writer.add_image("Previous images", vutils.make_grid(prev_x, nrow=prev_imag, padding=2, normalize=True))
                writer.add_image("Previous images", vutils.make_grid(prev_x, nrow=prev_imag, padding=2, normalize=True))
                writer.close()
            load_params(netG, backup_para)

        # Update encountered classes
        for y in train_y:
            enc_classes[y.item()] |= 1
        print(enc_classes)

        train_x = train_x_proc
        train_x_proc = torch.zeros([train_x.size(0),train_x.size(1),im_size,im_size]).type(torch.FloatTensor)

        if i != 0:
            prev_x_proc = torch.zeros([prev_x.size(0),prev_x.size(1),im_size,im_size]).type(torch.FloatTensor)
            current_batch_size = (prev_label.size + 1)*n_im_mb
            num_epochs = inum_epochs
            it_x_ep = train_x.size(0) // n_im_mb
            print(it_x_ep)
        else:
            it_x_ep = train_x.size(0) // batch_size

        for ep in range(num_epochs):
            print("training ep: ", ep)
            data_encountered = 0
            correct_cnt = 0

            for im in range(train_x.shape[0]):
                im_proc = data_transforms((train_x[im]).cpu())
                train_x_proc[im] = im_proc.type(torch.FloatTensor)
            if i != 0:
                for im in range(prev_x.shape[0]):
                    im_proc = data_transforms((prev_x[im]).cpu())
                    prev_x_proc[im] = im_proc.type(torch.FloatTensor)

            for it in range(it_x_ep):

                if i != 0:
                    start = it * n_im_mb
                    end = (it + 1) * n_im_mb
                    real_image = maybe_cuda(train_x_proc[start:end], use_cuda=use_cuda).to('cuda:5')
                    y_mb = maybe_cuda(train_y[start:end], use_cuda=use_cuda).to('cuda:5')

                    for c in prev_label:
                        prev_x_aux = prev_x_proc[prev_y.cpu().numpy() == c]
                        prev_y_aux = prev_y[prev_y.cpu().numpy() == c]
                        indexes = np.random.randint(0, (prev_x_aux.size(0)), size = n_im_mb)
                        real_image = torch.cat((real_image, maybe_cuda(prev_x_aux[indexes], use_cuda=use_cuda).to('cuda:5')))
                        y_mb = torch.cat((y_mb, maybe_cuda(prev_y_aux[indexes], use_cuda=use_cuda).to('cuda:5')))

                    del prev_x_aux
                    del prev_y_aux

                    indexes = np.random.permutation(y_mb.size(0))

                    real_image = real_image[indexes]
                    y_mb = y_mb[indexes]

                else:
                    start = it * batch_size
                    end = (it + 1) * batch_size

                    real_image = maybe_cuda(train_x_proc[start:end], use_cuda=use_cuda).to('cuda:5')
                    y_mb = maybe_cuda(train_y[start:end], use_cuda=use_cuda).to('cuda:5')

                    current_batch_size = real_image.size(0)

                data_encountered += current_batch_size

                current_classes = np.array(list({x:enc_classes[x] for x in enc_classes if enc_classes[x]==1}.keys()))

                noise = torch.FloatTensor(current_batch_size, nz).normal_(0, 1)
                noise_ = np.random.normal(0, 1, (current_batch_size, nz))
                label = np.random.choice(current_classes, current_batch_size)
                onehot = np.zeros((current_batch_size, n_class))
                onehot[np.arange(current_batch_size), label] = 1
                noise_[np.arange(current_batch_size), :n_class] = onehot[np.arange(current_batch_size)]
                noise_ = (torch.from_numpy(noise_))
                noise.data.copy_(noise_.view(current_batch_size, nz))
                noise = maybe_cuda(noise, use_cuda=use_cuda).to('cuda:5')

                label = ((torch.from_numpy(label)).long())
                label = maybe_cuda(label, use_cuda=use_cuda).to('cuda:5')

                fake_images = netG(noise)

                real_image = DiffAugment(real_image, policy=policy)
                fake_images = [DiffAugment(fake, policy=policy) for fake in fake_images]

                ## 2. train Discriminator
                netD.zero_grad()

                err_dr_real, rec_img_all, rec_img_small, rec_img_part, err_class_real = train_d(netD, real_image, y_mb, label="real")
                class_acc = correct_cnt.item() / data_encountered
                err_dr_fake, err_class_fake = train_d(netD, [fi.detach() for fi in fake_images], label, label="fake")
                optimizerD.step()

                ## 3. train Generator
                netG.zero_grad()
                pred_g, classes = netD(fake_images, "fake")
                err_class_gen = class_loss(torch.log(classes),label)
                err_g = -pred_g.mean() + err_class_gen

                err_g.backward()
                optimizerG.step()

                for p, avg_p in zip(netG.parameters(), avg_param_G):
                    avg_p.mul_(0.999).add_(0.001 * p.data)

                if it % 20 == 0:
                    print("GAN: loss d: %.5f    loss g: %.5f"%(err_dr_real, -err_g.item()))

            tot_it_step +=1

            writer.add_scalar('class_accuracy', class_acc, tot_it_step)
            writer.add_scalar('discriminator_real_loss', err_dr_real, tot_it_step)
            writer.add_scalar('discriminator_fake_loss', err_dr_fake, tot_it_step)
            writer.add_scalar('discriminator_class_real_loss', err_class_real, tot_it_step)
            writer.add_scalar('discriminator_class_fake_loss', err_class_fake, tot_it_step)
            writer.add_scalar('generator_loss', -err_g.item(), tot_it_step)
            writer.add_scalar('generator_class_loss', err_class_gen, tot_it_step)
            writer.close()

            # backup_para = copy_G_params(netG)
            # load_params(netG, avg_param_G)
            # torch.save({'g':netG.state_dict(),'d':netD.state_dict()}, saved_model_folder+'/%d_%d.pth'%(i,ep))
            # load_params(netG, backup_para)
            if (ep == num_epochs - 1):
                print("saving in: /all_%d_%d.pth"%(i,ep))
                torch.save({'g':netG.state_dict(),
                            'd':netD.state_dict(),
                            'g_ema': avg_param_G,
                            'opt_g': optimizerG.state_dict(),
                            'opt_d': optimizerD.state_dict(),
                            'trained_classes': enc_classes}, saved_model_folder+'/all_%d_%d.pth'%(i,ep))

        backup_para = copy_G_params(netG)
        load_params(netG, avg_param_G)
        with torch.no_grad():
            writer.add_image("Generated images", vutils.make_grid(netG(fixed_noise)[0].add(1).mul(0.5), nrow=n_imag, padding=2, normalize=True))
        load_params(netG, backup_para)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='region gan')

    parser.add_argument('--cuda', type=int, default=1, help='index of gpu to use')
    parser.add_argument('--name', type=str, default='test1', help='experiment name')
    parser.add_argument('--start_iter', type=int, default=0, help='the iteration to start training')
    parser.add_argument('--batch_size', type=int, default=20, help='mini batch number of images')
    parser.add_argument('--im_size', type=int, default=256, help='image resolution')
    parser.add_argument('--ckpt', type=str, default='None', help='checkpoint weight path')


    args = parser.parse_args()
    print(args)

    train(args)

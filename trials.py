import torch
import torch.nn as nn
from fastgan_models import weights_init, Discriminator, Generator
import torch.nn.functional as F
from PerceptualSimilarity import models
from utils import *
import random

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

netD = Discriminator(ndf=64, im_size=256, n_class=10)
netD.apply(weights_init)
netD = netD.to("cuda:5")
netD = nn.DataParallel(netD,device_ids=[5])
class_loss = nn.NLLLoss()
percept = models.PerceptualLoss(model='net-lin', net='vgg', use_gpu=True, gpu_ids=[5])

data = torch.randn(12,3,256,256).to("cuda:5")
part = random.randint(0, 3)
y = torch.randint(10,(12,)).to("cuda:5")
pred, [rec_all, rec_small, rec_part], classes = netD(data, "real", part)

err = F.relu(  torch.rand_like(pred) * 0.2 + 0.8 -  pred).mean() + \
    percept( rec_all, F.interpolate(data, rec_all.shape[2]) ).sum() +\
    percept( rec_small, F.interpolate(data, rec_small.shape[2]) ).sum() +\
    percept( rec_part, F.interpolate(crop_image_by_part(data, part), rec_part.shape[2]) ).sum()
conditioned_loss = class_loss(torch.log(classes),y)
err += conditioned_loss
err.backward()

print(pred.shape)
print(rec_all.shape)
print(rec_small.shape)
print(rec_part.shape)
print(classes.shape)

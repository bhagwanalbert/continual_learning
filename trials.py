import torch
import torch.nn as nn
from fastgan_models import weights_init, Discriminator, Generator

netD = Discriminator(ndf=64, im_size=256, n_class=10)
netD.apply(weights_init)
netD = netD.to("cuda:0")
netD = nn.DataParallel(netD,device_ids=[0, 1, 2, 3, 4])

pred, [rec_all, rec_small, rec_part], classes = netD(torch.randn(20,3,256,256).to("cuda:0"), "real", 2)

print(pred.shape)
print(rec_all.shape)
print(rec_small.shape)
print(rec_part.shape)
print(classes.shape)

# class dummy_net(nn.Module):
#     def __init__(self):
#         super(dummy_net, self).__init__()
#     def forward(self, input):
#         return input, [torch.zeros(1).to(input.device),torch.ones(1).to(input.device),2*torch.ones(1).to(input.device)], torch.zeros(4,4).to(input.device), torch.ones(10).to(input.device)

# model = dummy_net()
# model = nn.DataParallel(model, device_ids=[0,1,2,3,4])
#
# a, [b,c,d], e, f = model(torch.randn(100,2,2).to("cuda:0"))

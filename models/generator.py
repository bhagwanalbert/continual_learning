import torch
from torch import nn

class Interpolate(nn.Module):
    def __init__(self, size, mode):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode

    def forward(self, x):
        x = self.interp(x, size=self.size, mode=self.mode)
        return x

class generator(nn.Module):

    #generator model
    def __init__(self,nz,ngf=64,nc=3):
        super(generator,self).__init__()

        self.input = nn.Linear(nz, ngf*16*4*4)

        self.main = nn.Sequential(
            # input is Z, going into a convolution
            # nn.ConvTranspose2d( nz, ngf * 16, 4, 1, 0, bias=False),
            # nn.BatchNorm2d(ngf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ngf*16) x 4 x 4
            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ngf*8) x 8 x 8
            nn.ConvTranspose2d( ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ngf*4) x 16 x 16
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ngf*2) x 32 x 32
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ngf) x 64 x 64
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 128 x 128

        )

    def forward(self,input):
    	return self.main(self.input(input).view(-1, 4, 4))

class generator_v2(nn.Module):

    #generator model
    def __init__(self,nz,ngf=64,nc=3):
        super(generator_v2,self).__init__()

        self.main = nn.Sequential(
            # input is Z, going into a convolution
            Interpolate(size=(4, 4), mode='nearest'),
            nn.Conv2d(nz, ngf*16, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ngf*16) x 4 x 4
            Interpolate(size=(8, 8), mode='nearest'),
            nn.Conv2d(ngf*16, ngf*8, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ngf*8) x 8 x 8
            Interpolate(size=(16, 16), mode='nearest'),
            nn.Conv2d(ngf*8, ngf*4, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ngf*4) x 16 x 16
            Interpolate(size=(32, 32), mode='nearest'),
            nn.Conv2d(ngf*4, ngf*2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ngf*2) x 32 x 32
            Interpolate(size=(64, 64), mode='nearest'),
            nn.Conv2d(ngf*2, ngf, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ngf) x 64 x 64
            Interpolate(size=(128, 128), mode='nearest'),
            nn.Conv2d(ngf, nc, 3, 1, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 128 x 128

        )

    def forward(self,input):
    	return self.main(input)



class generator_big(nn.Module):

    #generator model
    def __init__(self,nz,ngf=64,nc=3):
        super(generator_big,self).__init__()

        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ngf*16) x 4 x 4
            nn.ConvTranspose2d(ngf * 16, ngf * 16, 3, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # ...
            nn.ConvTranspose2d(ngf * 16, ngf * 16, 3, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # ...
            nn.ConvTranspose2d(ngf * 16, ngf * 16, 3, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # ...
            nn.ConvTranspose2d(ngf * 16, ngf * 16, 3, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # ...
            nn.ConvTranspose2d(ngf * 16, ngf * 16, 3, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # ...
            nn.ConvTranspose2d(ngf * 16, ngf * 16, 3, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # ...
            nn.ConvTranspose2d(ngf * 16, ngf * 16, 3, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # ...
            nn.ConvTranspose2d(ngf * 16, ngf * 14, 3, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 14),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ngf*14) x 8 x 8
            nn.ConvTranspose2d( ngf * 14, ngf * 12, 3, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 12),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ngf*12) x 10 x 10
            nn.ConvTranspose2d( ngf * 12, ngf * 10, 3, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 10),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ngf*10) x 20 x 20
            nn.ConvTranspose2d( ngf * 10, ngf * 8, 3, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ngf * 8) x 22 x 22
            nn.ConvTranspose2d( ngf * 8, ngf * 6, 3, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 6),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ngf * 6) x 44 x 44
            nn.ConvTranspose2d( ngf * 6, ngf * 4, 3, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ngf * 4) x 46 x 46
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 3, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ngf * 2) x 40 x 40
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ngf) x 42 x 42
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 44 x 44

        )

    def forward(self,input):
    	return self.main(input)

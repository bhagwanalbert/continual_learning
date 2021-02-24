import torch
from torch import nn
class discriminator(nn.Module):
    def __init__(self,nc=3,ndf=64):
        super(discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 128 x 128
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 64 x 64
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 32 x 32
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 16 x 16
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 8 x 8
            nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*16) x 4 x 4
            #nn.Conv2d(ndf * 16, 1, 4, 1, 0, bias=False),
        )

        self.fc_dis = nn.Linear(ndf*16*4*4, 1)
        self.sig = nn.Sigmoid()

        self.ndf = ndf


    def forward(self, input):
        features = self.main(input)
        flat_features = features.contiguous().view(-1,self.ndf*16*4*4)

        source = self.sig(self.fc_dis(flat_features))

        source = source.view(source.shape[0])

        return source

class conditioned_discriminator_feat(nn.Module):
    def __init__(self,ndf=64,num_classes=50):
        super(conditioned_discriminator_feat, self).__init__()

        self.main = nn.Sequential(
            # input is (ndf*8) x 8 x 8
            nn.Conv2d(ndf * 8, ndf * 12, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ndf * 12),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*12) x 8 x 8
            nn.Conv2d(ndf * 12, ndf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*16) x 4 x 4
            nn.Conv2d(ndf * 16, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 2 x 2
            nn.Conv2d(ndf * 8, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 1 x 1
            # nn.Conv2d(ndf * 16, 1, 4, 1, 0, bias=False),
        )
        self.fc_dis = nn.Linear(ndf*4, 1)
        self.fc_aux = nn.Linear(ndf*4, num_classes)

        self.softmax = nn.Softmax(dim=1)
        self.sig = nn.Sigmoid()

        self.ndf = ndf

    def forward(self, input):
        features = self.main(input)
        flat_features = features.contiguous().view(-1,self.ndf*16*4*4)

        classes = self.softmax(self.fc_aux(flat_features))
        source = self.sig(self.fc_dis(flat_features))

        source = source.view(source.shape[0])

        return classes, source

class conditioned_discriminator(nn.Module):
    def __init__(self,nc=3,ndf=64,num_classes=10):
        super(conditioned_discriminator, self).__init__()

        self.main = nn.Sequential(
            # input is (nc) x 128 x 128
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 64 x 64
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 32 x 32
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 16 x 16
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 8 x 8
            nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*16) x 4 x 4
            # nn.Conv2d(ndf * 16, 1, 4, 1, 0, bias=False),
        )
        self.fc_dis = nn.Linear(ndf*16*4*4, 1)
        self.fc_aux = nn.Linear(ndf*16*4*4, num_classes)

        self.softmax = nn.Softmax(dim=1)
        self.sig = nn.Sigmoid()

        self.ndf = ndf

    def forward(self, input):
        features = self.main(input)
        flat_features = features.contiguous().view(-1,self.ndf*16*4*4)

        classes = self.softmax(self.fc_aux(flat_features))
        source = self.sig(self.fc_dis(flat_features))

        source = source.view(source.shape[0])

        return classes, source

class conditioned_discriminator_v2(nn.Module):
    def __init__(self,nc=3,ndf=64,num_classes=10):
        super(conditioned_discriminator_v2, self).__init__()

        self.main = nn.Sequential(
            # input is (nc) x 128 x 128
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 64 x 64
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 32 x 32
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 16 x 16
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 8 x 8
            nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*16) x 4 x 4
            # nn.Conv2d(ndf * 16, 1, 4, 1, 0, bias=False),
        )
        self.fc_dis = nn.Linear(ndf*16*4*4, 1)
        self.fc_aux = nn.Linear(ndf*16*4*4, num_classes)

        self.softmax = nn.Softmax(dim=1)
        self.sig = nn.Sigmoid()

        self.ndf = ndf

    def forward(self, input):
        features = self.main(input)
        flat_features = features.contiguous().view(-1,self.ndf*16*4*4)

        classes = self.softmax(self.fc_aux(flat_features))
        source = self.sig(self.fc_dis(flat_features))

        source = source.view(source.shape[0])

        return classes, source


if __name__ == "__main__":

    model = conditioned_discriminator()
    print(model)

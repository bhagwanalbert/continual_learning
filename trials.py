import torch
import torch.nn as nn

class dummy_net(nn.Module):
    def __init__(self):
        super(dummy_net, self).__init__()
    def forward(self, input):
        return input, [torch.zeros(1).to(input.device),torch.ones(1).to(input.device),2*torch.ones(1).to(input.device)], torch.zeros(4,4).to(input.device), torch.ones(10).to(input.device)

model = dummy_net()
model = nn.DataParallel(model, device_ids=[0,1,2,3,4])

a, [b,c,d], e, f = model(torch.randn(100,2,2).to("cuda:0"))

print(a)
print(b)
print(c)
print(d)
print(e)
print(f)

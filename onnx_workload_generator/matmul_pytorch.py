from torch import nn
import torch

class matmul_workload(nn.Module):
    expansion = 4
    #def __init__(self, in_planes=256):
    #def __init__(self, in_planes=512, planes=256):
    def __init__(self, in_planes=256, planes=64):
        super(matmul_workload, self).__init__()
        #self.matmul1 = nn.Conv2d(in_planes, planes)
        #self.matmul2 = nn.Conv2d(planes, planes)
        #self.matmul3 = nn.Conv2d(planes, in_planes)
        #self.relu1 = nn.ReLU()
        #self.relu2 = nn.ReLU()
        #self.relu3 = nn.ReLU()


    def forward(self, x):
        out = torch.matmul(x, x)
        #out = self.relu1(out)
        #out = torch.matmul(out, x)
        #out = self.relu2(out)
        #out = torch.matmul(out, x)
        #out = out+x
        #out = self.relu3(out)
        return out

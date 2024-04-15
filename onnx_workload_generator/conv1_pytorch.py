from torch import nn

class conv1_workload(nn.Module):
        expansion = 4
        #def __init__(self, in_planes=256):
        #def __init__(self, in_planes=512, planes=256):
        def __init__(self, in_planes=256, planes=64):
            super(conv1_workload, self).__init__()

            # self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=1,bias=False)  # initially trying out a single convolution
            # self.conv2 = nn.Conv2d(in_planes, in_planes, kernel_size=3,bias=False,stride=2)
            # self.conv3 = nn.Conv2d(in_planes, self.expansion *in_planes, bias=False,kernel_size=1)
            
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
            self.conv3 = nn.Conv2d(planes, in_planes, kernel_size=1, bias=False)
            self.relu1 = nn.ReLU()
            self.relu2 = nn.ReLU()
            self.relu3 = nn.ReLU()
            
            
        def forward(self, x):
            
            out = self.conv1(x)
            out = self.relu1(out)
            out = self.conv2(out)
            out = self.relu2(out)
            out = self.conv3(out)
            out = out+x
            out = self.relu3(out)
            return out
    
        

    
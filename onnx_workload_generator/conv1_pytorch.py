from torch import nn

class conv1_workload(nn.Module):
        expansion = 4
        # def __init__(self, in_planes=256, planes=64):
        def __init__(self, in_planes=512, planes=256):
            super(conv1_workload, self).__init__()

            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,bias=False)  # initially trying out a single convolution
            #self.conv2 = nn.Conv2d(planes, planes, kernel_size=1,bias=False)
            #self.conv3 = nn.Conv2d(planes, self.expansion *planes, kernel_size=1,bias=False)
            #self.relu1 = ReLU()
            #self.relu2 = ReLU()
            #self.relu3 = ReLU()
            #self.pooling = nn.MaxPool2d(3, stride=3)

        def forward(self, x):
            out = self.conv1(x)
            #out = self.relu1(out)
            #out = self.conv2(out)
            # out = self.relu2(out)
            # out = self.conv3(out)
            # out = out+x
            # out = self.relu3(out)

            #out = self.pooling(out)
            return out
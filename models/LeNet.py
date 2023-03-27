"""
conv1 (1,32,32) -> (6,28,28)
32 -> 28?
27 = (32+0-5)/1 s=1 p=0 k=5
 
subsampling1 (6,28,28) -> (6,14,14) 13 = (28+2p-k)/s k=2 s=2 avgpooling
 
 
conv2 (6,14,14) -> (16,10,10)
9 = 14+2p-5/s k=5 p=0 s=1

subsampling2 (16,10,10) -> (16,5,5)
4 = 10+2p-k/s 
k=2 s=2 avgpooling

fc1 (in=16*5*5,out=120)
fc2 (in=120,out=84)
gaussian(84,10)
"""

import torch
import torch.nn as nn

class Subsampling_block(nn.Module):
    def __init__(self, kernel_size=2, stride=2):
        super(Subsampling_block, self).__init__()
        self.avgpool = nn.AvgPool2d(kernel_size, stride)
        #self.batchnorm = nn.BatchNorm2d(out_channels)#논문이 나왔을 당시엔 없었지만 성능 향상을 위해 추가
        
    def forward(self, x):
        return self.avgpool(x)
    
class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, padding=0):
        super(conv_block,self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        return self.relu(self.conv(x))
    
class LeNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(LeNet, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = conv_block(in_channels, 6)
        self.subsample = Subsampling_block()
        self.conv2 = conv_block(6, 16)
        self.conv3 = conv_block(16,120, kernel_size=5, stride=1)
        #(16,5,5) => (120,1,1) -> 1 = (5 + 2p -f / s) +1 -> f=5
        self.fc1 = nn.Linear(in_features=120, out_features=84)
        self.fc2 = nn.Linear(in_features=84, out_features=10)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.subsample(x)
        x = self.conv2(x)
        x = self.subsample(x)
        x = self.conv3(x)
        x = x.reshape(x.shape[0], -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x

if __name__ == '__main__':
    x = torch.randn(1, 1, 32, 32)
    model=LeNet()
    print(model(x).shape)

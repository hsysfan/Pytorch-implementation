"""
이렇듯 1x1 컨볼루션에 필터의 갯수를 줄여 연산량을 획기적으로 감소할 수 있습니다. 
이것이 바로 dimension reductions의 핵심입니다. 
이렇게 1x1 컨볼루션에서 필터 개수를 줄인 뒤 다시 키우는 구조를 BottleNeck 이라고 부르기도 합니다.

(3, 224, 224) -> (64, 112, 112) 
Step 1 : (224 + 2 * x - 7) / 2 + 1 = 112 =>(217 + 2 * 3) = 223 ==> x = 3

"""

# class InceptionModuleV1(nn.Module):
#     def __init__(self, in_channels, out_1x1, out_3x3, out_5x5, pool):
#         super(InceptionModuleV1, self).__init__()
#         self.conv1x1 = conv_block(in_channels, out_1x1, kernel_size=1, padding='same')
#         self.conv3x3 = conv_block(in_channels, out_3x3, kernel_size=3, padding='same')
#         self.conv5x5 = conv_block(in_channels, out_5x5, kernel_size=5, padding='same')
#         self.pool = nn.Sequential(
#             nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
#             conv_block(in_channels, pool, kernel_size=1, padding='same')
#         )

# class InceptionModuleV2(nn.Module):
#     def __init__(self,in_channels, out_1x1, out_3x3_reduce, out_3x3, out_5x5_reduce, out_5x5, pool):
#         super(InceptionModuleV2, self).__init__()
#         self.conv1x1 = conv_block(in_channels, out_1x1, kernel_size=1, padding='same')
#         self.conv3x3 = nn.Sequential(
#             conv_block(in_channels, out_3x3_reduce, kernel_size=1),
#             conv_block(out_3x3_reduce, out_3x3, kernel_size=3)
#         )
#         self.conv5x5 = nn.Sequential(
#             conv_block(in_channels, out_5x5_reduce, kernel_size=1, padding='same'),
#             conv_block(out_5x5_reduce,out_5x5,kernel_size=5, padding='same')
#         )
#         self.pool = nn.Sequential(
#             nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
#             conv_block(in_channels, pool, kernel_size=1, padding='same')
#         )

import torch
import torch.nn as nn

class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(conv_block, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, **kwargs)
        self.relu = nn.ReLU(inplace=True)
        self.batchnorm = nn.BatchNorm2d(out_channels)#논문이 나왔을 당시엔 없었지만 성능 향상을 위해 추가
        
    def forward(self, x):
        return self.relu(self.batchnorm(self.conv(x)))

class Inception_block(nn.Module):
    def __init__(self,in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out1x1pool):
        super(Inception_block, self).__init__()
        
        self.branch1 = conv_block(in_channels, out_1x1, kernel_size=1)
        
        self.branch2 = nn.Sequential(
            conv_block(in_channels, red_3x3, kernel_size=1),
            conv_block(red_3x3, out_3x3, kernel_size=3, padding=1)
        )
        
        self.branch3 = nn.Sequential(
            conv_block(in_channels, red_5x5, kernel_size=1),
            conv_block(red_5x5, out_5x5, kernel_size=5, padding=2)
        )
        
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            conv_block(in_channels, out1x1pool, kernel_size=1)
        )
    

    def forward(self, x):
        # N x filters x 28 x 28
        return torch.cat([self.branch1(x),self.branch2(x),self.branch3(x),self.branch4(x)], 1)

class GoogLeNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000):
        super(GoogLeNet, self).__init__()
        self.conv1 = conv_block(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3)
        #(3, 224, 224) => (64, 112, 112) -> (64, (224 + 2 * x - 7) / 2 + 1,  (224 + 2 * x - 7) / 2 + 1)
        #(3, 224, 224) => (64, 112, 112) -> (64, (217 + 2 * x) / 2 + 1,  (217 + 2 * x) / 2 + 1)
        #(3, 224, 224) => (64, 112, 112) -> (217 + 2 * x) / 2 + 1 = 112 -> 217 + 2 * x = 222
        #padding 2,3 으로 주는 경우가 있음 딱 떨어지는 정수값이 아님
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        #(64, 112, 112) => (64, 56, 56) -> (64, (112 + 2 * x - 3) / 2 + 1, (112 + 2 * x - 3) / 2 + 1)
        #(64, 112, 112) => (64, 56, 56) -> (64, (109 + 2 * x) / 2 + 1, (109 + 2 * x) / 2 + 1)
        #(64, 112, 112) => (64, 56, 56) -> (109 + 2x) / 2 + 1 = 56 -> 109 + 2x = 110
        self.conv2 = conv_block(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1)
        #(64, 56, 56) => (192, 56, 56) -> (56 + 2x - 3)/1 + 1 = 56 -> 53 + 2x = 55 -> x = 1
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        #(192, 56, 56) => (192, 28, 28)
        self.inception3a = Inception_block(in_channels=192, out_1x1=64, red_3x3=96, out_3x3=128, red_5x5=16, out_5x5=32, out1x1pool=32)
        #(192, 28, 28) => (256, 28, 28)
        self.inception3b = Inception_block(in_channels=256, out_1x1=128, red_3x3=128, out_3x3=192, red_5x5=32, out_5x5=96, out1x1pool=64)
        #(256, 28, 28) => (480, 28, 28)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        #(480, 28, 28) => (480, (28 - 3) / 2 + 1, (28 - 3) / 2 + 1)
        #(480, 28, 28) => (480, 14, 14)
        self.inception4a = Inception_block(in_channels=480, out_1x1=192, red_3x3=96, out_3x3=208, red_5x5=16, out_5x5=48, out1x1pool=64)
        self.inception4b = Inception_block(in_channels=512, out_1x1=160, red_3x3=112, out_3x3=224, red_5x5=24, out_5x5=64, out1x1pool=64)
        self.inception4c = Inception_block(in_channels=512, out_1x1=128, red_3x3=128, out_3x3=256, red_5x5=24, out_5x5=64, out1x1pool=64)
        self.inception4d = Inception_block(in_channels=512, out_1x1=112, red_3x3=144, out_3x3=288, red_5x5=32, out_5x5=64, out1x1pool=64)
        self.inception4e = Inception_block(in_channels=528, out_1x1=256, red_3x3=160, out_3x3=320, red_5x5=32, out_5x5=128, out1x1pool=128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        #(832, 14, 14) => (832, 7, 7)
        self.inception5a = Inception_block(in_channels=832, out_1x1=256, red_3x3=160, out_3x3=320, red_5x5=32, out_5x5=128, out1x1pool=128)
        self.inception5b = Inception_block(in_channels=832, out_1x1=384, red_3x3=192, out_3x3=384, red_5x5=48, out_5x5=128, out1x1pool=128)
        #(1024, 7, 7)
        self.avgpool1 = nn.AvgPool2d(kernel_size=7, stride=1)
        #(1024, 7, 7) => (1024, 1, 1) -> (7 - 3 + 2x)/ 1 + 1 = 1 - > 4 = 0?
        self.dropout = nn.Dropout(p=0.4)
        self.fc1 = nn.Linear(1024,1000)
        
    def forward(self,x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)
        
        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)
        
        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.avgpool1(x)
        x = x.reshape(x.shape[0], -1)
        #fc 연결전에 flatten 작업을 해줘야함, (1,1024,1,1) -> 1,1024 || (1,1024,3,3) -> 1, 1024*3*3
        x = self.dropout(x)
        x = self.fc1(x)
        
        return x
        
        
if __name__ == '__main__':
    x = torch.randn(1,3,224,224)
    model = GoogLeNet()
    print(model(x).shape)
    
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# import torchsummary
# model = GoogLeNet()
# model.to(device)
# torchsummary.summary(model, (3,224,224),device)

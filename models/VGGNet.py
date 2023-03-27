import torch
import torch.nn as nn

VGG_types={
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

#Then flatten and 4096x4096x1000 Linear Layers

class VGG_Net(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000):
        super(VGG_Net, self).__init__()
        self.in_channels = in_channels
        self.conv_layers = self.create_conv_layers(VGG_types['VGG16'])
    
        self.fcs = nn.Sequential(
            nn.Linear(in_features=(512*7*7),out_features=4096), # 7 이라는 값은 224(Input_size) / 2**5(MaxPool의 갯수) conv가 계속 같아서 아웃, 인 채널만 변하고 이미지 사이즈는 MaxPool에서만 줄어드는 듯?
            nn.ReLU(),
            nn.Dropout(p=0.5), #일반적으로 사용하기때문에 사용함
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096,num_classes)
            )
    
    def create_conv_layers(self, architecture):
        layers=[]
        in_channels = self.in_channels
        
        for x in architecture:
            if type(x) == int:
                out_channels = x
                
                layers += [nn.Conv2d(in_channels=in_channels,out_channels=out_channels, 
                                    kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(x),#논문엔 없지만 성능 향상을 위해 추가 
                                    nn.ReLU()]
                in_channels = x
            elif x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            
        return nn.Sequential(*layers)
    
    def forward(self,x):
        x = self.conv_layers(x)
        x = x.reshape(x.shape[0],-1)
        x = self.fcs(x)    
        
        return x
    
if __name__=='__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model=VGG_Net(3,1000).to(device)
    x = torch.randn(1,3,224,224).to(device)
    print(model(x).shape)

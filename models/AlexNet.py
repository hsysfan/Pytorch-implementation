"""
AlexNet Architecture

Input Width, Height, channel = 227 x 227 x3
    Conv1:
        Filter = 11x11 
        Stride = 4
        Activate = ReLU
        MaxPooling => Filter = 3x3, Stride = 2
        
    Conv2:
        Filter = 5x5
        Stride = 1
        Padding = 2
        Activate = ReLU
        MaxPooling => Filter = 3x3, Stride = 2
    
    Conv3:
        Filter = 3x3
        Strider = 1
        Padding = 1
        Activate ReLU
        
    Conv4:
        Filter = 3x3
        Strider = 1
        Padding = 1
        Activate ReLU
        
    Conv5:
        Filter = 3x3
        Strider = 1
        Padding = 1
        Activate ReLU
        MaxPooling => Filter 3x3, Stride = 2
    
    Dropout:
    rate = 0.5

"""

import torch
import torch.nn as nn

class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
      
        self.layers = nn.Sequential(
          
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=0),
            #Calcultation Layer => (3, 227, 227) -> (??, (227 + 0 - 11) / 4 + 1, (227 + 0 - 11) / 4 + 1)
            #Calcultation Layer => (3, 227, 227) -> (??, (216) / 4 + 1, (216) / 4 + 1)
            #Calcultation Layer => (3, 227, 227) -> (96, 54 + 1, 54 + 1)
            #Output_size = (96, 55, 55)
            nn.ReLU(),
            #Output_size = (96, 55, 55)
            nn.MaxPool2d(kernel_size=3, stride=2),
            #Calculation Layer => (96, 55, 55) -> (96, 27, 27)
            #Output_size = (96, 27, 27)
            nn.Conv2d(in_channels=96,out_channels=256,kernel_size=5, stride=1, padding=2),
            #Calculation Layer => (96, 27, 27) -> (256, (27 + 2 * 2 - 5) / 1 + 1, (27 + 2 * 2 - 5) / 1 + 1)
            #Calculation Layer => (96, 27, 27) -> (256, (26) / 1 + 1, (26) / 1 + 1)
            #Calculation Layer => (96, 27, 27) -> (256, 27, 27)
            #Output_size = (256, 27, 27)
            nn.ReLU(),
            #Output_size = (256, 27, 27)
            nn.MaxPool2d(kernel_size=3, stride=2),
            #Calculation Layer => (256, 27, 27) -> (256, 27 / 2, 27 / 2)
            #Calculation Layer => (256, 27, 27) -> (256, 13.xx, 13.xx)
            #Output_size = (256, 13, 13)
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1),
            #Calculation Layer => (256, 13, 13) -> (384, (13 + 2 * 1 - 3) / 1 + 1, (13 + 2 * 1 - 3) / 1 + 1)
            #Calculation Layer => (256, 13, 13) -> (384, (12) / 1 + 1, (12) / 1 + 1)
            #Calculation Layer => (256, 13, 13) -> (384, 13, 13)
            #Output_size = (384, 13, 13)
            nn.ReLU(),
            #Output_size = (384, 13, 13)
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1),
            #Calculation Layer => (384, 13, 13) -> (384, (13 + 2 * 1 - 3) / 1 + 1, (13 + 2 * 1 - 3) / 1 + 1)
            #Calculation Layer => (384, 13, 13) -> (384, (12) / 1 + 1, (12) / 1 + 1)
            #Calculation Layer => (384, 13, 13) -> (384, 13, 13)
            #Output_size = (384, 13, 13)
            nn.ReLU(),
            #Output_size = (384, 13, 13)
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),
            #Calculation Layer => (384, 13, 13) -> (256, (13 + 2 * 1 - 3) / 1 + 1, (13 + 2 * 1 - 3) / 1 + 1)
            #Calculation Layer => (384, 13, 13) -> (256, (12) / 1 + 1, (12) / 1 + 1)
            #Calculation Layer => (384, 13, 13) -> (256, 13, 13)
            nn.ReLU(),
            #Output_size = (256, 13, 13)
            nn.MaxPool2d(kernel_size=3, stride=2),
            #Output_size = (256, 6, 6)
            
        )
      
        self.classifier = nn.Sequential(
          nn.Dropout(p=0.5),#생략 가능
          #Input_size = (256, 6, 6)
          nn.Linear(in_features=256*6*6,out_features=4096),
          nn.ReLU(inplace=True),
          nn.Linear(in_features=4096,out_features=4096),
          nn.ReLU(),
          nn.Linear(in_features=4096,out_features=num_classes)          
        )

    def init_bias_weights(self):
        for layer in self.layers:
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, mean=0, std=0.01)
                nn.init.constant_(layer.bias, 0)
        nn.init.constant_(self.net[4].bias, 1)
        nn.init.constant_(self.net[10].bias, 1)
        nn.init.constant_(self.net[12].bias, 1)
        
    def forward(self, x):
        x = self.layers(x)
        x = x.view(-1, 256 * 6 * 6)
        x = self.classifier(x)
        return x   

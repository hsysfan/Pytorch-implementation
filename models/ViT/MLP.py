import torch.nn as nn


class Multi_Layer_Perceptron(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, p=0.0):
        super(Multi_Layer_Perceptron, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        # Linear 레이어를 통해 차원의 크기를 늘려서 더 세세하게 표현함
        self.activation = nn.GELU()
        # GELU 활성화 함수를 사용
        self.fc2 = nn.Linear(hidden_features, out_features)
        # Linear 레이어를 통해 차원의 크기를 줄여서 out_features 에 맞춰줌
        self.dropout = nn.Dropout(p)
        # Dropout

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)

        return x
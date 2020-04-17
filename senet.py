import torch
from torch import nn

class SEModule(nn.Module):
    def __init__(self, in_channels, reduction):
        super().__init__()
        mid_channels = in_channels // reduction
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc1 = nn.Linear(in_channels, mid_channels)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(mid_channels, in_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        n_batches, n_channels, h, w = x.size()
        out = self.avg_pool(x).view(n_batches, n_channels)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = nn.Sigmoid()(out).view(n_batches, n_channels, 1, 1)
        return x * out


x = torch.randn(100, 64, 32, 32)  #ダミー入力
se_module = SEModule(64,)
y = se_module(x)
print(y.size())
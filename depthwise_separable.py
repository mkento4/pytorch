import torch
from torch import nn

class depthwise_separable_conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        print('depth',out.size())
        out = self.pointwise(out)
        return out



x = torch.randn(100, 64, 32, 32)  #ダミー入力
model = depthwise_separable_conv(64, 128)
y = model(x)
print(y.size())
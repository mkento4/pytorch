import torch
from torch import nn


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, groups=1):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )
    
    def forward(self, x):
        out = self.model(x)
        return out

class InvertedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super().__init__()
        self.stride = stride

        hidden_dim = int(round(in_channels * expand_ratio))

        self.use_residual = self.stride == 1 and in_channels == out_channels

        layers = []
        if expand_ratio != 1:
            #point wise
            layers.append(ConvBNReLU(in_channels, hidden_dim, stride=stride, groups=hidden_dim))
        
        layers.extend([
            #depth wise 
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            #point wise linear
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels)
        ])

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_residual:
            return x + self.model(x)
        else:
            return self.model(x)

class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.0, inverted_residual_setting=None, round_nearest=8, block=None):
        super().__init__()

        if block is None:
            block = InvertedResidualBlock
        
        in_channels = 32
        last_channels = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]
        
        in_channels = _make_divisible(in_channels * width_mult, round_nearest)

        self.last_channels = _make_divisible(last_channels * max(1.0, width_mult), round_nearest)
        
        features = [ConvBNReLU(3, in_channels, stride=2)]
        
        for t, c, n, s in inverted_residual_setting:
            out_channels = _make_divisible(c * width_mult, round_nearest)

            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(in_channels, out_channels, stride, expand_ratio=t))
                in_channels = out_channels

        features.append(ConvBNReLU(in_channels, self.last_channels, kernel_size=1))

        self.features = nn.Sequential(*features)

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channels, num_classes),
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.zeros_(m.bias)
        
    def forward(self, x):
        out = self.features(x)
        out = self.avg_pool(out).view(out.size(0), -1)
        out = self.classifier(out)

        return out
import torch
from torch import nn

import random
import math
import time
import pandas as pd
import numpy as np
from PIL import Image

import torch
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import transforms

class Generator(nn.Module):
    def __init__(self, z_dim, img_size):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=z_dim, out_channels=img_size* 8, kernel_size=4, stride=1),
            nn.BatchNorm2d(img_size* 8),
            nn.ReLU(inplace=True)
        )

        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=img_size* 8, out_channels=img_size* 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(img_size* 4),
            nn.ReLU()
        )

        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=img_size* 4, out_channels=img_size* 2,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(img_size * 2),
            nn.ReLU(inplace=True))

        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels = img_size * 2, out_channels = img_size,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(img_size),
            nn.ReLU(inplace=True))

        self.last = nn.Sequential(
            nn.ConvTranspose2d(in_channels=img_size, out_channels=1, kernel_size=4,
                               stride=2, padding=1),
            nn.Tanh()
            )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.last(out)

        return out


class Discriminator(nn.Module):
    def __init__(self, z_dim=20, img_size=64):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=img_size, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(img_size),
            nn.ReLU(inplace=True)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(img_size, img_size* 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=img_size* 2, out_channels=img_size* 4, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=img_size* 4, out_channels=img_size* 8, kernel_size=4,
                      stride=2, padding=1),
            nn.ReLU(inplace=True)
            )

        self.last = nn.Conv2d(in_channels=img_size * 8, out_channels=1, kernel_size=4, stride=1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.last(out)

        return out

g = Generator(z_dim=20, img_size=64)
d = Discriminator(z_dim=20, img_size=64)

input_z = torch.randn(1, 20)
input_z = input_z.view(input_z.size(0), input_z.size(1), 1, 1)
fake_imgs = g(input_z)

d_out = d(fake_imgs)

print(nn.Sigmoid()(d_out))
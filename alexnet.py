import torch
from torch import nn
import matplotlib.pyplot as plt
import cv2
import numpy as np
import seaborn as sns
class AlexNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # 特徴量抽出
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        # 分類
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
    
    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), 256 * 6 * 6)
        out = self.classifier(out)

        return out




img = cv2.imread('hidden_layer/pizza.jpg')
x = torch.from_numpy(img.astype(np.float32)).clone()
x = x.unsqueeze(0).permute(0, 3, 1, 2)

model = AlexNet(num_classes=1000).features

y = model(x)

plt.imshow(model[0].weight[0][0].detach().numpy())
plt.savefig('alex.png')
plt.close()
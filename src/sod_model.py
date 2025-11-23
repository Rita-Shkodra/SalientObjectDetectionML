import torch
import torch.nn as nn
import torch.nn.functional as F

class SODModel(nn.Module):
    def __init__(self):
        super(SODModel, self).__init__()
        self.enc1 = nn.Sequential(
          nn.Conv2d(3, 32, kernel_size=3, padding=1),
          nn.ReLU(inplace=True),
          nn.MaxPool2d(2)
        )
        self.enc2 = nn.Sequential(
          nn.Conv2d(32, 64, kernel_size=3, padding=1),
          nn.ReLU(inplace=True),
          nn.MaxPool2d(2)
        )
        self.enc3 = nn.Sequential(
          nn.Conv2d(64, 128, kernel_size=3, padding=1),
          nn.ReLU(inplace=True),
          nn.MaxPool2d(2)
        )

    def forward(self, x):
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        return x

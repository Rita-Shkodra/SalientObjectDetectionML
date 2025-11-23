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
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )
        
      
        self.out_conv = nn.Conv2d(16, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.dec3(x)
        x = self.dec2(x)
        x = self.dec1(x)
        x = self.out_conv(x)
        x = self.sigmoid(x)
        return x

import torch
import torch.nn as nn
import torch.nn.functional as F

class SODModel(nn.Module):
    def __init__(self):
        super(SODModel, self).__init__()

        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        self.dec3_up = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec3 = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.dec2_up = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.dec1_up = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.BatchNorm2d(19),
            nn.ReLU(inplace=True),
            nn.Conv2d(19, 16, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.out_conv = nn.Conv2d(16, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x0 = x
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)

        d3 = self.dec3_up(x3)
        d3 = torch.cat([d3, x2], dim=1)
        d3 = self.dec3(d3)

        d2 = self.dec2_up(d3)
        d2 = torch.cat([d2, x1], dim=1)
        d2 = self.dec2(d2)

        d1 = self.dec1_up(d2)
        d1 = torch.cat([d1, x0], dim=1)
        d1 = self.dec1(d1)

        out = self.out_conv(d1)
        return self.sigmoid(out)

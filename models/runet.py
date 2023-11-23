import torch
import torch.nn as nn
import torch.nn.functional as F


# Based on https://github.com/cerniello/Super_Resolution_DNN/blob/master/NN_Final_Project_3_Moschettieri.ipynb

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding='same', r=False):
        super().__init__()
        self.path_a = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels)
        )
        self.path_b = nn.Conv2d(in_channels, out_channels, (1, 1), stride, padding) if r else nn.Identity()

    def forward(self, x):
        return self.path_a(x) + self.path_b(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding='same', upscale_factor=2):
        super().__init__()
        self.upscale = nn.PixelShuffle(upscale_factor)
        self.path = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding),
            nn.ReLU(),
            nn.ReLU()
        )

    def forward(self, x, skip):
        return self.path(torch.concatenate([self.upscale(x), skip], dim=1))


def bottleneck(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding='same'):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.ReLU()
    )


class RUNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, (7, 7), stride=(1, 1), padding='same'),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d((2, 2)),
            DownBlock(64, 64),
            DownBlock(64, 64),
            DownBlock(64, 64),
            DownBlock(64, 128, r=True)
        )
        self.down3 = nn.Sequential(
            nn.MaxPool2d((2, 2), stride=(2, 2)),
            DownBlock(128, 128),
            DownBlock(128, 128),
            DownBlock(128, 128),
            DownBlock(128, 256, r=True)
        )
        self.down4 = nn.Sequential(
            nn.MaxPool2d((2, 2)),
            DownBlock(256, 256),
            DownBlock(256, 256),
            DownBlock(256, 256),
            DownBlock(256, 256),
            DownBlock(256, 256),
            DownBlock(256, 512, r=True)
        )
        self.down5 = nn.Sequential(
            nn.MaxPool2d((2, 2)),
            DownBlock(512, 512),
            DownBlock(512, 512),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.bn = nn.Sequential(
            bottleneck(512, 1024),
            bottleneck(1024, 512)
        )
        self.up1 = UpBlock(1024, 512, upscale_factor=1)
        self.up2 = UpBlock(640, 384, upscale_factor=2)
        self.up3 = UpBlock(352, 256, upscale_factor=2)
        self.up4 = UpBlock(192, 96, upscale_factor=2)
        self.up5_upscale = nn.PixelShuffle(2)
        self.up5 = nn.Sequential(
            nn.Conv2d(88, 99, (3, 3), stride=(1, 1), padding='same'),
            nn.ReLU(),
            nn.Conv2d(99, 99, (3, 3), stride=(1, 1), padding='same'),
            nn.ReLU()
        )
        self.out = nn.Conv2d(99, out_channels, (1, 1), padding='same')

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='bilinear')
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        bn = self.bn(d5)
        u1 = self.up1(bn, d5)
        u2 = self.up2(u1, d4)
        u3 = self.up3(u2, d3)
        u4 = self.up4(u3, d2)
        u5 = self.up5(torch.concatenate([self.up5_upscale(u4), d1], dim=1))
        return self.out(u5)

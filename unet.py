import torch
import torch.nn as nn
import torch.nn.functional as F


def contracting_block(in_channels: int, out_channels: int):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=0),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=0),
        nn.ReLU(),
    )


def expanding_block(in_channels: int, out_channels: int):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=2),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=2),
        nn.ReLU(),
    )


class UNet(nn.Module):
    def __init__(self, channels: int, num_classes: int):
        super(UNet, self).__init__()
        self.channels = channels
        self.num_classes = num_classes

        self.lvl_a0 = contracting_block(self.channels, 64)
        self.lvl_a0_mp = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.lvl_a1 = contracting_block(64, 128)
        self.lvl_a1_mp = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.lvl_a2 = contracting_block(128, 256)
        self.lvl_a2_mp = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.lvl_a3 = contracting_block(256, 512)
        self.lvl_a3_mp = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.lvl_c = contracting_block(512, 1024)
        self.lvl_c_conv = nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=2)
        self.lvl_b3 = expanding_block(1024, 512)
        self.lvl_b3_conv = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=2)
        self.lvl_b2 = expanding_block(512, 256)
        self.lvl_b2_conv = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=2)
        self.lvl_b1 = expanding_block(256, 128)
        self.lvl_b1_conv = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=2)
        self.lvl_b0 = expanding_block(128, 64)
        self.conv_cls = nn.Conv2d(64, self.num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='bilinear')
        l0 = self.lvl_a0(x)
        l1 = self.lvl_a1(self.lvl_a0_mp(l0))
        l2 = self.lvl_a2(self.lvl_a1_mp(l1))
        l3 = self.lvl_a3(self.lvl_a2_mp(l2))
        lc = self.lvl_c(self.lvl_a3_mp(l3))
        lc = self.lvl_c_conv(F.interpolate(lc, scale_factor=2, mode='bilinear'))
        l3 = l3[:, :, :lc.shape[2], :lc.shape[3]]
        l3 = self.lvl_b3(torch.concatenate([l3, lc], dim=1))
        l3 = self.lvl_b3_conv(F.interpolate(l3, scale_factor=2, mode='bilinear'))
        l2 = l2[:, :, :l3.shape[2], :l3.shape[3]]
        l2 = self.lvl_b2(torch.cat([l2, l3], dim=1))
        l2 = self.lvl_b2_conv(F.interpolate(l2, scale_factor=2, mode='bilinear'))
        l1 = l1[:, :, :l2.shape[2], :l2.shape[3]]
        l1 = self.lvl_b1(torch.cat([l1, l2], dim=1))
        l1 = self.lvl_b1_conv(F.interpolate(l1, scale_factor=2, mode='bilinear'))
        l0 = l0[:, :, :l1.shape[2], :l1.shape[3]]
        l0 = self.lvl_b0(torch.cat([l0, l1], dim=1))
        l0 = self.conv_cls(l0)
        return l0

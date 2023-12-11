import torch.nn as nn

from .runet import RUNet


class RUNetFC(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.seq = nn.Sequential(
            RUNet(in_channels, 1024),
            nn.Conv2d(1024, 1024, (1, 1), stride=(1, 1), padding='same'),
            nn.Conv2d(1024, out_channels, (1, 1), stride=(1, 1), padding='same')
        )

    def forward(self, x):
        return self.seq(x)

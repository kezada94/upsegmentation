import torch.nn as nn

from .runet import RUNet


class RUNetFC(nn.Module):
    def __init__(self, in_channels, out_channels, nfc=3):
        super().__init__()
        self.seq = nn.Sequential(
            RUNet(in_channels, out_channels),
            *[
                nn.Conv2d(out_channels, out_channels, (1, 1), stride=(1, 1), padding='same')
                for _ in range(nfc)
            ]
        )

    def forward(self, x):
        return self.seq(x)

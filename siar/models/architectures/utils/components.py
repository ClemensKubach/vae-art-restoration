import torch
from torch import nn
from torch.nn import Module


class ResBlock(Module):
    def __init__(self, in_channels, out_channels):
        nn.Module.__init__(self)

        self.conv_block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, in_channels, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv_block(x)


class ResBlockVDVAE(Module):
    def __init__(self, in_channels: int, hid_channels: int, out_channels: int, residual: bool = False):
        super().__init__()

        self.residual = residual
        self.conv_block = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(in_channels, hid_channels, kernel_size=1, stride=1, padding=0),
            nn.GELU(),
            nn.Conv2d(hid_channels, hid_channels, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(hid_channels, hid_channels, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(hid_channels, out_channels, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x: torch.tensor) -> torch.Tensor:
        out = x + self.conv_block(x) if self.residual else self.conv_block(x)
        return out


class TripleRes(Module):
    def __init__(self, in_channels, hid_channels, out_channels):
        super().__init__()

        self.triple_res = nn.Sequential(
            ResBlockVDVAE(in_channels, hid_channels, out_channels, residual=True),
            ResBlockVDVAE(in_channels, hid_channels, out_channels, residual=True),
            ResBlockVDVAE(in_channels, hid_channels, out_channels, residual=True),
        )

    def forward(self, x: torch.tensor) -> torch.Tensor:
        x = self.triple_res(x)
        return x

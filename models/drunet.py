import torch
import torch.nn as nn
from models.resblock import ResBlocks

class DRUNet(nn.Module):
    """Deep Residual U-Net model."""

    def __init__(self, nb_channels, depth, color):
        """Initialization."""
        super(DRUNet, self).__init__()

        if color: img_channels = 3
        else: img_channels = 1
        self.resblocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.conv_first = nn.Conv2d(img_channels, nb_channels, kernel_size=3, padding=1, bias=False)
        self.conv_last = nn.Conv2d(nb_channels, img_channels, kernel_size=3, padding=1, bias=False)

        for k in range(3):
            self.resblocks.append(ResBlocks(depth, nb_channels*2**k))
            self.downsamples.append(nn.Conv2d(nb_channels*2**k, nb_channels*2**(k+1), kernel_size=2, stride=2, bias=False))
            self.upsamples.append(nn.ConvTranspose2d(nb_channels*2**(3-k), nb_channels*2**(3-k-1), kernel_size=2, stride=2, bias=False))

        for k in range(4):
            self.resblocks.append(ResBlocks(depth, nb_channels*2**(3-k)))

        self.gamma = torch.nn.Parameter(torch.tensor(0.01))
        self.alpha = torch.nn.Parameter(torch.tensor(5.))

    def forward(self, x):
        """Forward pass."""
        x1 = self.conv_first(x)
        x2 = self.downsamples[0](self.resblocks[0](x1))
        x3 = self.downsamples[1](self.resblocks[1](x2))
        x4 = self.downsamples[2](self.resblocks[2](x3))
        x = self.upsamples[0](self.resblocks[3](x4) + x4)
        x = self.upsamples[1](self.resblocks[4](x) + x3)
        x = self.upsamples[2](self.resblocks[5](x) + x2)
        x = self.conv_last(self.resblocks[6](x) + x1)
        return x
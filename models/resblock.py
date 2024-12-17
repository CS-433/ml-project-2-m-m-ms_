import torch.nn as nn

class ResBlock(nn.Module):
    """Residual block."""

    def __init__(self, nb_channels):
        """Initialization."""
        super(ResBlock, self).__init__()

        self.network = nn.ModuleList([nn.Conv2d(nb_channels, nb_channels, kernel_size=3, padding=1, bias=False), \
                            nn.ReLU(inplace=True), nn.Conv2d(nb_channels, nb_channels, kernel_size=3, padding=1, bias=False)])
        self.network = nn.Sequential(*self.network)

    def forward(self, x):
        """Forward pass."""
        return x + self.network(x)
    

class ResBlocks(nn.Module):
    """Residual blocks."""

    def __init__(self, depth, nb_channels):
        super(ResBlocks, self).__init__()

        self.network = nn.ModuleList([ResBlock(nb_channels) for _ in range(depth)])

    def forward(self, x):
        """Forward pass."""
        for layer in self.network:
            x = layer(x)
        return x
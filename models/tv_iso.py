import torch
import torch.nn.functional as F 

class TV_ISO():
    """Total variation isotropic operator."""

    def __init__(self, device):
        """Initialization."""

        self.filters1 = torch.Tensor([[[[1., -1.]]]]).double().to(device)
        self.filters2 = torch.Tensor([[[[1.], [-1.]]]]).double().to(device)

    def L(self, x, h): 
        """Forward finite difference operator."""
        conv_1 = F.conv2d(F.pad(x, (1, 1, 1, 0)), self.filters1)
        conv_2 = F.conv2d(F.pad(x, (1, 0, 1, 1)), self.filters2)

        L1_1 = F.pad(conv_1, (0, 1, 0, 1))
        L2_1 = F.pad(conv_2, (0, 1, 0, 1))

        Lx = torch.cat((L1_1, L2_1), dim=1) 

        return Lx * h 
    
    def Lt(self, y, h):
        """Adjoint finite difference operator."""

        Lt1y_1 = F.conv_transpose2d(y[:, 0:1, :-1, :-1], self.filters1)[:, :, 1:, 1:-1]
        Lt2y_1 = F.conv_transpose2d(y[:, 1:2, :-1, :-1], self.filters2)[:, :, 1:-1, 1:]

        Lty = Lt1y_1 + Lt2y_1 
        
        return Lty * h 

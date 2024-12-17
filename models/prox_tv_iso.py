import torch
import torch.nn.functional as F 
import numpy as np 
from models.tv_iso import TV_ISO

class prox_tv_iso(): 
    """Proximal operator of the TV-ISO norm."""

    def __init__(self, device, box_const=None):
        """Initialization."""
        self.device = device
        self.tv = TV_ISO(device)
        self.box_const = box_const
    
    def P_c(self, x):
        """Projection onto the box constraint."""
        if self.box_const is None:
            return x
        else:
            return torch.clip(x, self.box_const[0], self.box_const[1])

    def eval(self, y, niter, lmbda, h=1, verbose=False, stop=-1):
        """Evaluation of the proximal operator."""

        # Initialization
        v_k = torch.zeros((1, 2, y.size(2)+2, y.size(3)+2), requires_grad=False, device=self.device).double()
        u_k = torch.zeros((1, 2, y.size(2)+2, y.size(3)+2), requires_grad=False, device=self.device).double()
        t_k = 1 

        alpha = 1 / (8 * lmbda)

        self.loss_list = list()
        loss_old = ((y)**2).sum() / 2
        
        for iters in range(niter):
            # Compute the adjoint operator of v_k
            LTv = self.tv.Lt(v_k, h)
            # Compute the gradient of the data fidelity term
            Lpc = self.tv.L(self.P_c(y - lmbda * LTv), h)

            if verbose: 
                # Compute the loss and append to the loss list
                loss = ((y - self.P_c(y - lmbda * LTv))**2).sum() / 2 + lmbda * torch.norm(Lpc, dim=1, p=2).sum()
                self.loss_list.append(loss.item())
            
            # Update u_k and v_k using FISTA
            u_kp1 = F.normalize(v_k + alpha * Lpc, eps=1, dim=1, p=2)
            t_kp1 = (1 + np.sqrt(4*t_k**2+1)) / 2
            v_kp1 = u_kp1  + (t_k - 1) / t_kp1 * (u_kp1 - u_k)                                                                                                                                                                                                                                                                                                                                                         

            u_k = u_kp1
            v_k = v_kp1
            t_k = t_kp1

            if stop > 0 and verbose:
                # Check for convergence
                loss_new = loss
                crit = torch.abs(loss_new - loss_old)
                if (crit < (stop * loss_old)) and (iters > 3):
                    self.niters = iters + 1
                    break
                
                loss_old = loss_new

        # Compute the final result
        c = y - lmbda * self.tv.Lt(u_k, h)
        c = self.P_c(c)

        return c
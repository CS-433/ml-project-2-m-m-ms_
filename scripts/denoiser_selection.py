from skimage.restoration import denoise_tv_chambolle, denoise_wavelet, denoise_nl_means
import torch
from scipy.ndimage import gaussian_filter, median_filter
from bm3d import bm3d

from scripts.helpers import *
from models.prox_tv_iso import prox_tv_iso
from models.drunet import DRUNet


def select_denoiser(denoiser_name):
    """Select the denoiser and its parameters based on the given denoiser name."""

    if denoiser_name == "Gaussian":
        denoiser = gaussian_filter
        denoiser_params = {'sigma': 2}
    elif denoiser_name == "Median":
        denoiser = median_filter
        denoiser_params = {'size': 2}
    elif denoiser_name == "TV-Chambolle":
        denoiser = denoise_tv_chambolle
        denoiser_params = {'weight': 0.1}
    elif denoiser_name == "TV-ISO":
        denoiser = prox_tv_iso
        denoiser_params = {'lmbda': 0.08, 'niter': 200}
    elif denoiser_name == "Wavelet":
        denoiser = denoise_wavelet
        denoiser_params = {}
    elif denoiser_name == "NL-Means":
        denoiser = denoise_nl_means
        denoiser_params = {}
    elif denoiser_name == "BM3D":
        denoiser = bm3d
        denoiser_params = {'sigma_psd': 0.1}
    elif denoiser_name == "Noise2Noise":
        denoiser = load_model_noise2noise()
        denoiser_params = {}
    else:
        raise ValueError("Unsupported denoiser.")
    return denoiser, denoiser_params


def load_model_noise2noise():
    """Load a pre-trained model for Noise2Noise."""
    
    current_working_dir = os.getcwd()
    model_path = os.path.join(current_working_dir, f'Noise2Noise/exps/Noise2Noise/checkpoints/checkpoint.pth')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    infos = torch.load(model_path, map_location=device)
    config = infos['config']
    log.info(config)

    model = DRUNet(config['net_params']['nb_channels'], config['net_params']['depth'], config['training_options']['color'])
    model.load_state_dict(infos['state_dict'])  # Loads the saved model weights into the new model
    model.eval()  # Set the model to evaluation mode
    
    return model


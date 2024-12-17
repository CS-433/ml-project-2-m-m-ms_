from scripts.helpers import *
from scripts.denoising_pipeline import *

def main():
    """Main script to evaluate denoisers and save results."""
    # Get paths
    data_path, output_path = get_paths()

    # Define global parameter ranges
    PARAMETER_RANGES = {
        "Gaussian": {"param_name": "sigma", "values": [2]},
        "Median": {"param_name": "size", "values": [5]},
        "TV-Chambolle": {"param_name": "weight", "values": [0.1, 0.2, 0.3]},
        "Wavelet": {"param_name": None, "values": None},  # Use default parameters
        "NL-Means": {"param_name": None, "values": None},  # Use default parameters
        "TV-ISO": {"param_name": "lmbda", "values": [0.05]}, # 0.08, 0.1
        "BM3D": {"param_name": "sigma_psd", "values": [0.05, 0.1, 0.15]}, 
        "Noise2Noise": {"param_name": None, "values": None},  # Use default parameters
        "Noise2Void": {"param_name": None, "values": None}# Use default parameters
    }

    disable_progress = True  # Set to False to enable progress bar

    # Choose denoiser
    denoiser_name = "Gaussian"  # "Gaussian", "Median", "TV-Chambolle", "Wavelet", "NL-Means", "TV-ISO", "BM3D", "Noise2Noise", "Noise2Void"

    # Run the denoising pipeline
    denoising_pipeline(data_path, output_path, denoiser_name, PARAMETER_RANGES, num_images=120, num_channels=3, disable_progress=disable_progress)

if __name__ == "__main__":
    main()
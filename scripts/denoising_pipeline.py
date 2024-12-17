import os
import cv2
import pandas as pd
from tqdm import tqdm
import logging as log

from scripts.helpers import *
from models.prox_tv_iso import *
from scripts.denoiser_selection import select_denoiser
from scripts.metrics import compute_metrics

# Set up logging
log.basicConfig(level=log.INFO, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')


def denoising_pipeline(data_path, output_path, denoiser_name, parameter_ranges, num_images=120, num_channels=3, disable_progress=False): 
    """Run the denoising pipeline for a given denoiser and save results to CSV files."""
    
    log.info("Starting denoising pipeline...")
    all_results = []
    
    for channel in tqdm(range(num_channels), desc="Channels", disable=disable_progress):
        log.info(f"Processing channel {channel}...")
        results = process_channel(data_path, output_path, denoiser_name, parameter_ranges, num_images, channel, disable_progress)
        all_results.extend(results)
        log.info(f"Saving results for channel {channel}...")
        save_results_to_csv(results, output_path, denoiser_name, channel)
        
    log.info("Saving average results...")
    save_average_results_to_csv(all_results, output_path, denoiser_name)


def process_channel(data_path, output_path, denoiser_name, parameter_ranges, num_images, channel, disable_progress):
    """Process images in a given channel and return results."""
    
    results = []
    
    for i in tqdm(range(num_images), desc=f"Images in channel {channel}"):
        
        image_index = str(i + 1).zfill(3)
        
        log.info(f"Processing image {image_index}...")
        
        # Load and prepare images
        image, noisy_image, ground_truth_img = load_and_prepare_images(data_path, channel, image_index)
        
        # Select the denoiser
        denoiser, denoiser_params = select_denoiser(denoiser_name)
        
        # Denoise images and compute metrics
        results.extend(denoise_images(denoiser, denoiser_name, parameter_ranges[denoiser_name], noisy_image, ground_truth_img, image_index, channel, disable_progress, output_path))
        
    return results


def load_and_prepare_images(data_path, channel, image_index):
    """Load and prepare images for denoising."""
    
    # Construct the image path
    image_path = f'/Image{image_index}/wf_channel{channel}.npy'
    
    # Load the image
    image = load_image(data_path, image_path)
    
    # Sample and normalize the noisy image
    noisy_image = normalize_image(sample_image(image))
    noisy_image = noisy_image - noisy_image.mean()
    
    # Calculate and normalize the ground truth image
    ground_truth_img = ground_truth(image)
    ground_truth_img = normalize_image(ground_truth_img)
    
    return image, noisy_image, ground_truth_img


def denoise_images(denoiser, denoiser_name, param_config, noisy_image, ground_truth_img, image_index, channel, disable_progress, output_path):
    """Denoise images using the given denoiser and parameter configuration."""
    
    results = []
    
    # If parameter values are provided, iterate over them
    if param_config["values"]:
        param_name = param_config["param_name"]
        for value in tqdm(param_config["values"], disable=disable_progress):
            denoiser_params = {param_name: value}
            denoised_image = apply_denoiser(denoiser, denoiser_name, denoiser_params, noisy_image)
            results.append([image_index, f"{channel}", *compute_metrics(denoised_image, ground_truth_img), denoiser_name, value])
            
            # Save the denoised image for channel 0 and image 001
            if channel == 0 and image_index == '001':
                save_denoised_image(denoised_image, output_path, denoiser_name, channel, image_index, value)
    
    # If no parameter values are provided, apply the denoiser without parameters
    else:
        denoiser_params = {}
        denoised_image = apply_denoiser(denoiser, denoiser_name, denoiser_params, noisy_image)
        results.append([image_index, f"{channel}", *compute_metrics(denoised_image, ground_truth_img), denoiser_name, denoiser_params])
        
        # Save the denoised image for channel 0 and image 001
        if channel == 0 and image_index == '001':
            save_denoised_image(denoised_image, output_path, denoiser_name, channel, image_index)
        
    return results


def apply_denoiser(denoiser, denoiser_name,  denoiser_params, noisy_image):
    """Apply the denoiser to the noisy image."""
    
    log.info(f"Applying denoiser {denoiser_name} with parameters {denoiser_params}...")
    
    # Apply the denoiser to the noisy image (TV-ISO)
    if denoiser == prox_tv_iso:
        denoiser_params['niter'] = 200
        prox_tv = denoiser(device='cpu')
        return prox_tv.eval(torch.from_numpy(noisy_image).unsqueeze(0).unsqueeze(0).float(), **denoiser_params).squeeze().numpy()
    
    # Apply the denoiser to the noisy image (Noise2Noise)
    elif denoiser_name == 'Noise2Noise':
        noisy_image_reshaped = noisy_image.reshape(1, 1, 512, 512)  # Reshape the image to (Batch, Channels, Height, Width)
        noisy_image_tensor = torch.tensor(noisy_image_reshaped, dtype=torch.float32)
        with torch.no_grad():
            denoised_image = denoiser(noisy_image_tensor).view(512, 512).numpy()
        return denoised_image
    
    # Apply the denoiser to the noisy image (Noise2Void)
    elif denoiser_name == 'Noise2Void':
        return 0
    
    # Apply the denoiser to the noisy image (all other denoisers)
    else:
        return denoiser(noisy_image, **denoiser_params)


def save_denoised_image(denoised_image, output_path, denoiser_name, channel, image_index, value=None):
    """Save the denoised image to the specified output path."""
    
    processed_folder = output_path.replace('output', 'processed')
    os.makedirs(processed_folder, exist_ok=True)
    
    # Save the denoised image as a .npy file
    if value:
        denoiser_folder = os.path.join(processed_folder, f'{denoiser_name}')
        os.makedirs(denoiser_folder, exist_ok=True)
        denoised_image_path = os.path.join(denoiser_folder, f"{denoiser_name}_denoised_channel{channel}_image{image_index}_param{value}.npy")
    else:
        denoiser_folder = os.path.join(processed_folder, f'{denoiser_name}')
        os.makedirs(denoiser_folder, exist_ok=True)
        denoised_image_path = os.path.join(denoiser_folder, f"{denoiser_name}_denoised_channel{channel}_image{image_index}.npy")
    
    np.save(denoised_image_path, denoised_image)
    log.info(f"Denoised image saved to {denoised_image_path}")


def save_results_to_csv(results, output_path, denoiser_name, channel):
    """Save denoising results to a CSV file."""
    
    # Create the denoiser-specific folder if it doesn't exist
    denoiser_folder = os.path.join(output_path, denoiser_name)
    os.makedirs(denoiser_folder, exist_ok=True)
    
    # Save results to CSV
    results_df = pd.DataFrame(results, columns=['ImageIndex', 'Channel', 'PSNR', 'SI-PSNR', 'SSIM', 'DenoiserType', 'Parameter'])
    results_df = results_df[['DenoiserType', 'Parameter', 'ImageIndex', 'Channel', 'PSNR', 'SI-PSNR', 'SSIM']]
    results_df.to_csv(os.path.join(denoiser_folder, f"{denoiser_name}_channel{channel}_denoiser_results.csv"), index=False)


def save_average_results_to_csv(all_results, output_path, denoiser_name):
    """Save average denoising results to a CSV file."""
    
    # Create the denoiser-specific folder if it doesn't exist
    denoiser_folder = os.path.join(output_path, denoiser_name)
    os.makedirs(denoiser_folder, exist_ok=True)
    
    # Save average results to CSV
    all_results_df = pd.DataFrame(all_results, columns=['ImageIndex', 'Channel', 'PSNR', 'SI-PSNR', 'SSIM', 'DenoiserType', 'Parameter'])
    all_results_df = all_results_df[['DenoiserType', 'Parameter', 'ImageIndex', 'Channel', 'PSNR', 'SI-PSNR', 'SSIM']]
    
    if (all_results_df['Parameter'] == {}).all():
        # If no parameter, perform groupby on 'DenoiserType' and 'Channel' only
        all_results_df = all_results_df.groupby(['DenoiserType', 'Channel'], as_index=False)[['PSNR', 'SI-PSNR', 'SSIM']].mean().reset_index(drop=True)
    else:
        # If parameter, perform groupby on 'DenoiserType', 'Parameter', and 'Channel'
        all_results_df = all_results_df.groupby(['DenoiserType', 'Parameter', 'Channel'], as_index=False)[['PSNR', 'SI-PSNR', 'SSIM']].mean().reset_index(drop=True)
    
    all_results_df.to_csv(os.path.join(denoiser_folder, f"avg_{denoiser_name}_denoiser_results.csv"), index=False)
    

def add_inset(ax, x, y, size, color='white'):
    """Add a white border inset to the image."""

    # Draw all borders
    ax.plot([x, x+size], [y, y], color=color, linewidth=1)
    ax.plot([x, x+size], [y+size, y+size], color=color, linewidth=1)
    ax.plot([x, x], [y, y+size], color=color, linewidth=1)
    ax.plot([x+size, x+size], [y, y+size], color=color, linewidth=1)


def process_image(image, ax, title, inset_size):
    """Process the image and display it with a white border inset."""

    # Extract the region from (x1, y1) to (x2, y2)
    x1, y1 = 64, 64
    x2, y2 = 192, 192
    region = image[x1:x2, y1:y2]

    # Resize the region to inset_size * inset_size
    resized_region = cv2.resize(region, (inset_size, inset_size), interpolation=cv2.INTER_LINEAR)

    # Copy this resized region to the top-right corner of the image
    image_copy = image.copy()
    image_copy[1:1+inset_size, -inset_size:] = resized_region

    # Display the image
    ax.imshow(image_copy, cmap='viridis')
    ax.set_title(title, fontsize=15)
    ax.axis('off')

    # Add white border inset to the original and the copied region
    add_inset(ax, 64, 64, 128)
    add_inset(ax, image_copy.shape[1] - inset_size, 1, inset_size)

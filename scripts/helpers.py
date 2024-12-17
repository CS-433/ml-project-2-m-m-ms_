import os
import torch
import numpy as np
import logging as log
from IPython.display import display
from matplotlib import pyplot as plt

log.basicConfig(level=log.INFO)

IMAGE_INDEX = 249


def get_paths():
    """
    Returns the data and output paths relative to the current working directory.
    
    :return: A tuple (data_path, output_path)
    """
    data_path = os.path.abspath(os.path.join(os.getcwd(), 'data/raw')) + '/'
    output_path = os.path.abspath(os.path.join(os.getcwd(), 'data/output')) + '/'
    return data_path, output_path


def load_image(path_to_data, image_path):
    """Load a .npy image file from the specified path."""
    return np.load(path_to_data + image_path)


def ground_truth(image):
    """Calculate ground truth as the mean projection of image along the first axis."""
    return image.mean(axis=0)


def sample_image(image, random=False):
    """Randomly sample an image slice."""
    if random:
        sampled_image_index = np.random.randint(0, image.shape[0])
        return image[sampled_image_index]
    return image[IMAGE_INDEX]


def data_range(ground_truth_image):
    """Calculate data range for PSNR and SSIM calculations."""
    return ground_truth_image.max() - ground_truth_image.min()


def normalize_image(image): 
    """Normalizes an image if pixels range are not between 0 and 1."""
    if (not (0 <= image.min() and image.max() <=1 )): 
        return (image - image.min()) / (image.max() - image.min()) 
    else : 
        return image


def display_styled_results(df, output_path, output_file, title):
    """
    Display styled DataFrame and save to a CSV file.

    Args:
        df (pd.DataFrame): The DataFrame to display and save.
        output_path (str): Output directory path.
        output_file (str): Filename for saving the CSV.
        title (str): Title to display before the DataFrame.
    """
    log.info(df.columns)
    
    # Format and style the DataFrame
    styled_df = df.style.format({
        'PSNR': "{:.2f}",
        'SI-PSNR': "{:.2f}",
        'SSIM': "{:.4f}", 
        'Runtime': "{:.2f}"
    }).background_gradient(subset=['PSNR', 'SI-PSNR', 'SSIM', 'TotalRuntime'])

    # Display styled DataFrame
    log.info(f"\n{title}:")
    display(styled_df)


def plot_denoiser_results(images, noisy_images, denoised_images, si_psnr_noisy_list, si_psnr_denoised_list, title, output_path):
    """
    Plot the ground truth, noisy, and denoised images with their SI-PSNR values for multiple channels.

    :param images: List of ground truth images (numpy arrays)
    :param noisy_images: List of noisy images (numpy arrays)
    :param denoised_images: List of denoised images (numpy arrays or torch tensors)
    :param si_psnr_noisy_list: List of SI-PSNR values of the noisy images
    :param si_psnr_denoised_list: List of SI-PSNR values of the denoised images
    :param title: Title for the plot
    :param output_path: Path to save the plot image
    """
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle(title, fontsize=16)
    channels = ['Channel 0', 'Channel 1', 'Channel 2']
    
    for i in range(3):
        axes[i, 0].imshow(images[i])
        axes[i, 0].set_title(f'{channels[i]} Ground Truth')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(noisy_images[i])
        axes[i, 1].set_title(f'Noisy, SI-PSNR={np.round(si_psnr_noisy_list[i], 2)}')
        axes[i, 1].axis('off')

        if isinstance(denoised_images[i], torch.Tensor):
            denoised_image = denoised_images[i][0, 0].detach().cpu().numpy()
        else:
            denoised_image = denoised_images[i]
        
        axes[i, 2].imshow(denoised_image)
        axes[i, 2].set_title(f'Denoised, SI-PSNR={np.round(si_psnr_denoised_list[i], 2)}')
        axes[i, 2].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make room for the title

    # Create the plots directory if it doesn't exist
    plots_path = os.path.join(output_path, 'plots')
    os.makedirs(plots_path, exist_ok=True)

    # Save the plot to a file
    plot_file_path = os.path.join(plots_path, f"{title.replace(' ', '_')}.png")
    plt.savefig(plot_file_path)
    
    print(f"\n{title}:")
    plt.show()
    print(f"Plot saved to {plot_file_path}")

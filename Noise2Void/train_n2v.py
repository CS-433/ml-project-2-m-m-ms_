from careamics import CAREamist
from careamics.config import create_n2v_configuration
import numpy as np
import logging as log
import sys
import os

# Add the top-level and the script directories to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..', 'scripts')))

from scripts.helpers import get_paths, ground_truth, normalize_image
from scripts.metrics import compute_metrics

log.basicConfig(level=log.INFO)

def load_dataset(data_path):
    """
    Load all images from the dataset into a single NumPy array.

    :param data_path: Path to the root data directory.
    :return: NumPy array of shape (120, 3, 512, 512) containing the images.
    """
    log.info(f"Loading dataset from {data_path}")
    
    # Define the dataset parameters
    num_images = 120
    num_channels = 3
    image_shape = (512, 512)
    
    # Initialize an empty array to hold the dataset
    dataset = np.zeros((num_images, num_channels, *image_shape), dtype=np.float32)
    
    for channel in range(num_channels):
        for image_index in range(1, num_images + 1):
            image_index_str = str(image_index).zfill(3)
            image_path = os.path.join(data_path, f'Image{image_index_str}', f'wf_channel{channel}.npy')
            image = np.load(image_path)
            dataset[image_index - 1, channel, :, :] = image[249, :, :]
    
    return dataset

current_working_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_working_dir, '..'))

data_path = os.path.join(parent_dir, f'data/raw')
output_path = os.path.join(parent_dir, f'data/processed')
log.info(f"Data path: {data_path}")
log.info(f"Output path: {output_path}")

dataset = load_dataset(data_path)
log.info(f"Dataset shape: {dataset.shape}")

image_path = os.path.join(data_path, f'Image001/wf_channel0.npy')
image = np.load(image_path)

noisy_image = image[249, :, :]

ground_truth_image = normalize_image(ground_truth(image))

log.info(ground_truth_image.shape)
log.info(noisy_image.shape)


dataset_bis = dataset.reshape(360, 1, 512, 512)

log.info("Splitting the dataset into training and validation sets...")
split_ratio = 0.8
split_idx = int(len(dataset_bis) * split_ratio)

seed = 42
np.random.seed(seed)
np.random.shuffle(dataset_bis)
train, val = dataset_bis[:split_idx], dataset_bis[split_idx:]
log.info(f"Training set shape: {train.shape}")
log.info(f"Validation set shape: {val.shape}")

config = create_n2v_configuration(
    experiment_name="w2s_n2v_test",
    data_type="array",
    axes="SYX",
    patch_size=(64, 64),
    batch_size=32,
    num_epochs=15,
    n_channels=1
)

log.info("Initializing CAREamist...")
careamist = CAREamist(
    source=config,
    work_dir='models/noise2void_weights/'
)

log.info(config)

log.info("Training the model...")
careamist.train(train_source=train.reshape(-1, 512, 512), val_source=val.reshape(-1, 512, 512))
log.info("Training complete.")

log.info("Predicting on a noisy image...")

prediction = careamist.predict(
    source=noisy_image.reshape(1, 512, 512),
    batch_size=1,
)

log.info("Computing metrics...")
metrics = compute_metrics(np.array(prediction[0]).squeeze(), ground_truth_image)

log.info(f"PSNR: {metrics[0]}")
log.info(f"SI-PSNR: {metrics[1]}")
log.info(f"SSIM: {metrics[2]}")

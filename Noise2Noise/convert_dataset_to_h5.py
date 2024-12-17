"""Convert dataset to h5 format for Noise2Noise training in order to load the data faster."""

import h5py
import numpy as np
import os

num_images = 120
num_channels = 3
num_frames = 400
image_size = (512, 512)

with h5py.File('w2s_bw_test.h5', 'w') as f:
    for image_index in range(num_images):
        for channel_index in range(num_channels):  
            
            data_path = os.getcwd()
            image_path = f'/data/raw/Image{str(image_index + 1).zfill(3)}/wf_channel{channel_index}.npy'
            image_data = np.load(data_path + image_path)
            
            for frame_idx in range(num_frames):
                
                print(f'Image {image_index + 1} / {num_images}, Channel {channel_index + 1} / {num_channels}, Frame {frame_idx + 1} / {num_frames}')

                dataset_name = str(image_index * num_frames * num_channels + channel_index * num_frames + frame_idx)
                f.create_dataset(dataset_name, data=image_data[frame_idx, :, :])
    

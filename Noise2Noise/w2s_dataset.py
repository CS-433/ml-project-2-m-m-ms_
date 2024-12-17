import h5py
import torch
from torch.utils.data import Dataset
import numpy as np


class W2SDataset(Dataset):
    """Dataset class for the W2S dataset."""

    def __init__(self, color, patch_size, nb_iter, data_augmentation=True):
        """Initialize the dataset class."""
        super(Dataset, self).__init__()
        self.color = color
        if self.color: self.data_file = 'w2s_color.h5'
        else: self.data_file = 'w2s_bw.h5'
        self.patch_size = patch_size # size of the image patches to be extracted
        self.nb_iter = nb_iter # number of iterations (length of the dataset)
        self.data_augmentation = data_augmentation 
        self.dataset = None

    def __len__(self):
        """Return the number of iterations."""
        return self.nb_iter

    def __getitem__(self, idx):
        """Return a random image patch."""
        if self.dataset is None:
            self.dataset = h5py.File(self.data_file, 'r')
        image_idx = torch.randint(0, 120, (1,)).item()
        channel_idx = torch.randint(0, 3, (1,)).item()
        frame_idx1, frame_idx2 = torch.randint(0, 400, (1,)).item(), torch.randint(0, 400, (1,)).item()
        while frame_idx1 == frame_idx2: frame_idx2 = torch.randint(0, 400, (1,)).item()
        pos_x = torch.randint(0, 512-self.patch_size, (1,)).item()
        pos_y = torch.randint(0, 512-self.patch_size, (1,)).item()
        if self.color: 
            data = torch.zeros(6, self.patch_size, self.patch_size)
            data[:3] = torch.Tensor(np.array(self.dataset[str(image_idx*400+frame_idx1)]))[:,pos_x:pos_x+self.patch_size,pos_y:pos_y+self.patch_size]
            data[3:] = torch.Tensor(np.array(self.dataset[str(image_idx*400+frame_idx2)]))[:,pos_x:pos_x+self.patch_size,pos_y:pos_y+self.patch_size]
        else:
            data = torch.zeros(2, self.patch_size, self.patch_size)
            data[0] = torch.Tensor(np.array(self.dataset[str(image_idx*400*3+channel_idx*400+frame_idx1)]))[pos_x:pos_x+self.patch_size,pos_y:pos_y+self.patch_size]
            data[1] = torch.Tensor(np.array(self.dataset[str(image_idx*400*3+channel_idx*400+frame_idx2)]))[pos_x:pos_x+self.patch_size,pos_y:pos_y+self.patch_size]
        data = data - data.mean(dim=(1,2), keepdim=True)
        if self.data_augmentation:
            augment_idx = torch.randint(0, 7, (1,)).item()
            if augment_idx == 1:
                data = torch.flip(data, [1])
            elif augment_idx == 2:
                data = torch.flip(data, [2])
            elif augment_idx == 1:
                data = torch.rot90(data, 1, [1, 2])
            elif augment_idx == 2:
                data = torch.rot90(data, 2, [1, 2])
            elif augment_idx == 3:
                data = torch.rot90(data, 3, [1, 2])
            elif augment_idx == 6:
                data = torch.flip(torch.rot90(data, 1, [1, 2]), [1])
            elif augment_idx == 7:
                data = torch.flip(torch.rot90(data, 1, [1, 2]), [2])
        return data
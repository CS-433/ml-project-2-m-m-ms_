import torch
import os, sys
import json
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils import tensorboard
from w2s_dataset import W2SDataset
from torch_metrics import scale_invariant_psnr
import logging as log 

# Add the top-level and the models directories to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..', 'models')))

from models.drunet import DRUNet

torch.manual_seed(0)
log.basicConfig(level=log.INFO)

class Trainer:
    """ Trainer class for training the model."""
    def __init__(self, config, device):
        self.config = config
        self.device = device
        
        log.info(config)

        # Prepare dataset classes
        self.color = config['training_options']['color']
        train_dataset = W2SDataset(self.color, config['training_options']['patch_size'], \
                        config['training_options']['total_steps']*config["training_options"]["batch_size"])

        log.info('Preparing the dataloaders')
        # Prepare dataloaders 
        self.train_dataloader = DataLoader(train_dataset, batch_size=config["training_options"]["batch_size"], \
                                           shuffle=True, num_workers=config["training_options"]["num_workers"], drop_last=True)
        self.batch_size = config["training_options"]["batch_size"]

        log.info('Building the model')
        # Build the model
        self.model = DRUNet(config['net_params']['nb_channels'], config['net_params']['depth'], \
                             self.color)
        self.model = self.model.to(device)

        log.info(self.model)
        
        # Set up the optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config["training_options"]["lr"])
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.config["training_options"]["gamma_scheduler"])
        
        self.criterion = torch.nn.MSELoss(reduction='mean')

        # CHECKPOINTS & TENSOBOARD
        run_name = config['exp_name']
        self.checkpoint_dir = os.path.join(config['log_dir'], config["exp_name"], 'checkpoints')
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        config_save_path = os.path.join(config['log_dir'], config["exp_name"], 'config.json')
        with open(config_save_path, 'w') as handle:
            json.dump(self.config, handle, indent=4, sort_keys=True)

        writer_dir = os.path.join(config['log_dir'], config["exp_name"], 'tensorboard_logs')
        self.writer = tensorboard.SummaryWriter(writer_dir)

        self.total_training_step = 1
        self.total_testing_step = 0    


    def train(self):
        """Training the model."""
        self.model.train()
        tbar = tqdm(self.train_dataloader, ncols=135, position=0, leave=True)
        log = {}
        for batch_idx, data in enumerate(tbar):
            if self.color:
                data1 = data[:,:3].to(self.device)
                data2 = data[:,3:].to(self.device)
            else:
                data1 = data[:,:1].to(self.device)
                data2 = data[:,1:2].to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data1)
            # data fidelity
            data_fidelity = self.criterion(output, data2)
                    
            # regularization
            total_loss = data_fidelity
            total_loss.backward()
            self.optimizer.step()
                    
            log['train_loss'] = total_loss.detach().cpu().item()

            self.wrt_step = self.total_training_step
            self.write_scalars_tb(log)

            tbar.set_description('TotalLoss {:.5f} |'.format(log['train_loss'])) 
            if self.total_training_step % self.config['training_options']['lr_decay_step'] == 0:
                self.scheduler.step()
            if self.total_training_step % self.config['training_options']['testing_step'] == 0:
                self.valid_epoch(self.total_testing_step)
                self.save_checkpoint()
                self.total_testing_step += 1
                self.model.train()
            self.total_training_step += 1
        self.writer.flush()
        self.writer.close()


    def valid_epoch(self, epoch):
        """Validation of the model."""
        self.model.eval()
        si_psnr_mean = 0.
        tbar = tqdm(range(120), ncols=135, position=0, leave=True)
        with torch.no_grad():
            for k in tbar:
                image_index = str(k + 1).zfill(3)
                
                current_working_dir = os.getcwd()
                parent_dir = os.path.abspath(os.path.join(current_working_dir, '..'))
                data_path = os.path.join(parent_dir, f'data/raw')
                
                if self.color:
                    gt = torch.tensor(np.load(f'raw/Image{k+1}/wf_mean.npy'))
                    frame = torch.zeros(1, 3, 512, 512, device=self.device)
                    frame[0,0] = torch.tensor(np.load(os.path.join(data_path, f'Image{image_index}/wf_channel0.npy')))[200]
                    frame[0,1] = torch.tensor(np.load(os.path.join(data_path, f'Image{image_index}/wf_channel1.npy')))[200]
                    frame[0,2] = torch.tensor(np.load(os.path.join(data_path, f'Image{image_index}/wf_channel2.npy')))[200]
                else:
                    gt = torch.tensor(np.load(os.path.join(data_path, f'Image{image_index}/wf_channel1.npy')).mean(axis=0))[1]
                    frame = torch.tensor(np.load(os.path.join(data_path, f'Image{image_index}/wf_channel1.npy')))[200][None,None,...].float().to(self.device)
                frame = frame - frame.mean(dim=(2,3), keepdim=True)
                output = self.model(frame).squeeze()
                si_psnr_mean += scale_invariant_psnr(np.array(gt.cpu()), np.array(output.cpu())) / 120
        self.wrt_mode = 'val'
        self.writer.add_scalar(f'{self.wrt_mode}/SI-PSNR', si_psnr_mean, epoch)


    def write_scalars_tb(self, logs):
        """Write scalars to tensorboard."""
        for k, v in logs.items():
            self.writer.add_scalar(f'train/{k}', v, self.wrt_step)


    def save_checkpoint(self):
        """Save the model checkpoint."""
        state = {
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.config
        }

        log.info('Saving a checkpoint:')
        filename = self.checkpoint_dir + '/checkpoint.pth'
        torch.save(state, filename)
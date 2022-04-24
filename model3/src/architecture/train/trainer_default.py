from pickletools import optimize

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.nn import MSELoss
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import StepLR
from architecture.dataset import Dataset
from torch.utils.data import DataLoader
import os.path as osp
from PIL import Image

class TrainerDefault:
    # If in debug mode, print some more stuff
    MODE = 'd' # either 'p' (production) or 'd' (debug)

    # Filesystem parameters 
    output_dir = None
    train_csv = None
    val_csv = None
    dataset_root = None

    # Training parameters
    learning_rate = 0.0005
    epoch_num = 25
    save_checkpoint_every_n_epochs = 5
    batch_size = 100
    batches_per_step = 1

    # Training moderators
    optimizer = SGD
    optimizer_kwds = {
        'momentum': 0.1,
        'weight_decay': 0.0001
    }

    criterion = MSELoss
    criterion_kwds = {
        'reduction': 'sum'
    }

    scheduler = StepLR
    scheduler_kwds = {
        'step_size': 7,
        'gamma': 0.3
    }

    num_workers = 4

    # Model details
    model = None

    # Dataset details
    # data_transforms default values are NOT provided because user has to be
    # aware of the input dimensions of the images.
    dataset_train_kwds = {
        'transform': None,
        'image_directions': ['N', 'E', 'S', 'W']
    }
    dataset_val_kwds = {
        'transform': None,
        'image_directions': ['N', 'E', 'S', 'W']
    }

    # Training progress tracking
    losses_train = []
    losses_val = []

    # Other
    device = 'cpu'

    def initialize(self):
        """
        This function is automatically called at the beginning of training.
        It requires that all the hyperparameters are set up.
        """
        self.device = torch.device(self.device)
        self.best_validation_loss = torch.inf
        self.criterion = self.criterion(**self.criterion_kwds)
        self.optimizer = self.optimizer(
            self.model.parameters(),
            lr=self.learning_rate,
            **self.optimizer_kwds)
        self.scheduler = self.scheduler(self.optimizer, **self.scheduler_kwds)

        self.dataset_train = Dataset(
            self.dataset_root, 
            csv_file=self.train_csv, 
            **self.dataset_train_kwds)

        self.dataset_val = Dataset(
            self.dataset_root, 
            csv_file=self.val_csv,
            **self.dataset_val_kwds)

        self.dataloader_train = DataLoader(
            self.dataset_train, 
            batch_size=self.batch_size, 
            shuffle=True, 
            drop_last=True, 
            num_workers=self.num_workers)

        self.dataloader_val = DataLoader(
            self.dataset_val, 
            batch_size=self.batch_size, 
            drop_last=False, 
            num_workers=self.num_workers)

    def visualize_dataset(self):
        """
        This method can be called after self.initialize() has been processed.
        It saves all the input images after transforms to the output directory.
        """
        print('Visualizing dataset.')

        def save_img(prefix, img, batch_ind, dir):
            Image.fromarray(
                np.array(
                    torch.flatten(
                        torch.permute(img, (0, 2, 3, 1)),
                        start_dim=0,
                        end_dim=1
                    )*255., 
                    dtype='uint8'
                )
            ).save(osp.join(self.output_dir, f'{prefix}_{batch_ind:05}_{dir}.png'))

        for batch_ind, batch in enumerate(self.dataloader_train):
            print(f'-Batch {batch_ind+1}.')
            image_N = batch['image_N']
            image_E = batch['image_E']
            image_S = batch['image_S']
            image_W = batch['image_W']
            
            save_img('train_b', image_N, batch_ind, 'N')
            save_img('train_b', image_E, batch_ind, 'E')
            save_img('train_b', image_S, batch_ind, 'S')
            save_img('train_b', image_W, batch_ind, 'W')

        for batch_ind, batch in enumerate(self.dataloader_val):
            print(f'-Batch {batch_ind+1}.')
            image_N = batch['image_N']
            image_E = batch['image_E']
            image_S = batch['image_S']
            image_W = batch['image_W']
            
            save_img('val_b', image_N, batch_ind, 'N')
            save_img('val_b', image_E, batch_ind, 'E')
            save_img('val_b', image_S, batch_ind, 'S')
            save_img('val_b', image_W, batch_ind, 'W')
        
        print('Visualizations finished.')
        print(f'All outputs saved to {self.output_dir}')

    def train(self):
        assert self.model is not None, "TrainerDefault cannot be used on it's" \
            " own. Custom Trainer should either inherit TrainerDefault and" \
            " specify Trainer.model or should be defined from scratch."
        
        self.initialize()
        
        self.model.to(self.device)

        self.output_path_parameters = osp.join(self.output_dir, 'parameters')
        self.output_path_losses_train = osp.join(self.output_dir, 
                                                 'losses_train.csv')
        self.output_path_losses_val = osp.join(self.output_dir, 
                                               'losses_val.csv')
        self.output_path_losses_img = osp.join(self.output_dir, 
                                               'losses_img.png')

        print(self.model)

        print('Initial validation.')
        self.validation_epoch()

        for epoch in range(self.epoch_num):
            print(f'-Training. Epoch {epoch+1}. '
                f'lr={self.scheduler.get_last_lr()}')
            self.training_epoch()
            
            print(f'-Validation. Epoch {epoch+1}.')
            self.validation_epoch()

            self.handle_saving(epoch)
        
        print(f'Training finished. All outputs saved to {self.output_dir}.')
    
    def handle_saving(self, epoch):
        if (epoch+1) % self.save_checkpoint_every_n_epochs == 0:
            self.save_model_parameters(
                f'{self.output_path_parameters}_epoch_{epoch+1}.prms')
            print(f'-Checkpoint saved to '
                f'{self.output_path_parameters}_epoch_{epoch+1}.prms')
        if len(self.losses_val)>=1 \
            and self.losses_val[-1] < self.best_validation_loss:
            self.best_validation_loss = self.losses_val[-1]
            self.save_model_parameters(
                f'{self.output_path_parameters}_best.prms')
            print(f'-Checkpoint saved to '
                f'{self.output_path_parameters}_best.prms')

        self.save_losses(
            self.output_path_losses_img, 
            self.output_path_losses_train,
            self.output_path_losses_val)

    def training_epoch(self):
        self.optimizer.zero_grad()
        self.model.train()

        loss_running = torch.tensor(0., device=self.device)

        for batch_ind, batch in enumerate(self.dataloader_train):
            input = {
                'image_N': batch['image_N'].to(self.device),
                'image_E': batch['image_E'].to(self.device),
                'image_S': batch['image_S'].to(self.device),
                'image_W': batch['image_W'].to(self.device)
            }
            
            output_gt = batch['output'].to(self.device)
            output_train = self.model(input)
            loss_batch = self.criterion(output_train, output_gt) 
            loss_batch.backward()
            loss_running += loss_batch.clone().detach()

            if (batch_ind+1)%self.batches_per_step == 0:
                loss_running /= (self.batch_size*self.batches_per_step)
                self.losses_train.append(float(loss_running))
                if self.MODE == 'd':
                    print(f'--Train loss: {loss_running}')
                loss_running = torch.tensor(0., device=self.device)

                self.optimizer.step()
                self.optimizer.zero_grad()

        self.scheduler.step()

    def validation_epoch(self):
        self.model.eval()
        loss_running = torch.tensor(0., device=self.device)
        
        for batch_ind, batch in enumerate(self.dataloader_val):
            input = {
                'image_N': batch['image_N'].to(self.device),
                'image_E': batch['image_E'].to(self.device),
                'image_S': batch['image_S'].to(self.device),
                'image_W': batch['image_W'].to(self.device)
            }

            output_gt = batch['output'].to(self.device)
            output_val = self.model(input)
            loss_running += self.criterion(
                output_val, output_gt
            ).clone().detach()
        loss_running /= len(self.dataset_val)   
        self.losses_val.append(float(loss_running))
        if self.MODE == 'd':
            print(f'--Validation loss: {loss_running}.')
    
    def save_model_parameters(self, filename):
        torch.save(self.model.state_dict(), filename)

    def save_losses(self, output_path_losses_img, output_path_losses_train, 
        output_path_losses_val):
        plt.clf()
        plt.plot(np.linspace(0, 1, len(self.losses_train)), self.losses_train, 
            label='Train loss')
        plt.plot(np.linspace(0, 1, len(self.losses_val)), self.losses_val, 
            label='Validation loss')
        plt.savefig(output_path_losses_img)
        pd.DataFrame({
            'loss_train': self.losses_train
            }).to_csv(output_path_losses_train)
        pd.DataFrame({
            'loss_val': self.losses_val
            }).to_csv(output_path_losses_val)
        


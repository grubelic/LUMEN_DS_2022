from pickletools import optimize

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os 

import torch
from torch.nn import MSELoss, DataParallel
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import StepLR
from architecture.dataset import Dataset
from torch.utils.data import DataLoader
import os.path as osp
from PIL import Image
from datetime import datetime

from architecture.evaluation.evaluation import distance_descriptive
from architecture.evaluation.evaluation import distance_histogram
from architecture.evaluation.evaluation import distance_density


class TrainerDefault:
    # If in debug mode, print some more stuff
    RUN_MODE = 'd' # either 'p' (production) or 'd' (debug)

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
    device = ['cpu']

    def log_message(self, msg):
        print(datetime.now().strftime("[%Y/%b/%d %H:%M:%S]"), msg)

    def initialize(self):
        """
        This function is automatically called at the beginning of training.
        It requires that all the hyperparameters are set up.
        """
        self.device = [torch.device(dev) for dev in self.device]
        if len(self.device) > 1:
            raise NotImplementedError("Multi GPU not yet supported")
            # self.model = DataParallel(self.model, device_ids=self.device)
        self.best_validation_loss = torch.inf
        self.criterion = self.criterion(**self.criterion_kwds)
        self.optimizer = self.optimizer(
            self.model.parameters(),
            lr=self.learning_rate,
            **self.optimizer_kwds)
        self.scheduler = self.scheduler(self.optimizer, **self.scheduler_kwds)
        print('Loading training dataset...')
        self.dataset_train = Dataset(
            self.dataset_root, 
            csv_file=self.train_csv, 
            **self.dataset_train_kwds)
        print('Loading validation dataset...')
        self.dataset_val = Dataset(
            self.dataset_root, 
            csv_file=self.val_csv,
            **self.dataset_val_kwds)
        print('Preparing dataloaders...')
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
        
        self.model.to(self.device[0])

        self.output_path_parameters = osp.join(self.output_dir, 'parameters')
        self.output_path_losses_train = osp.join(self.output_dir, 
                                                 'losses_train.csv')
        self.output_path_losses_val = osp.join(self.output_dir, 
                                               'losses_val.csv')
        self.output_path_losses_img = osp.join(self.output_dir, 
                                               'losses_img.png')

        print(self.model)

        self.log_message('Initial validation.')
        self.validation_epoch(epoch=0)
        
        self.make_dir_train(output_dir = self.output_dir)

        for epoch in range(self.epoch_num):
            
            self.log_message(f'Training. Epoch {epoch+1}. '
                f'lr={self.scheduler.get_last_lr()}')
            self.training_epoch()
            
            self.log_message(f'Validation. Epoch {epoch+1}.')
            self.validation_epoch(epoch=epoch+1)

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

        loss_running = torch.tensor(0., device=self.device[0])

        for batch_ind, batch in enumerate(self.dataloader_train):
            input = {
                'image_N': batch['image_N'].to(self.device[0]),
                'image_E': batch['image_E'].to(self.device[0]),
                'image_S': batch['image_S'].to(self.device[0]),
                'image_W': batch['image_W'].to(self.device[0])
            }
            
            output_gt = batch['output'].to(self.device[0])
            output_train = self.model(input)
            loss_batch = self.criterion(output_train, output_gt)
            loss_batch /= self.batches_per_step*self.batch_size
            loss_batch.backward()
            loss_running += loss_batch.clone().detach()

            if (batch_ind+1)%self.batches_per_step == 0:
                # loss_running /= (self.batch_size*self.batches_per_step)
                self.losses_train.append(float(loss_running))
                if self.RUN_MODE == 'd':
                    print(f'--Train loss: {loss_running}')
                loss_running = torch.tensor(0., device=self.device[0])

                self.optimizer.step()
                self.optimizer.zero_grad()

        self.scheduler.step()

    def validation_epoch(self, epoch):
        self.model.eval()
        loss_running = torch.tensor(0., device=self.device[0])
        
        validation_df = pd.DataFrame(
            columns=['uuid', 'gt_latitude', 'gt_longitude', 
            'mo_latitude', 'mo_longitude'])

        for batch_ind, batch in enumerate(self.dataloader_val):
            input = {
                'image_N': batch['image_N'].to(self.device[0]),
                'image_E': batch['image_E'].to(self.device[0]),
                'image_S': batch['image_S'].to(self.device[0]),
                'image_W': batch['image_W'].to(self.device[0])
            }

            output_gt = batch['output'].to(self.device[0])
            output_val = self.model(input)
            loss_running += self.criterion(
                output_val, output_gt
            ).clone().detach()
            
            output_gt_denorm = self.dataset_val.denormalize_output(
                output_gt.clone().detach())
            output_mo_denorm = self.dataset_val.denormalize_output(
                output_val.clone().detach())

            validation_df = pd.concat([
                validation_df, 
                pd.DataFrame({
                    'uuid': batch['uuid'],
                    'gt_latitude': output_gt_denorm[:, 0],
                    'gt_longitude': output_gt_denorm[:, 1],
                    'mo_latitude': output_mo_denorm[:, 0],
                    'mo_longitude': output_mo_denorm[:, 1]
                })], ignore_index=True)
            
            distance_analysis(validation_df, self.output_dir)
         
        self.epoch_analysis(self, self.output_dir, epoch, validation_df)   
            
        
            
        self.validation_plot(validation_df, epoch)
            
        loss_running /= len(self.dataset_val)   
        self.losses_val.append(float(loss_running))

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

    def validation_plot(self, validation_df, epoch):
        """
        TODO: Description
        Params:
            validation_df (pandas.DataFrame): DataFrame with columns 'uuid', 
                'gt_latitude', 'gt_longitude', 'mo_latitude', 'mo_longitude'. 
                Each row represents one sample in validation dataset.
            epoch (int): Current epoch number.
        """
            
        """
            !!! Important:
            Path of output directory is in:
                self.output_dir
        """
        pass
    
    def make_dir_train(self, output_dir):
        
        desc_dir =  output_dir + '/Descriptive Statistics'
        hist_dir = output_dir + '/Histogram'
        dist_dir = output_dir + '/Distance Density'
        
        if not os.path.exists(desc_dir):
            os.makedirs(desc_dir)  
            
        if not os.path.exists(hist_dir):
            os.makedirs(hist_dir)  
            
        if not os.path.exists(dist_dir):
            os.makedirs(dist_dir)
            
            
    def epoch_analysis(self, output_dir, epoch, data):
        
        epoch = str(epoch)
        
        desc_dir =  output_dir + '/Descriptive Statistics' + '/Epoh_' + epoch
        hist_dir = output_dir + '/Histogram' + '/Epoh_' + epoch
        dens_dir = output_dir + '/Distance Density' + '/Epoh_' + epoch
        
        
        if not os.path.exists(hist_dir):
            os.makedirs(hist_dir)  
            
        distance_histogram(data, hist_dir)
            
        if not os.path.exists(dens_dir):
            os.makedirs(dens_dir) 
            
        distance_density(data, dens_dir)
        
        if not os.path.exists(desc_dir):
            os.makedirs(desc_dir)
            
        distance_descriptive(data, desc_dir) 
            
    
        
        


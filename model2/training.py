from model import Net

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.nn import MSELoss
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from dataset import Dataset
from torch.utils.data import DataLoader
import os.path as osp
import matplotlib
from torchvision import transforms

# Setup backend in order to work inside Docker without having available display.
matplotlib.use('Agg')

# Training parameters, hardcoded for now
lr = 0.001
epoch_num = 25
batch_size = 100
batches_per_step = 1

model = Net()
optimizer = Adam(model.parameters(), lr=lr)
criterion = MSELoss(reduction='sum')
scheduler = StepLR(optimizer, step_size=7, gamma=0.1)
# Dataset parameters
dataset_root = '/workspaces/Dataset'

# Other parameters
num_workers = 4

class Normalize():
    def __init__(self, mean, std):
        self.normalize = transforms.Normalize(mean=mean, std=std)

    def __call__(self, sample):
        sample['image'] = self.normalize(sample['image'])
        return sample

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.25, 1.00)),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

def training(output_dir, train_csv, val_csv):
    losses_train = []
    losses_val = []

    dataset_train = Dataset(dataset_root, csv_file=train_csv, 
        transform=data_transforms['train'])
    dataset_val = Dataset(dataset_root, csv_file=val_csv,
        transform=data_transforms['val'])
    
    output_path_parameters = osp.join(output_dir, 'parameters')
    output_path_losses_train = osp.join(output_dir, 'losses_train.csv')
    output_path_losses_val = osp.join(output_dir, 'losses_val.csv')
    output_path_losses_img = osp.join(output_dir, 'losses_img.png')

    print('Loading training data.')
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, 
        shuffle=True, drop_last=True, num_workers=num_workers)
    print('Loading validation data.')
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, 
        drop_last=False, num_workers=num_workers)

    print(model)

    # Initial Validation step
    model.eval()
    loss_running = torch.tensor(0.)
    print('Initial validation.')
    for batch_ind, batch in enumerate(dataloader_val):
        print(f'..Batch {batch_ind+1}.')
        input = batch['image']
        output_gt = batch['output']
        output_val = model(input)
        loss_running += criterion(output_val, output_gt)
    loss_running /= len(dataset_val)   
    losses_val.append(float(loss_running))
    print(f'..Validation loss: {loss_running}.')

    for epoch in range(epoch_num):
        tmp_lr = optimizer.state_dict()['param_groups'][0]['lr']
        print(f'Epoch {epoch+1} (lr={tmp_lr}).')
        print('.Training.')
        optimizer.zero_grad()
        model.train()

        loss_running = torch.tensor(0.)

        for batch_ind, batch in enumerate(dataloader_train):
            print(f'..Batch {batch_ind+1}.')
            input = batch['image']
            output_gt = batch['output']
            output_train = model(input)
            loss_batch = criterion(output_train, output_gt) 
            loss_batch.backward()
            loss_running += loss_batch.clone().detach()

            if (batch_ind+1)%batches_per_step == 0:
                loss_running /= (batch_size*batches_per_step)
                losses_train.append(float(loss_running))
                print(f'..Train loss: {loss_running}')
                loss_running = torch.tensor(0.)

                optimizer.step()
                optimizer.zero_grad()

        scheduler.step()

        model.eval()
        loss_running = torch.tensor(0.)
        print('.Validation.')
        for batch_ind, batch in enumerate(dataloader_val):
            print(f'..Batch {batch_ind+1}.')
            input = batch['image']
            output_gt = batch['output']
            output_val = model(input)
            loss_running += criterion(output_val, output_gt)
        loss_running /= len(dataset_val)   
        losses_val.append(float(loss_running))
        print(f'..Validation loss: {loss_running}.')

        save_model_parameters(f'{output_path_parameters}_epoch_{epoch+1}')
        print(f'Checkpoint saved to {output_path_parameters}_epoch_{epoch+1}')
        save_losses(output_path_losses_img, output_path_losses_train,
            output_path_losses_val, losses_train, losses_val)
    
    print(f'All outputs saved to {output_dir}.')

def save_model_parameters(filename):
    torch.save(model.state_dict(), filename)

def save_losses(output_path_losses_img, output_path_losses_train, 
    output_path_losses_val, losses_train, losses_val):
    plt.clf()
    plt.plot(np.linspace(0, 1, len(losses_train)), losses_train, 
        label='Train loss')
    plt.plot(np.linspace(0, 1, len(losses_val)), losses_val, 
        label='Validation loss')
    plt.savefig(output_path_losses_img)
    pd.DataFrame({'loss_train': losses_train}).to_csv(output_path_losses_train)
    pd.DataFrame({'loss_val': losses_val}).to_csv(output_path_losses_val)
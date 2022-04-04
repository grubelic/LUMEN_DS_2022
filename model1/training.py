from model import Net

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.nn import MSELoss
from torch.optim import Adam
from dataset import Dataset
from torch.utils.data import DataLoader
import os.path as osp
import matplotlib

# Setup backend in order to work inside Docker without having available display.
matplotlib.use('Agg')

# Training parameters, hardcoded for now
lr = 0.01
epoch_num = 20
batch_size = 100
batches_per_step = 1

model = Net()
optimizer = Adam(model.parameters(), lr=lr)
criterion = MSELoss(reduction='sum')

# Dataset parameters
dataset_root = '/workspaces/Dataset'

# Other parameters
num_workers = 4

def training(output_dir, train_csv, val_csv):
    losses_train = []
    losses_val = []
    dataset_train = Dataset(dataset_root, csv_file=train_csv)
    dataset_val = Dataset(dataset_root, csv_file=val_csv)
    
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
        print(f'Epoch {epoch+1}.')
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
    
    torch.save(model.state_dict(), output_path_parameters)
    print(f'Model saved to {output_path_parameters}.')

    plt.plot(np.linspace(0, 1, len(losses_train)), losses_train, 
        label='Train loss')
    plt.plot(np.linspace(0, 1, len(losses_val)), losses_val, 
        label='Validation loss')
    plt.savefig(output_path_losses_img)
    pd.DataFrame({'loss_train': losses_train}).to_csv(output_path_losses_train)
    pd.DataFrame({'loss_val': losses_val}).to_csv(output_path_losses_val)
    print(f'All outputs saved to {output_dir}.')    
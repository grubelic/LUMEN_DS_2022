# https://www.analyticsvidhya.com/blog/2019/10/building-image-classification-models-cnn-pytorch/#h2_7
# https://towardsdatascience.com/how-to-code-a-simple-neural-network-in-pytorch-for-absolute-beginners-8f5209c50fdd

import csv
from curses import erasechar
from model import Net

from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
from torch.autograd import Variable
from torch.nn import Linear, ReLU, MSELoss, Sequential, Conv2d, \
    MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD
from dataset import Dataset
# training parameters, hardcoded for now
lr = 0.07
epoch_num = 100

model = Net()

optimizer = Adam(model.parameters(), lr=lr)
criterion = MSELoss()

# Dataset parameters
dataset_root = '/workspaces/Dataset'

def training(parameters_output_path, csv_file, dataset_size=None):
    train_losses = []
    val_losses = []
    dataset = Dataset(dataset_root, dataset_size=dataset_size, 
        csv_file=csv_file, val_perc=None)
    print(model)
    for epoch in tqdm(range(epoch_num)):
        loss_train, loss_val = train_step(dataset.train_x, dataset.train_y, 
            dataset.val_x, dataset.val_y)
        
        train_losses.append(loss_train)
        val_losses.append(loss_val)
        print(f'Epoch {epoch}: train loss({loss_train}); val loss ({loss_val})')
    
    torch.save(model.state_dict(), parameters_output_path)
    print(f'Model saved to {parameters_output_path}')
    
    """
    tmp_dataset = Dataset(dataset_root, dataset_size=tmp_dataset_size, 
        csv_file='tmp_data_2.csv', val_perc=None)
    model.eval()
    data_x = Variable(tmp_dataset.train_x)
    data_y = Variable(tmp_dataset.train_y)
    output = model(data_x)
    """
    # lat, lon = torch.permute(output, (1,0))
    # lat, lon = tmp_dataset.denormalize_output(lat, lon)
    # print(lat, lon)


def train_step(train_x, train_y, val_x, val_y):
    """ 
    Returns (train_loss, val_loss)
        train_loss: float
            Training loss in current training step.
        val_loss: float
            Validation loss in current training step.
    """
    model.train()
    x_train, y_train = Variable(train_x), Variable(train_y)
    x_val, y_val = Variable(val_x), Variable(val_y)

    # clearing the Gradients of the model parameters
    optimizer.zero_grad()
    
    # prediction for training and validation set
    output_train = model(x_train)
    output_val = model(x_val)

    # computing the training and validation loss
    loss_train = criterion(output_train, y_train)
    loss_val = criterion(output_val, y_val)

    # computing the updated weights of all the model parameters
    loss_train.backward()
    optimizer.step()

    return loss_train.item(), loss_val.item()

def inference(parameters_path, csv_file, dataset_size=None):
    dataset = Dataset(dataset_root, dataset_size=dataset_size, 
        csv_file=csv_file, val_perc=None)
    model.load_state_dict(torch.load(parameters_path))
    model.eval()
    data_x = dataset.train_x
    data_y = dataset.train_y
    output = model(data_x)
    output = output.detach().numpy()
    output = dataset.denormalize_output(output)
    data_y = data_y.numpy()
    data_y = dataset.denormalize_output(data_y)
    print('Total MSE:', ((output-data_y)**2).sum(axis=1).mean())

if __name__ == "__main__":
    training('model1_parameters_2', csv_file='tmp_data_1.csv', dataset_size=100)
    inference('model1_parameters_2', csv_file='tmp_data_1.csv', dataset_size=100)
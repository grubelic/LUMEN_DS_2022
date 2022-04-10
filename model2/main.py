import csv
from curses import erasechar
from inference import evaluate_model
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
from torch.utils.data import DataLoader
import argparse
import sys
from training import training
from inference import evaluate_model
import os
import os.path as osp

"""
Usage example (training):
    ~ Create new directory for output storage, for example 'model-1_output'.
    ~ Run in command line:
        python3 main.py \
            --action=train \
            --output_dir=model-1_output \
            --train_csv=2022-mar-31_data_train.csv \
            --val_csv=2022-mar-31_data_val.csv

Usage example (inference):
    ~ TODO
"""


parser = argparse.ArgumentParser()
parser.add_argument('--action', required=True, choices=['train', 'inference'],
    help="Action to be performed.")
parser.add_argument('--train_csv', required=False, 
    help='Filename of the training csv.')
parser.add_argument('--val_csv', required=False,
    help='Filename of the validation csv.')
parser.add_argument('--output_dir', required=False,
    help='Path to existing directory in which all the output will be saved.')
parser.add_argument('--parameters_path', required=False,
    help='Path to model parameters.')

def main(args):
    args = parser.parse_args(args)
    if(args.action == 'train'):
        assert isinstance(args.output_dir, str) and osp.isdir(args.output_dir),\
            'Please specify existing directory with --output_dir.'
        training(args.output_dir, args.train_csv, args.val_csv)
    else:
        assert isinstance(args.output_dir, str) and osp.isdir(args.output_dir),\
            'Please specify existing directory with --output_dir.'
        evaluate_model(args.parameters_path, args.output_dir, args.val_csv)
    

if __name__ == "__main__":
    main(sys.argv[1:])
    """training(
        parameters_output_path='/workspaces/LUMEN_DS_2022/model1/parameters_2',
        losses_output_path='/workspaces/LUMEN_DS_2022/model1/losses_2', 
        train_csv='2022-mar-31_data_train.csv', 
        val_csv='2022-mar-31_data_val.csv')
    """
    """
    training(
        parameters_output_path='/workspaces/LUMEN_DS_2022/model1/parameters_2',
        losses_output_path='/workspaces/LUMEN_DS_2022/model1/losses_2', 
        train_csv='2022-mar-31_data-debug_train.csv', 
        val_csv='2022-mar-31_data-debug_val.csv')
    """
    # inference('model1_parameters_2', csv_file='tmp_data_1.csv', dataset_size=100)
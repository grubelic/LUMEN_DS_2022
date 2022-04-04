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
from tqdm import tqdm
from vincenty import vincenty


# Setup backend in order to work inside Docker without having available display.
matplotlib.use('Agg')

# Inference parameters
batch_size = 100
model = Net()
criterion = MSELoss(reduction='sum')
dataset_root = '/workspaces/Dataset'
num_workers = 4

def evaluate_model(parameters_path, output_dir, val_csv):
    model.load_state_dict(torch.load(parameters_path))
    dataset_val = Dataset(dataset_root, val_csv)

    model.eval()

    loss_running = torch.tensor(0.)

    output = pd.DataFrame({
        'uuid': [],
        'gt_latitude': [],
        'gt_longitude': [],
        'mo_latitude': [],
        'mo_longitude': []
        })

    print('Model inference.')
    for sample_ind, sample in tqdm(enumerate(dataset_val)):
        input = torch.unsqueeze(sample['image'], 0)
        output_gt = torch.unsqueeze(sample['output'], 0)
        output_val = model(input)
        output_val = dataset_val.denormalize_output(output_val)
        output = output.append({
            'uuid': dataset_val.data.iloc[sample_ind]['uuid'],
            'gt_latitude': dataset_val.data.iloc[sample_ind]['latitude'],
            'gt_longitude': dataset_val.data.iloc[sample_ind]['longitude'],
            'mo_latitude': float(output_val[0][0]),
            'mo_longitude': float(output_val[0][1])
        }, ignore_index=True)
    
    apply_metrics(output)
    output.to_csv(osp.join(output_dir, 'predictions.csv'), index=False)
    
def apply_metrics(data):
    """
    Apply different metrics comparing model output (mo) and ground truth (gt).
    
    Params:    
        data: pd.DataFrame with columns 'gt_latitude', 'gt_longitude', 
            'mo_latitude', 'mo_longitude'
    """

    earth_distances = []
    for ind in range(len(data)):
        gt = tuple(data.iloc[ind][['gt_latitude', 'gt_longitude']].values)
        mo = tuple(data.iloc[ind][['mo_latitude', 'mo_longitude']].values)
        earth_distances.append(vincenty(gt, mo))
    
    data['distance'] = earth_distances
    print(data['distance'].describe())


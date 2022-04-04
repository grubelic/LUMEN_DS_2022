import pandas as pd
import os
import os.path as osp
from sklearn.model_selection import train_test_split
import torch
from PIL import Image
import numpy as np
from tqdm import tqdm

class Dataset(torch.utils.data.Dataset):
    lat_mean = 45.125808030990285
    lat_std = 0.8563826794590563
    lon_mean = 16.402045631961176
    lon_std = 1.2739828910340654
    out_mean = torch.tensor([lat_mean, lon_mean], dtype=torch.float32)
    out_std = torch.tensor([lat_std, lon_std], dtype=torch.float32)

    def __init__(self, dataset_root, csv_file):
        self.dataset_root = dataset_root
        self.csv_file = csv_file

        self.data = pd.read_csv(osp.join(dataset_root, csv_file))
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        uuid = self.data.iloc[index]['uuid']
        latitude = self.data.iloc[index]['latitude']
        longitude = self.data.iloc[index]['longitude']
        img_north = Image.open(
            osp.join(self.dataset_root, 'data', uuid, '0.jpg'))
        img_north = img_north.resize((80, 80))
        img_north_np = np.array(img_north, dtype='float32')
        img_north_tsr = torch.tensor(img_north_np)
        img_north_tsr = torch.permute(img_north_tsr, (2, 0, 1))
        img_north_tsr /= 255.

        return {
            'image': img_north_tsr,
            'output': self.normalize_output(
                torch.tensor([latitude, longitude], dtype=torch.float32))
        }
        
    def normalize_output(self, output):
        return (output-self.out_mean)/self.out_std

    def denormalize_output(self, output):
        return output*self.out_std+self.out_mean

        

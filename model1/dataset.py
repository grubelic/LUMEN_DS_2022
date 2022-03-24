import pandas as pd
import os
import os.path as osp
from sklearn.model_selection import train_test_split
import torch
from PIL import Image
import numpy as np
from tqdm import tqdm

class Dataset:
    lat_mean = 45.125808030990285
    lat_std = 0.8563826794590563
    lon_mean = 16.402045631961176
    lon_std = 1.2739828910340654

    def __init__(self, dataset_root, val_perc=0.2, dataset_size=None,
        csv_file='data.csv'):

        self.dataset_root = dataset_root
        self.all_data = pd.read_csv(osp.join(self.dataset_root, csv_file))
        
        if dataset_size is None:
            dataset_size = len(self.all_data)

        images_all = []
        for uuid in tqdm(self.all_data['uuid'][:dataset_size]):
            img = Image.open(osp.join(self.dataset_root, 'data', uuid, '0.jpg'))
            img = img.resize((80, 80))
            img_np = np.expand_dims(np.array(img, dtype='float32'), 0)
            img_tsr = torch.tensor(img_np)
            img_tsr = torch.permute(img_tsr, (0, 3, 1, 2))
            img_tsr /= 255.
            images_all.append(img_tsr)

        all_data_input = torch.vstack(images_all)
        lat = np.array(self.all_data['latitude'][:dataset_size], 
            dtype='float32')
        lon = np.array(self.all_data['longitude'][:dataset_size], 
            dtype='float32')

        all_data_output = torch.tensor(np.hstack([
            np.expand_dims((lat-self.lat_mean)/self.lat_std, 1),
            np.expand_dims((lon-self.lon_mean)/self.lon_std, 1)
        ]))

        if val_perc is None:
            self.train_x = all_data_input
            self.train_y = all_data_output
            self.val_x = all_data_input # TODO: ovo treba maknuti
            self.val_y = all_data_output  # TODO: ovo treba maknuti
        else:
            self.train_x, self.val_x, \
                self.train_y, self.val_y = train_test_split(all_data_input,
                    all_data_output, test_size=val_perc)
    
    def denormalize_output(self, output):
        #return lat*self.lat_std+self.lat_mean, lon*self.lon_std+self.lon_mean
        return output*np.array([[self.lat_std, self.lon_std]]) + \
            np.array([[self.lat_mean, self.lon_mean]])

        

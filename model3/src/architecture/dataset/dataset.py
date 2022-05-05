import pandas as pd
import os
import os.path as osp
from sklearn.model_selection import train_test_split
import torch
from PIL import Image
import numpy as np
from torchvision import transforms


class Dataset(torch.utils.data.Dataset):
    lat_mean = 45.125808030990285
    lat_std = 0.8563826794590563
    lon_mean = 16.402045631961176
    lon_std = 1.2739828910340654

    def __init__(self, dataset_root, csv_file, transform=None,
        image_directions=['N', 'E', 'S', 'W'], device=["cpu"]):
        self.out_mean = torch.tensor(
            [Dataset.lat_mean, Dataset.lon_mean],
            dtype=torch.float32, device=device[0]
        )
        self.out_std = torch.tensor(
            [Dataset.lat_std, Dataset.lon_std],
            dtype=torch.float32, device=device[0]
        )

        self.dataset_root = dataset_root
        self.csv_file = csv_file

        self.data = pd.read_csv(osp.join(dataset_root, csv_file))
        self.image_directions = image_directions
        self.contains_gt = 'latitude' in self.data and \
            'longitude' in self.data

        if transform is not None:
            self.transform = transform 
        else:
            self.transform = transforms.Compose([])
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        uuid = self.data.iloc[index]['uuid']

        sample = {
            'uuid': uuid,
            'image_N': self.transform(self.load_image(uuid, 'N')) \
                if 'N' in self.image_directions else torch.zeros(1),
            'image_E': self.transform(self.load_image(uuid, 'E')) \
                if 'E' in self.image_directions else torch.zeros(1), 
            'image_S': self.transform(self.load_image(uuid, 'S')) \
                if 'S' in self.image_directions else torch.zeros(1),
            'image_W': self.transform(self.load_image(uuid, 'W')) \
                if 'W' in self.image_directions else torch.zeros(1),
        }
        
        if self.contains_gt:
            latitude = self.data.iloc[index]['latitude']
            longitude = self.data.iloc[index]['longitude']
            sample['output'] = self.normalize_output(
                    torch.tensor([latitude, longitude], dtype=torch.float32)
                )

        return sample

    def load_image(self, uuid, direction):
        directions = {
            'N': '0.jpg',
            'E': '90.jpg',
            'S': '180.jpg',
            'W': '270.jpg'
        }
        assert direction in directions, "Parameter "\
            "direction should be one of the chars 'N', 'E', 'S', 'W'."

        img = Image.open(
            osp.join(self.dataset_root, 'data', uuid, directions[direction]))
        img = torch.tensor(np.array(img, dtype='float32'))
        img = torch.permute(img, (2, 0, 1))
        img /= 255.
        return img

    def normalize_output(self, output):
        return (output-self.out_mean)/self.out_std

    def denormalize_output(self, output, device="cuda:0"):
        return output*self.out_std+self.out_mean

import matplotlib
from sklearn.utils import shuffle
from dataset import Dataset
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

def visualize_one_sample(sample):
    img = sample['image']
    img = torch.permute(img, (1, 2, 0))
    plt.imshow(img)
    plt.show()

def test_dataset():
    dataset = Dataset(
        dataset_root='/workspaces/Dataset/', 
        csv_file='2022-mar-31_data-debug_train.csv')
    print(len(dataset))
    visualize_one_sample(dataset[1])

def visualize_batch(batch):
    imgs = batch['image']
    disp_img = torch.permute(imgs, (0, 2, 3, 1))
    disp_img = torch.flatten(disp_img, start_dim=0, end_dim=1)
    plt.imshow(disp_img)
    plt.show()

def test_dataloader():
    dataset = Dataset(
        dataset_root='/workspaces/Dataset/', 
        csv_file='2022-mar-31_data-debug_train.csv')
    dl = DataLoader(dataset, batch_size=6, shuffle=True, drop_last=True)
    for batch_ind, batch in enumerate(dl):
        print(f'Batch {batch_ind}')
        visualize_batch(batch)

if __name__ == '__main__':
    #test_dataset()
    test_dataloader()
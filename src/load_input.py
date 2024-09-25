import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os
import numpy as np
import cv2

class HomographyDataset(Dataset):
    def __init__(self, dataset_dir, num_pairs=1000, transform=None):
        self.dataset_dir = dataset_dir
        self.num_pairs = num_pairs
        self.transform = transform

    def __len__(self):
        return self.num_pairs

    def __getitem__(self, idx):
        img1_path = os.path.join(self.dataset_dir, f'image_{idx}_original.png')
        img2_path = os.path.join(self.dataset_dir, f'image_{idx}_transformed.png')
        H_path = os.path.join(self.dataset_dir, f'homography_{idx}.txt')

        img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
        H = np.loadtxt(H_path).flatten()[:8]

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        img_pair = torch.cat((img1, img2), dim=0)
        H = torch.tensor(H, dtype=torch.float32)

        return img_pair, H

def load_transform_input(dataset_dir):
    # Transform to normalize the images
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load the dataset
    #dataset_dir = 'homography_dataset'
    dataset = HomographyDataset(dataset_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    return dataset, dataloader


from swin_UNet import SwinUNet
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from PIL import Image
import os
from torchvision import transforms


from centroid import pipeline

class Amida13Dataset():
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(os.path.join(data_dir, 'images')) if f.endswith('.jpg') or f.endswith('.png')]


    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        image_path = os.path.join(self.data_dir, 'images', image_name)
        mask_path = os.path.join(self.data_dir, 'masks', image_name.replace('.jpg', '.png'))
        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        return image, mask


# Define transforms for data augmentation and normalization
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
])

# Create an instance of the custom dataset
data_dir = r'AMIDA13'
dataset = Amida13Dataset(data_dir, transform=transform)


# Create data loaders for train, validation, and test sets
batch_size = 64

test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SwinUNet(224,224,1,32,1,3,4).to(device)
model.load_state_dict(torch.load("model/model v4.pth", map_location=device))









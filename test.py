from swin_UNet import SwinUNet
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image
import os
from torchvision import transforms
from torch.utils.data import random_split

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


# Define the sizes of train, validation, and test sets
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size

# Split the dataset into train, validation, and test sets
train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])


# Create data loaders for train, validation, and test sets
batch_size = 64

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

# load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SwinUNet(224,224,1,32,1,3,4).to(device)
model.load_state_dict(torch.load("model/model v4.pth", map_location=device))

# test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)
# fig, ax = plt.subplots(3, 4, figsize=(10, 8))
# with torch.no_grad():
#   for i in range(3):  # Adjust the range to visualize more rows
#     # Get a new batch of images and masks
#     x_og, y_og = next(iter(test_loader))
#     x = x_og[0]
#     y = y_og[0]

#     # Visualize image, mask, prediction, and thresholded prediction
#     ax[i, 0].imshow(x.squeeze(0).squeeze(0), cmap='gray')
#     ax[i, 0].set_title('Image')
#     ax[i, 1].imshow(y.squeeze(0).squeeze(0), cmap='gray')
#     ax[i, 1].set_title('Mask')

#     x_og = x_og.to(device)
#     out = model(x_og[:1])
#     out = nn.Sigmoid()(out)
#     out = out.squeeze(0).squeeze(0).cpu()

#     apply_threshold,_,_ = pipeline((out.numpy()*255).astype(np.uint8), 0.25, 10, 150)

#     # save the images
#     # cv2.imwrite(f"image_{i}.png", x.squeeze(0).squeeze(0).numpy()*255)
#     # cv2.imwrite(f"mask_{i}.png", y.squeeze(0).squeeze(0).numpy()*255)
#     # cv2.imwrite(f"prediction_{i}.png", (out.numpy()*255).astype(np.uint8))
#     # cv2.imwrite(f"thresholded_prediction_{i}.png", apply_threshold)


#     ax[i, 2].imshow(out, cmap='gray')
#     ax[i, 2].set_title('Prediction')

#     ax[i, 3].imshow(apply_threshold, cmap='gray')
#     ax[i, 3].set_title('Thresholded Prediction')


# plt.show()







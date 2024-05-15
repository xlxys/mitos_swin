from swin_UNet import SwinUNet
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import os
import cv2
import numpy as np

class Amida13TestDataset():
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.image_files = [f for f in os.listdir(os.path.join(data_dir, 'images')) if f.endswith('.jpg') or f.endswith('.png')]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        image_path = os.path.join(self.data_dir, 'images', image_name)
        mask_path = os.path.join(self.data_dir, 'masks', image_name)
        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        return image, mask

# Define transforms for data augmentation and normalization
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Create an instance of the custom dataset
data_dir = r'test/output_dir'
dataset = Amida13TestDataset(data_dir, transform=transform)

# Verify the dataset
print(len(dataset))
print(dataset[0][0].shape)
print(dataset[0][1].shape)

# Create data loaders for test set
batch_size = 64
test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SwinUNet(224,224,1,32,1,3,4).to(device)
model.load_state_dict(torch.load("model/model v4.pth", map_location=device))
model.eval()

# Create output directory if it does not exist
output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)

# Iterate over the test set and make predictions
for i, (images, _) in enumerate(test_loader):
    images = images.to(device)
    with torch.no_grad():
        pred_masks = model(images)
    pred_masks = torch.sigmoid(pred_masks)
    for j in range(pred_masks.size(0)):
        pred_mask = pred_masks[j].squeeze().cpu().numpy()
        pred_mask = (pred_mask * 255).astype(np.uint8)
        cv_file_name = os.path.join(output_dir, dataset.image_files[i * batch_size + j])
        cv2.imwrite(cv_file_name, pred_mask)

print("Prediction completed.")



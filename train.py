import torch
import torch.nn as nn
from swin_UNet import SwinUNet
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from PIL import Image
import os


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
    # transforms.Grayscale(num_output_channels=1),
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SwinUNet(224,224,1,32,1,3,4).to(device)

for p in model.parameters():
    if p.dim() > 1:
            nn.init.kaiming_uniform_(p)

optimizer = torch.optim.Adam(model.parameters())
loss_fn = nn.BCEWithLogitsLoss()

max_epochs=3


# Loop over epochs
for epoch in range(max_epochs):
    # Training
    print(f"Epoch {epoch+1}/{max_epochs}:")
    train_loss = 0.0
    for i, (local_batch, local_labels) in enumerate(train_loader, 1):
        # Transfer to GPU
        local_batch, local_labels = local_batch.to(device), local_labels.to(device)

        # Model computations
        optimizer.zero_grad()
        out = model(local_batch)
        loss = loss_fn(out, local_labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        # Print training progress
        if i % 100 == 0:  # Print every 100 batches
            print(f"  Batch {i}/{len(train_loader)}, Loss: {loss.item():.4f}")

    # Calculate average training loss
    train_loss /= len(train_loader)

    # Validation
    with torch.set_grad_enabled(False):
        val_loss = 0.0
        for i, (local_batch, local_labels) in enumerate(val_loader, 1):
            # Transfer to GPU
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)

            # Model computations
            out = model(local_batch)
            loss = loss_fn(out, local_labels)
            val_loss += loss.item()

    # Calculate average validation loss
    val_loss /= len(val_loader)

    # Print epoch summary
    print(f"Epoch {epoch+1}/{max_epochs}: Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

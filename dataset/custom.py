import torch
import numpy as np
import os
from torch.utils.data import Dataset
from PIL import Image

class ImageDataset(Dataset):
    """
    Custom PyTorch Dataset class to load images and corresponding .npy representations.
    Works when train and val images are stored in separate folders.
    """

    def __init__(self, image_dir, rep_dir, transform=None):
        """
        Args:
            image_dir (str): Path to the directory containing images.
            rep_dir (str): Path to the directory containing .npy representation files.
            transform (callable, optional): Transformations to apply to images.
        """
        self.image_dir = image_dir
        self.rep_dir = rep_dir
        self.transform = transform

        # Get list of image file paths
        self.image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) 
                            if f.endswith(('jpg', 'jpeg', 'png','JPG', 'JPEG', 'PNG'))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Get image path and filename
        img_path = self.image_paths[idx]
        img_name = os.path.basename(img_path)  # Extract filename only

        # Load image
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # Load corresponding .npy representation
        rep_name = os.path.splitext(img_name)[0] + ".npy"
        if self.rep_dir is not None:  
            rep_path = os.path.join(self.rep_dir, rep_name)
            
            try:
                representation = np.load(rep_path)  # Load as NumPy array
                representation = torch.tensor(representation, dtype=torch.float32)  # Convert to tensor
            except FileNotFoundError:
                representation = torch.zeros(1)  # Default if representation file is missing
                print(f"Representation file not found for {img_name}")
        else:
            representation = torch.zeros(1)

        return image, representation
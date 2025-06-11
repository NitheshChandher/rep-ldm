import torch
import numpy as np
import os
from torch.utils.data import Dataset
from PIL import Image


class PCAMDataset(Dataset):
    """
    Custom Dataset for PCAM. Loads images, corresponding representations, and labels (0 or 1).
    """

    def __init__(self, image_dir, rep_dir, transform=None):
        """
        Args:
            image_dir (str): Path to pcam_images/{train,val}
            rep_dir (str): Path to pcam_representations/{train,val}
            transform (callable, optional): Optional transform to be applied on images.
        """
        self.image_dir = image_dir
        self.rep_dir = rep_dir
        self.transform = transform

        # Collect all image paths recursively
        self.image_paths = []
        for root, _, files in os.walk(image_dir):
            for file in files:
                if file.lower().endswith(('jpg', 'jpeg', 'png')):
                    self.image_paths.append(os.path.join(root, file))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        # Load image
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Build representation path
        relative_path = os.path.relpath(img_path, self.image_dir)  # e.g., "0/x.jpg"
        rep_relative_path = os.path.splitext(relative_path)[0] + ".npy"  # e.g., "0/x.npy"
        rep_path = os.path.join(self.rep_dir, rep_relative_path)

        try:
            representation = np.load(rep_path)
            representation = torch.tensor(representation, dtype=torch.float32)
        except FileNotFoundError:
            representation = torch.zeros(1)
            print(f"Representation file not found for {relative_path}")

        # Extract label from the first part of relative path (folder name)
        label_str = relative_path.split(os.sep)[0]  # e.g., "0" or "1"
        label = int(label_str)  # convert to integer (0 or 1)
        label = torch.tensor(label, dtype=torch.long)

        return image, representation, label
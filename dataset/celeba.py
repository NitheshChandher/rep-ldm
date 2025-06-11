import torch
import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset
from PIL import Image

class CelebA(Dataset):
    def __init__(self, root, rep_root, split="train", transform=None):
        """
        Args:
            root (str): Path to CelebA dataset root.
            rep_root (str): Path to the folder containing .npy representation files.
            split (str): Dataset split - "train", "valid", or "test".
            transform (callable, optional): Transformations to apply to the images.
        """
        self.root = root
        self.rep_root = rep_root  # Path to representations
        self.transform = transform

        # Path to images, attribute labels, and split file
        self.img_dir = os.path.join(root, "img_align_celeba")
        self.attr_path = os.path.join(root, "list_attr_celeba.txt")
        self.split_path = os.path.join(root, "list_eval_partition.txt")

        # Load attributes and splits
        self.attrs = pd.read_csv(self.attr_path, sep="\s+", skiprows=1)
        self.splits = pd.read_csv(self.split_path, sep="\s+", header=None, names=["filename", "split"])

        # Map split names to numerical values
        split_map = {"train": 0, "valid": 1, "test": 2}
        self.splits = self.splits[self.splits["split"] == split_map[split]]
        self.image_files = self.splits["filename"].tolist()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]

        # Load Image
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")  # Convert to RGB

        if self.transform:
            image = self.transform(image)

        # Load Corresponding .npy Representation
        rep_name = os.path.splitext(img_name)[0] + ".npy"  # Change extension to .npy
        rep_path = os.path.join(self.rep_root, rep_name)
        
        try:
            representation = np.load(rep_path)  # Load as NumPy array
            representation = torch.tensor(representation, dtype=torch.float32)  # Convert to tensor
        except FileNotFoundError:
            representation = torch.zeros(1)  # Default value if file is missing
            print(f"Representation file not found for {img_name}")

        # Get attributes for the image
        attr_values = self.attrs.loc[img_name].values.astype(int)
        attr_values = torch.tensor(attr_values, dtype=torch.float32)

        return image, representation, attr_values

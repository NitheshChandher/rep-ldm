import torch
import numpy as np
import os
from torch.utils.data import Dataset

class RepresentationDataset(Dataset):
    """
    Custom PyTorch Dataset to load only .npy representations from a directory.
    """

    def __init__(self, rep_dir):
        """
        Args:
            rep_dir (str): Path to the directory containing .npy representation files.
        """
        self.rep_dir = rep_dir

        # List all .npy files in the representation directory
        self.rep_paths = [os.path.join(rep_dir, f) for f in os.listdir(rep_dir)
                          if f.endswith('.npy')]

    def __len__(self):
        return len(self.rep_paths)

    def __getitem__(self, idx):
        # Get representation path
        rep_path = self.rep_paths[idx]
        rep_name = os.path.basename(rep_path)

        try:
            # Load and convert to tensor
            representation = np.load(rep_path)
            representation = torch.tensor(representation, dtype=torch.float32)
        except Exception as e:
            representation = torch.zeros(1)
            print(f"Error loading {rep_name}: {e}")

        return representation

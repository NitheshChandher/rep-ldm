import os
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataset.celeba import CelebA
from dataset.ffhq import FFHQ
from dataset.pcam import PCAMDataset
from dataset.representation import RepresentationDataset

def load_and_prepare_dataset(dataset_name, batch_size=16, img_size=(256, 256), data_dir=None, rep_dir=None, transform=True, shuffle=True):
    """
    Function to load a dataset, apply transformations, and return a PyTorch DataLoader.

    Args:
    - dataset_name (str): The name of the dataset to load from Hugging Face.
    - batch_size (int): The batch size for the DataLoader (default is 16).
    - img_size (tuple): The desired image size (default is 256x256).
    - cache_dir (str, optional): Directory to cache the dataset.

    Returns:
    - train_dataloader: PyTorch DataLoader object for the training dataset with the specified transformations.
    - val_dataloader: PyTorch DataLoader object for the validation dataset with the specified transformations.
    """
    if transform is True: 
        transform = transforms.Compose([
                transforms.Resize(img_size),    # Resize images to 256x256
                transforms.ToTensor(),          # Convert images to PyTorch tensors
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])

    else:
         transform = transforms.Compose([
                transforms.Resize(img_size),    # Resize images to 256x256
                transforms.ToTensor(),
            ])
          
    if data_dir is not None:
        if dataset_name == "celeba":
            train_dataset = CelebA(root=data_dir, rep_root=rep_dir, split='train', transform=transform)
            val_dataset = CelebA(root=data_dir, rep_root=rep_dir, split='valid', transform=transform)

            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
            val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        elif dataset_name == "ffhq":
            train_dir = os.path.join(data_dir, "train")
            val_dir = os.path.join(data_dir, "test")
            rep_train_dir = os.path.join(rep_dir, "train")
            rep_val_dir = os.path.join(rep_dir, "test")
            train_dataset = FFHQ(image_dir=train_dir, rep_dir=rep_train_dir, transform=transform)
            val_dataset = FFHQ(image_dir=val_dir, rep_dir=rep_val_dir, transform=transform)  
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
            val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        elif dataset_name == "pcam":
            train_dir = os.path.join(data_dir, "train")
            val_dir = os.path.join(data_dir, "val")
            rep_train_dir = os.path.join(rep_dir, "train")
            rep_val_dir = os.path.join(rep_dir, "val")
            train_dataset = PCAMDataset(image_dir=train_dir, rep_dir=rep_train_dir, transform=transform)
            val_dataset = PCAMDataset(image_dir=val_dir, rep_dir=rep_val_dir, transform=transform)
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
            val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        else:
            raise ValueError("Dataset Not Available! Incorrect input config for dataset directory")
    else:
        if dataset_name == "ffhq":
            rep_train_dir = os.path.join(rep_dir, "train")
            rep_val_dir = os.path.join(rep_dir, "test")

            train_dataset = RepresentationDataset(rep_dir=rep_train_dir)
            val_dataset = RepresentationDataset(rep_dir=rep_val_dir)
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
            val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        else:
            raise ValueError("Representations Not Available! Incorrect input config for representation directory")

    return train_dataloader, val_dataloader
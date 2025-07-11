from torch.utils.data import Dataset
import torch
from pathlib import Path
from skimage.io import imread
from skimage.color import gray2rgb
import numpy as np
import torchvision as tv
import os

train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]


class ChallengeDataset(Dataset):
    def __init__(self, data, mode):
        """
        Args:
            data: pandas DataFrame containing image paths and labels
            mode: 'train' or 'val' to determine which transforms to apply
        """
        self.data = data
        self.mode = mode

        # Define transforms based on mode
        if mode == 'train':
            # Training transforms with data augmentation
            self.transform = tv.transforms.Compose([
                tv.transforms.ToPILImage(),
                tv.transforms.RandomHorizontalFlip(p=0.5),
                tv.transforms.RandomVerticalFlip(p=0.5),
                tv.transforms.RandomRotation(10),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(mean=train_mean, std=train_std)
            ])
        else:
            # Validation transforms without augmentation
            self.transform = tv.transforms.Compose([
                tv.transforms.ToPILImage(),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(mean=train_mean, std=train_std)
            ])

    def __len__(self):
        """Returns the length of the dataset"""
        return len(self.data)

    def __getitem__(self, index):
        """
        Returns a sample from the dataset
        Args:
            index: index of the sample
        Returns:
            tuple: (image, label) where image and label are torch.Tensors
        """
        # Get the row from the dataframe
        row = self.data.iloc[index]

        # Get the image path - could be 'filename', 'path', or first column
        if 'filename' in self.data.columns:
            img_path = row['filename']
        elif 'path' in self.data.columns:
            img_path = row['path']
        else:
            # Use the first column as the image path
            img_path = row.iloc[0]

        # Convert to Path object for easier handling
        img_path = Path(img_path)

        # If it's already an absolute path and exists, use it
        if img_path.is_absolute() and img_path.exists():
            final_path = img_path
        else:
            # Get the directory where data.py is located
            data_py_dir = Path(__file__).parent

            # Try to find the image file
            # First, try relative to data.py's directory
            final_path = data_py_dir / img_path

            # If not found and path starts with 'images/', try without it
            if not final_path.exists() and str(img_path).startswith('images/'):
                final_path = data_py_dir / img_path.name

            # If still not found, try in the images subdirectory
            if not final_path.exists():
                final_path = data_py_dir / 'images' / img_path.name

            if not final_path.exists():
                raise FileNotFoundError(f"Could not find image file: {img_path}")

        # Load the image
        image = imread(str(final_path))

        # Convert grayscale to RGB
        if len(image.shape) == 2:  # Grayscale image
            image = gray2rgb(image)

        # Apply transforms
        image = self.transform(image)

        # Get labels - try different possible column names
        crack_label = 0
        inactive_label = 0

        if 'crack' in self.data.columns:
            crack_label = row['crack']
        elif 'Crack' in self.data.columns:
            crack_label = row['Crack']

        if 'inactive' in self.data.columns:
            inactive_label = row['inactive']
        elif 'Inactive' in self.data.columns:
            inactive_label = row['Inactive']

        labels = torch.tensor([crack_label, inactive_label], dtype=torch.float32)

        return image, labels
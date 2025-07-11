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

        # Handle relative paths - try multiple possible locations
        if not Path(img_path).is_absolute():
            # Get the directory where the script is being run from
            script_dir = Path.cwd()

            # Try different possible base directories
            possible_paths = [
                script_dir / img_path,  # Current directory
                script_dir / 'images' / Path(img_path).name,  # images subdirectory with just filename
                script_dir / '..' / img_path,  # Parent directory
                script_dir / '..' / 'images' / Path(img_path).name,  # Parent/images directory
                script_dir / 'src_to_implement' / img_path,  # src_to_implement directory
                script_dir / 'src_to_implement' / '..' / img_path,  # From src_to_implement up one level
                Path(img_path),  # Try as is
            ]

            img_path_found = None
            for possible_path in possible_paths:
                if possible_path.exists():
                    img_path_found = str(possible_path)
                    break

            if img_path_found is None:
                # Print debug information
                print(f"Could not find image file: {img_path}")
                print(f"Current working directory: {script_dir}")
                print(f"Tried paths:")
                for p in possible_paths:
                    print(f"  - {p} (exists: {p.exists()})")

                # List contents of current directory for debugging
                print("Contents of current directory:")
                for item in script_dir.iterdir():
                    print(f"  - {item}")

                raise FileNotFoundError(f"Could not find image file: {img_path}")

            img_path = img_path_found

        # Load the image
        image = imread(img_path)

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
from torch.utils.data import Dataset
import torch
from PIL import Image
import torchvision as tv
import numpy as np

train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]


class ChallengeDataset(Dataset):
    def __init__(self, data, mode):
        self.data = data
        self.mode = mode

        if mode == 'train':
            self.transform = tv.transforms.Compose([
                tv.transforms.RandomHorizontalFlip(p=0.5),
                tv.transforms.RandomVerticalFlip(p=0.5),
                tv.transforms.RandomRotation(15),
                tv.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                tv.transforms.RandomResizedCrop(size=(300, 300), scale=(0.8, 1.0)),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(mean=train_mean, std=train_std)
            ])
        else:
            self.transform = tv.transforms.Compose([
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(mean=train_mean, std=train_std)
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]

        img_path = row['filename']
        image = Image.open(str(img_path))

        # Konvertiere Graustufenbilder zu RGB
        if image.mode == 'L':
            image = image.convert('RGB')
        elif image.mode == 'RGBA':
            image = image.convert('RGB')

        image = self.transform(image)

        crack_label = row['crack']
        inactive_label = row['inactive']

        labels = torch.tensor([crack_label, inactive_label], dtype=torch.float32)

        return image, labels
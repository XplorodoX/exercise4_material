import torch as t
from data import ChallengeDataset
from model import ResNet
from losses import FocalLoss
import pandas as pd
from sklearn.model_selection import train_test_split
from torch_lr_finder import LRFinder

if __name__ == '__main__':
    # Load and prepare the data, same as in your train.py
    data = pd.read_csv('data.csv', sep=';')
    train_data, _ = train_test_split(data, test_size=0.2, random_state=42,
                                     stratify=data[['crack', 'inactive']])
    train_data = train_data.reset_index(drop=True)
    train_dataset = ChallengeDataset(train_data, 'train')
    train_loader = t.utils.data.DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # Initialize the model, criterion, and optimizer
    model = ResNet()
    criterion = FocalLoss(alpha=1.0, gamma=2.0)
    optimizer = t.optim.AdamW(model.parameters(), lr=1e-7, weight_decay=1e-3) # Start with a very small LR

    # Set up the Learning Rate Finder
    lr_finder = LRFinder(model, optimizer, criterion, device="cuda" if t.cuda.is_available() else "cpu")

    # Run the LR finder
    print("Running Learning Rate Finder...")
    lr_finder.range_test(train_loader, end_lr=1, num_iter=100)

    # Plot the results
    lr_finder.plot()

    # Reset the model and optimizer to their initial states
    lr_finder.reset()

    print("\nLR Finder complete. Check the plot to find the optimal learning rate.")
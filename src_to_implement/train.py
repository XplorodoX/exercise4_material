import torch as t
from data import ChallengeDataset
from trainer import Trainer
from matplotlib import pyplot as plt
import numpy as np
from model import ResNet
import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Create checkpoints directory if it doesn't exist
os.makedirs('checkpoints', exist_ok=True)

# Load the data from the csv file and perform a train-test-split
data = pd.read_csv('data.csv', sep=';')  # Assuming semicolon separator based on test file

# Split the data into train and validation sets
train_data, val_data = train_test_split(data, test_size=0.2, random_state=42,
                                      stratify=data[['crack', 'inactive']])

# Reset indices for the split datasets
train_data = train_data.reset_index(drop=True)
val_data = val_data.reset_index(drop=True)

# Set up data loading for the training and validation set
batch_size = 32
num_workers = 4

train_dataset = ChallengeDataset(train_data, 'train')
val_dataset = ChallengeDataset(val_data, 'val')

train_loader = t.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True
)

val_loader = t.utils.data.DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True
)

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")

# Create an instance of our ResNet model
model = ResNet()

# Set up a suitable loss criterion for multi-label classification
# BCELoss is appropriate since we have sigmoid activation and multi-label problem
criterion = t.nn.BCELoss()

# Set up the optimizer
learning_rate = 0.001
optimizer = t.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

# Create an object of type Trainer and set its early stopping criterion
trainer = Trainer(
    model=model,
    crit=criterion,
    optim=optimizer,
    train_dl=train_loader,
    val_test_dl=val_loader,
    cuda=True,  # Use GPU if available
    early_stopping_patience=10  # Stop if no improvement for 10 epochs
)

print("Starting training...")

# Train the model
res = trainer.fit(epochs=50)  # Train for maximum 50 epochs

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(np.arange(len(res[0])), res[0], label='train loss', marker='o')
plt.plot(np.arange(len(res[1])), res[1], label='val loss', marker='s')
plt.yscale('log')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('losses.png', dpi=300, bbox_inches='tight')
plt.show()

print("Training completed!")
print(f"Final training loss: {res[0][-1]:.4f}")
print(f"Final validation loss: {res[1][-1]:.4f}")
print(f"Training completed in {len(res[0])} epochs")
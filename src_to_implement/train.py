import torch as t
from data import ChallengeDataset
from trainer import Trainer
from matplotlib import pyplot as plt
import numpy as np
from model import ResNet
from losses import FocalLoss, WeightedBCELoss
import pandas as pd
from sklearn.model_selection import train_test_split
import os

if __name__ == '__main__':
    os.makedirs('checkpoints', exist_ok=True)

    data = pd.read_csv('data.csv', sep=';')

    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42,
                                          stratify=data[['crack', 'inactive']])

    train_data = train_data.reset_index(drop=True)
    val_data = val_data.reset_index(drop=True)

    batch_size = 16  # Kleinere Batch-Größe für bessere Generalisierung
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

    model = ResNet()

    # Analysiere Klassenverteilung
    crack_pos = train_data['crack'].sum()
    crack_neg = len(train_data) - crack_pos
    inactive_pos = train_data['inactive'].sum()
    inactive_neg = len(train_data) - inactive_pos
    
    print(f"Crack distribution: {crack_pos} positive, {crack_neg} negative")
    print(f"Inactive distribution: {inactive_pos} positive, {inactive_neg} negative")
    
    # Gewichtete Loss-Funktion für unbalancierte Klassen
    crack_weight = crack_neg / crack_pos if crack_pos > 0 else 1.0
    inactive_weight = inactive_neg / inactive_pos if inactive_pos > 0 else 1.0
    pos_weight = t.tensor([crack_weight, inactive_weight])
    
    # Verwende Focal Loss
    criterion = FocalLoss(alpha=1.0, gamma=2.0)

    # Optimierte Hyperparameter
    learning_rate = 0.0005  # Niedrigere Learning Rate
    optimizer = t.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    
    # Learning Rate Scheduler
    scheduler = t.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    trainer = Trainer(
        model=model,
        crit=criterion,
        optim=optimizer,
        train_dl=train_loader,
        val_test_dl=val_loader,
        cuda=True,
        early_stopping_patience=15,  # Mehr Geduld für bessere Konvergenz
        scheduler=scheduler
    )

    print("Starting training...")

    res = trainer.fit(epochs=100)

    plt.figure(figsize=(15, 5))
    
    # Loss Plot
    plt.subplot(1, 3, 1)
    plt.plot(np.arange(len(res[0])), res[0], label='train loss', marker='o')
    plt.plot(np.arange(len(res[1])), res[1], label='val loss', marker='s')
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # F1 Score Plot
    plt.subplot(1, 3, 2)
    plt.plot(np.arange(len(res[2])), res[2], label='val F1 score', marker='d', color='green')
    plt.axhline(y=0.6, color='r', linestyle='--', label='Target (60%)')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('Validation F1 Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Combined Plot
    plt.subplot(1, 3, 3)
    plt.plot(np.arange(len(res[0])), res[0], label='train loss', alpha=0.7)
    plt.plot(np.arange(len(res[1])), res[1], label='val loss', alpha=0.7)
    plt.twinx()
    plt.plot(np.arange(len(res[2])), res[2], label='val F1 score', color='green', marker='d')
    plt.axhline(y=0.6, color='r', linestyle='--', label='Target (60%)')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('Loss and F1 Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('losses.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("Training completed!")
    print(f"Final training loss: {res[0][-1]:.4f}")
    print(f"Final validation loss: {res[1][-1]:.4f}")
    print(f"Final validation F1 score: {res[2][-1]:.4f}")
    print(f"Best validation F1 score: {max(res[2]):.4f}")
    print(f"Training completed in {len(res[0])} epochs")
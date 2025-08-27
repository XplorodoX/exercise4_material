import torch as t
from data import ChallengeDataset
from trainer import Trainer
from matplotlib import pyplot as plt
import numpy as np
from model import ResNet
from losses import FocalLoss
import pandas as pd
from sklearn.model_selection import train_test_split
import os
from sklearn.metrics import f1_score

def create_ensemble_model(n_models=3):
    """Erstelle ein Ensemble aus mehreren Modellen"""
    models = []
    for i in range(n_models):
        model = ResNet()
        models.append(model)
    return models

def train_ensemble(models, train_loader, val_loader, epochs=100):
    """Trainiere ein Ensemble von Modellen"""
    trainers = []
    results = []
    
    for i, model in enumerate(models):
        print(f"\n{'='*60}")
        print(f"Training Model {i+1}/{len(models)}")
        print(f"{'='*60}")
        
        # Verschiedene Hyperparameter für Diversität
        lrs = [0.0003, 0.0005, 0.0008]
        weight_decays = [1e-3, 1e-4, 5e-4]
        
        criterion = FocalLoss(alpha=1.0, gamma=2.0)
        optimizer = t.optim.AdamW(model.parameters(), 
                                 lr=lrs[i % len(lrs)], 
                                 weight_decay=weight_decays[i % len(weight_decays)])
        scheduler = t.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        
        trainer = Trainer(
            model=model,
            crit=criterion,
            optim=optimizer,
            train_dl=train_loader,
            val_test_dl=val_loader,
            cuda=True,
            early_stopping_patience=20,
            scheduler=scheduler
        )
        
        trainers.append(trainer)
        res = trainer.fit(epochs=epochs)
        results.append(res)
        
        # Speichere das Modell
        trainer.save_checkpoint(f'ensemble_{i}')
    
    return trainers, results

def evaluate_ensemble(models, val_loader):
    """Evaluiere das Ensemble"""
    all_predictions = []
    all_labels = []
    
    for model in models:
        model.eval()
        model_predictions = []
        
        with t.no_grad():
            for batch_idx, (x, y) in enumerate(val_loader):
                if t.cuda.is_available():
                    x = x.cuda()
                    y = y.cuda()
                
                predictions = model(x)
                model_predictions.append(predictions.cpu().numpy())
                
                if len(all_labels) == 0:  # Nur einmal sammeln
                    all_labels.append(y.cpu().numpy())
        
        all_predictions.append(np.vstack(model_predictions))
    
    all_labels = np.vstack(all_labels)
    
    # Ensemble-Prediction durch Mittelwert
    ensemble_predictions = np.mean(all_predictions, axis=0)
    
    # Optimale Schwellenwerte finden
    best_thresholds = [0.5, 0.5]
    best_f1_scores = [0.0, 0.0]
    
    for threshold in np.arange(0.1, 0.9, 0.05):
        binary_preds_thresh = (ensemble_predictions > threshold).astype(int)
        f1_crack_thresh = f1_score(all_labels[:, 0], binary_preds_thresh[:, 0])
        f1_inactive_thresh = f1_score(all_labels[:, 1], binary_preds_thresh[:, 1])
        
        if f1_crack_thresh > best_f1_scores[0]:
            best_f1_scores[0] = f1_crack_thresh
            best_thresholds[0] = threshold
        if f1_inactive_thresh > best_f1_scores[1]:
            best_f1_scores[1] = f1_inactive_thresh
            best_thresholds[1] = threshold
    
    # Finale Predictions
    binary_predictions = np.zeros_like(ensemble_predictions)
    binary_predictions[:, 0] = (ensemble_predictions[:, 0] > best_thresholds[0]).astype(int)
    binary_predictions[:, 1] = (ensemble_predictions[:, 1] > best_thresholds[1]).astype(int)
    
    f1_crack = f1_score(all_labels[:, 0], binary_predictions[:, 0])
    f1_inactive = f1_score(all_labels[:, 1], binary_predictions[:, 1])
    mean_f1 = (f1_crack + f1_inactive) / 2
    
    print(f"\nEnsemble Results:")
    print(f"Optimal Thresholds: Crack={best_thresholds[0]:.2f}, Inactive={best_thresholds[1]:.2f}")
    print(f"F1 Score - Crack: {f1_crack:.4f}")
    print(f"F1 Score - Inactive: {f1_inactive:.4f}")
    print(f"Mean F1 Score: {mean_f1:.4f}")
    
    return mean_f1, best_thresholds

if __name__ == '__main__':
    os.makedirs('checkpoints', exist_ok=True)
    
    # Daten laden
    data = pd.read_csv('data.csv', sep=';')
    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42,
                                          stratify=data[['crack', 'inactive']])
    
    train_data = train_data.reset_index(drop=True)
    val_data = val_data.reset_index(drop=True)
    
    # Datasets erstellen
    train_dataset = ChallengeDataset(train_data, 'train')
    val_dataset = ChallengeDataset(val_data, 'val')
    
    train_loader = t.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    val_loader = t.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)
    
    # Ensemble erstellen und trainieren
    models = create_ensemble_model(n_models=3)
    trainers, results = train_ensemble(models, train_loader, val_loader, epochs=100)
    
    # Ensemble evaluieren
    final_f1, thresholds = evaluate_ensemble([trainer._model for trainer in trainers], val_loader)
    
    print(f"\nFinal Ensemble F1 Score: {final_f1:.4f}")
    if final_f1 >= 0.6:
        print("🎉 Ziel erreicht! F1 Score >= 60%")
    else:
        print(f"Noch {0.6 - final_f1:.4f} Punkte bis zum Ziel von 60%")

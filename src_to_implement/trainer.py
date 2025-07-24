import torch as t
from sklearn.metrics import f1_score
from tqdm.autonotebook import tqdm
import numpy as np

class Trainer:

    def __init__(self,
                 model,
                 crit,
                 optim=None,
                 train_dl=None,
                 val_test_dl=None,
                 cuda=True,
                 early_stopping_patience=-1,
                 scheduler=None):
        self._model = model
        self._crit = crit
        self._optim = optim
        self._train_dl = train_dl
        self._val_test_dl = val_test_dl
        self._cuda = cuda
        self._scheduler = scheduler

        self._early_stopping_patience = early_stopping_patience

        if cuda and t.cuda.is_available():
            self._model = model.cuda()
            self._crit = crit.cuda()

    def save_checkpoint(self, epoch):
        t.save({'state_dict': self._model.state_dict()}, 'checkpoints/checkpoint_{:03d}.ckp'.format(epoch))

    def restore_checkpoint(self, epoch_n):
        device = 'cuda' if self._cuda and t.cuda.is_available() else 'cpu'
        ckp = t.load('checkpoints/checkpoint_{:03d}.ckp'.format(epoch_n), map_location=device)
        self._model.load_state_dict(ckp['state_dict'])

    def save_onnx(self, fn):
        m = self._model.cpu()
        m.eval()
        x = t.randn(1, 3, 300, 300, requires_grad=True)
        y = self._model(x)
        t.onnx.export(m,
                      x,
                      fn,
                      export_params=True,
                      opset_version=10,
                      do_constant_folding=True,
                      input_names=['input'],
                      output_names=['output'],
                      dynamic_axes={'input': {0: 'batch_size'},
                                    'output': {0: 'batch_size'}})

    def train_step(self, x, y):
        self._optim.zero_grad()

        predictions = self._model(x)

        loss = self._crit(predictions, y)

        loss.backward()

        self._optim.step()

        return loss.item()

    def val_test_step(self, x, y):
        predictions = self._model(x)

        loss = self._crit(predictions, y)

        return loss.item(), predictions

    def train_epoch(self):
        self._model.train()

        total_loss = 0.0
        num_batches = 0

        for batch_idx, (x, y) in enumerate(tqdm(self._train_dl, desc="Training")):
            if self._cuda and t.cuda.is_available():
                x = x.cuda()
                y = y.cuda()

            loss = self.train_step(x, y)
            total_loss += loss
            num_batches += 1

        avg_loss = total_loss / num_batches
        return avg_loss

    def val_test(self):
        self._model.eval()

        total_loss = 0.0
        num_batches = 0
        all_predictions = []
        all_labels = []

        with t.no_grad():
            for batch_idx, (x, y) in enumerate(tqdm(self._val_test_dl, desc="Validation")):
                if self._cuda and t.cuda.is_available():
                    x = x.cuda()
                    y = y.cuda()

                loss, predictions = self.val_test_step(x, y)
                total_loss += loss
                num_batches += 1

                all_predictions.append(predictions.cpu().numpy())
                all_labels.append(y.cpu().numpy())

        avg_loss = total_loss / num_batches

        all_predictions = np.vstack(all_predictions)
        all_labels = np.vstack(all_labels)

        # Optimale Schwellenwerte für jede Klasse finden
        best_thresholds = [0.5, 0.5]
        best_f1_scores = [0.0, 0.0]
        
        for threshold in np.arange(0.1, 0.9, 0.05):
            binary_preds_thresh = (all_predictions > threshold).astype(int)
            f1_crack_thresh = f1_score(all_labels[:, 0], binary_preds_thresh[:, 0])
            f1_inactive_thresh = f1_score(all_labels[:, 1], binary_preds_thresh[:, 1])
            
            if f1_crack_thresh > best_f1_scores[0]:
                best_f1_scores[0] = f1_crack_thresh
                best_thresholds[0] = threshold
            if f1_inactive_thresh > best_f1_scores[1]:
                best_f1_scores[1] = f1_inactive_thresh
                best_thresholds[1] = threshold

        # Verwende optimale Schwellenwerte
        binary_predictions = np.zeros_like(all_predictions)
        binary_predictions[:, 0] = (all_predictions[:, 0] > best_thresholds[0]).astype(int)
        binary_predictions[:, 1] = (all_predictions[:, 1] > best_thresholds[1]).astype(int)

        f1_crack = f1_score(all_labels[:, 0], binary_predictions[:, 0])
        f1_inactive = f1_score(all_labels[:, 1], binary_predictions[:, 1])
        mean_f1 = (f1_crack + f1_inactive) / 2

        print(f"Validation Loss: {avg_loss:.4f}")
        print(f"Optimal Thresholds: Crack={best_thresholds[0]:.2f}, Inactive={best_thresholds[1]:.2f}")
        print(f"F1 Score - Crack: {f1_crack:.4f}")
        print(f"F1 Score - Inactive: {f1_inactive:.4f}")
        print(f"Mean F1 Score: {mean_f1:.4f}")

        return avg_loss, mean_f1

    def fit(self, epochs=-1):
        assert self._early_stopping_patience > 0 or epochs > 0

        train_losses = []
        val_losses = []
        val_f1_scores = []
        epoch_counter = 0
        best_val_f1 = 0.0
        best_val_loss = float('inf')
        patience_counter = 0

        while True:
            if epochs > 0 and epoch_counter >= epochs:
                break

            print(f"\nEpoch {epoch_counter + 1}")
            print("-" * 50)

            train_loss = self.train_epoch()
            train_losses.append(train_loss)

            val_loss, val_f1 = self.val_test()
            val_losses.append(val_loss)
            val_f1_scores.append(val_f1)

            print(f"Training Loss: {train_loss:.4f}")

            # Learning Rate Scheduler
            if self._scheduler is not None:
                self._scheduler.step(val_loss)
                current_lr = self._optim.param_groups[0]['lr']
                print(f"Current Learning Rate: {current_lr:.6f}")

            self.save_checkpoint(epoch_counter)

            if self._early_stopping_patience > 0:
                # Early stopping basierend auf F1 Score
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    patience_counter = 0
                    print(f"New best F1 Score: {best_val_f1:.4f}")
                else:
                    patience_counter += 1

                if patience_counter >= self._early_stopping_patience:
                    print(f"\nEarly stopping triggered after {epoch_counter + 1} epochs")
                    print(f"Best F1 Score achieved: {best_val_f1:.4f}")
                    break

            epoch_counter += 1

        return train_losses, val_losses, val_f1_scores
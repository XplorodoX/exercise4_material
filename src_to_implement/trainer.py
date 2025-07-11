import torch as t
from sklearn.metrics import f1_score
from tqdm.autonotebook import tqdm
import numpy as np


class Trainer:

    def __init__(self,
                 model,  # Model to be trained.
                 crit,  # Loss function
                 optim=None,  # Optimizer
                 train_dl=None,  # Training data set
                 val_test_dl=None,  # Validation (or test) data set
                 cuda=True,  # Whether to use the GPU
                 early_stopping_patience=-1):  # The patience for early stopping
        self._model = model
        self._crit = crit
        self._optim = optim
        self._train_dl = train_dl
        self._val_test_dl = val_test_dl
        self._cuda = cuda

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
        t.onnx.export(m,  # model being run
                      x,  # model input (or a tuple for multiple inputs)
                      fn,  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=10,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'],  # the model's output names
                      dynamic_axes={'input': {0: 'batch_size'},  # variable lenght axes
                                    'output': {0: 'batch_size'}})

    def train_step(self, x, y):
        # Reset gradients
        self._optim.zero_grad()

        # Forward pass
        predictions = self._model(x)

        # Calculate loss
        loss = self._crit(predictions, y)

        # Backward pass
        loss.backward()

        # Update weights
        self._optim.step()

        return loss.item()

    def val_test_step(self, x, y):
        # Forward pass
        predictions = self._model(x)

        # Calculate loss
        loss = self._crit(predictions, y)

        return loss.item(), predictions

    def train_epoch(self):
        # Set training mode
        self._model.train()

        total_loss = 0.0
        num_batches = 0

        # Iterate through training set
        for batch_idx, (x, y) in enumerate(tqdm(self._train_dl, desc="Training")):
            # Transfer to GPU if available
            if self._cuda and t.cuda.is_available():
                x = x.cuda()
                y = y.cuda()

            # Perform training step
            loss = self.train_step(x, y)
            total_loss += loss
            num_batches += 1

        # Calculate average loss
        avg_loss = total_loss / num_batches
        return avg_loss

    def val_test(self):
        # Set eval mode
        self._model.eval()

        total_loss = 0.0
        num_batches = 0
        all_predictions = []
        all_labels = []

        # Disable gradient computation
        with t.no_grad():
            for batch_idx, (x, y) in enumerate(tqdm(self._val_test_dl, desc="Validation")):
                # Transfer to GPU if available
                if self._cuda and t.cuda.is_available():
                    x = x.cuda()
                    y = y.cuda()

                # Perform validation step
                loss, predictions = self.val_test_step(x, y)
                total_loss += loss
                num_batches += 1

                # Store predictions and labels for metrics calculation
                all_predictions.append(predictions.cpu().numpy())
                all_labels.append(y.cpu().numpy())

        # Calculate average loss
        avg_loss = total_loss / num_batches

        # Calculate metrics
        all_predictions = np.vstack(all_predictions)
        all_labels = np.vstack(all_labels)

        # Convert predictions to binary (threshold at 0.5)
        binary_predictions = (all_predictions > 0.5).astype(int)

        # Calculate F1 scores for each class
        f1_crack = f1_score(all_labels[:, 0], binary_predictions[:, 0])
        f1_inactive = f1_score(all_labels[:, 1], binary_predictions[:, 1])
        mean_f1 = (f1_crack + f1_inactive) / 2

        print(f"Validation Loss: {avg_loss:.4f}")
        print(f"F1 Score - Crack: {f1_crack:.4f}")
        print(f"F1 Score - Inactive: {f1_inactive:.4f}")
        print(f"Mean F1 Score: {mean_f1:.4f}")

        return avg_loss

    def fit(self, epochs=-1):
        assert self._early_stopping_patience > 0 or epochs > 0

        # Initialize lists for losses and counters
        train_losses = []
        val_losses = []
        epoch_counter = 0
        best_val_loss = float('inf')
        patience_counter = 0

        while True:
            # Check stopping conditions
            if epochs > 0 and epoch_counter >= epochs:
                break

            print(f"\nEpoch {epoch_counter + 1}")
            print("-" * 50)

            # Train for one epoch
            train_loss = self.train_epoch()
            train_losses.append(train_loss)

            # Validate
            val_loss = self.val_test()
            val_losses.append(val_loss)

            print(f"Training Loss: {train_loss:.4f}")

            # Save checkpoint
            self.save_checkpoint(epoch_counter)

            # Early stopping check
            if self._early_stopping_patience > 0:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= self._early_stopping_patience:
                    print(f"\nEarly stopping triggered after {epoch_counter + 1} epochs")
                    break

            epoch_counter += 1

        return train_losses, val_losses
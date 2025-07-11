import torch as t
from trainer import Trainer
from model import ResNet
import sys
import torchvision as tv

if len(sys.argv) != 2:
    print("Usage: python export_onnx.py <epoch_number>")
    sys.exit(1)

epoch = int(sys.argv[1])

# Create the model
model = ResNet()

# Set up criterion (needed for trainer initialization)
crit = t.nn.BCELoss()

# Create trainer
trainer = Trainer(model, crit, cuda=False)  # Use CPU for export

# Restore the checkpoint
trainer.restore_checkpoint(epoch)

# Export to ONNX format
output_filename = 'checkpoint_{:03d}.onnx'.format(epoch)
trainer.save_onnx(output_filename)

print(f"Model from epoch {epoch} exported to {output_filename}")
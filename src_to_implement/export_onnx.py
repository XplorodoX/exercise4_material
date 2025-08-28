import torch as t
from trainer import Trainer
from model import ResNet
import sys

#if len(sys.argv) != 2:
  #  print("Usage: python export_onnx.py <epoch_number>")
 #   sys.exit(1)

#epoch = int(sys.argv[1])

epoch = 53

model = ResNet()

crit = t.nn.BCELoss()

trainer = Trainer(model, crit, cuda=False)  # Use CPU for export

trainer.restore_checkpoint(epoch)

output_filename = 'checkpoint_{:03d}.onnx'.format(epoch)
trainer.save_onnx(output_filename)

print(f"Model from epoch {epoch} exported to {output_filename}")
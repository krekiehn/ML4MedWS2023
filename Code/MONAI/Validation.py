import os

import matplotlib.pyplot as plt
import torch
from monai.data import CacheDataset, DataLoader, decollate_batch
from monai.metrics import DiceMetric
from monai.networks.nets import SegResNetDS
from monai.transforms import (
    AsDiscrete,
    Compose
)
from monai.utils import set_determinism

from Code.MONAI import AppliedTransforms
from Code.MONAI.DataLoader import get_data_dicts

if os.path.isdir(r'.\ML4MedWS2023'):
    root_dir = r'.\ML4MedWS2023'
else:
    root_dir = os.getcwd()


# check if cuda or mps is available and set device accordingly
def set_device():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        device = torch.device("mps:0")
    else:
        device = torch.device("cpu:0")
    return device


# Settigns

device = set_device()
set_determinism(seed=1)
SPATIAL_DIMS = 2
LABELS = [0, 1, 2, 3, 4, 5, 6]
NUM_CLASSES = len(LABELS)
BATCH_SIZE = 8
MAX_EPOCHS = 128
VAL_INTERVAL = 1

# Get Data
train_files, val_files = get_data_dicts()

val_ds = CacheDataset(data=val_files, transform=AppliedTransforms.val_transforms_BASELINE1_2, cache_rate=1.0,
                      num_workers=2)
val_loader = DataLoader(val_ds, batch_size=1)

print("## Data loaded")

print(root_dir)
# Load Model
filepath = os.path.join(root_dir, "Model", "Save",
                        "best_metric_dice_model_0_2024-01-08_19-21-15_SEGRESNET_BASE1_2_schedulerOnPlateau.pth")
model = SegResNetDS(
    spatial_dims=SPATIAL_DIMS,
    in_channels=1,
    out_channels=NUM_CLASSES,
).to(device)
model.load_state_dict(torch.load(filepath))
model.eval()

print("## Model loaded")

# Metrics
dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=True)
dice_metric_values = []
metric_per_class_values = []

post_pred = Compose([AsDiscrete(argmax=True, to_onehot=NUM_CLASSES)])
post_label = Compose([AsDiscrete(to_onehot=NUM_CLASSES)])

print("## Metrics initialized")
print("## Start validation")
for epoch in range(MAX_EPOCHS):
    print(f"Epoch {epoch + 1}/{MAX_EPOCHS}")
    # perform validation
    if (epoch + 1) % VAL_INTERVAL == 0:
        model.eval()  # set model to evaluation mode (dropout, batch normalization...)
        with torch.no_grad():  # disable gradient calculations
            for val_data in val_loader:
                print("Validation")
                val_inputs, val_labels = (
                    val_data["image"].to(device),
                    val_data["label"].to(device),
                )

                val_outputs = model(val_inputs)
                val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                val_labels = [post_label(i) for i in decollate_batch(val_labels)]

                # Evaluate dice metric
                dice_metric(y_pred=val_outputs, y=val_labels)

            # aggregate the final mean dice result
            metric = dice_metric.aggregate().item()
            print(f"Dice metric: {metric}")
            # reset the status for next validation round
            metric_per_class = dice_metric.aggregate(reduction="none").mean(axis=0)
            dice_metric.reset()
            # Save the metric value
            dice_metric_values.append(metric)
            metric_per_class_values.append(metric_per_class)

################################################
# Plot "Dice per class"

plt.figure("train", (12, 6))

# Plot "Dice per class"
plt.title("Dice pro class")
x = [VAL_INTERVAL * (i + 1) for i in range(len(metric_per_class_values))]

for i in range(6):
    # Extract the metric values for the current class across all epochs
    y = [metric.cpu().numpy()[i] for metric in metric_per_class_values]
    plt.plot(x, y, label=i + 1)

plt.xlabel("epoch")
plt.ylabel("dice value")

plt.legend()
plt.show()

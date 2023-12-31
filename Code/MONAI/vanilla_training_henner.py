import os

import torch
from monai.config import print_config
from monai.data import DataLoader, Dataset, CacheDataset
from monai.losses import DiceLoss
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.networks.layers import Norm
from monai.networks.nets import UNet
from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    LoadImaged,
    Orientationd,
    DivisiblePadd,
    SpatialPadd,
    Resized,
    RandGaussianNoised
)
from monai.utils import set_determinism

from Code.AnalyseData import DataViz
from Code.MONAI import AppliedTransforms
from Code.MONAI.AppliedTransforms import train_transforms_RandGaussianNoise, train_transforms_Elastic
from Code.MONAI.CustomTransforms import ReplaceValuesNotInList
from Code.MONAI.DataLoader import get_data_dicts, check_transforms_in_dataloader
from Code.MONAI.TrainingLoop import TRAINING
import numpy as np

# from multiprocessing import Process, freeze_support, set_start_method

# Changes:
# ---------------------------------------
# - fix bug: validate with train files
# - use CacheDataset for performance reasons
# - use Resized to 1024 for performance reasons
# - change fixed path to relative ones
# - removed SpatialPadd (used for what?)
# - reduced Epochs
# - transforms from another file
# - RandGaussianNoise
# - Learning Rate verändert auf hoch -3


print_config()

if os.path.isdir(r'.\ML4MedWS2023'):
    root_dir = r'.\ML4MedWS2023'
else:
    root_dir = os.getcwd()

# CONFIG
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"{'':#<30}")
print(f"{' Used device is ' + str(device) + ' ':#^30}")
print(f"{'':#<30}")

set_determinism(seed=1)
SPATIAL_DIMS = 2
LABELS = [0, 1, 2, 3, 4, 5, 6]
NUM_CLASSES = len(LABELS)
LEARNING_RATE = 1e-3

# UNET ARCHITECTURE
UNET_CHANNELS = (16, 32, 64, 128, 256)
UNET_STRIDE = 2
UNET_STRIDES = tuple([UNET_STRIDE] * (len(UNET_CHANNELS) - 1))
K = 2 ** (len(UNET_CHANNELS) - 1)

debug_mode = False
if debug_mode:
    # Debug Mode
    BATCH_SIZE = 32
    MAX_EPOCHS = 2
    VAL_INTERVAL = 1

    train_files, val_files = get_data_dicts(stop_index=BATCH_SIZE)
else:
    # User Mode
    BATCH_SIZE = 32
    MAX_EPOCHS = 256
    VAL_INTERVAL = 1

    train_files, val_files = get_data_dicts()

# Check transforms in DataLoader
check_ds = Dataset(data=train_files, transform=AppliedTransforms.train_transforms_Anna)
check_transforms_in_dataloader(check_ds)

train_ds = CacheDataset(data=train_files, transform=AppliedTransforms.train_transforms_Anna, cache_rate=1.0, num_workers=2)
# train_ds = Dataset(data=train_files, transform=train_transforms)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

val_ds = CacheDataset(data=val_files, transform=AppliedTransforms.val_transforms_BASELINE2, cache_rate=1.0,
                      num_workers=2)
# val_ds = Dataset(data=val_files, transform=val_transforms)
val_loader = DataLoader(val_ds, batch_size=1)

# Model

# standard PyTorch program style: create UNet, DiceLoss and Adam optimizer
model = UNet(
    spatial_dims=SPATIAL_DIMS,
    in_channels=1,
    out_channels=NUM_CLASSES,
    channels=UNET_CHANNELS,
    strides=UNET_STRIDES,
    num_res_units=2,
    norm=Norm.BATCH,
).to(device)
loss_function = DiceLoss(to_onehot_y=True, softmax=True)
optimizer = torch.optim.Adam(model.parameters(), LEARNING_RATE)
dice_metric = DiceMetric(include_background=False, reduction="mean")
HD_metric = HausdorffDistanceMetric(include_background=False, reduction="mean")

# Define multiple metrics
metrics = {
    'dice': DiceMetric(include_background=False, reduction='mean'),
    'hausdorff': HausdorffDistanceMetric(include_background=False, reduction='mean'),
}
# TRAINING
model, epoch_loss_values, metric_values \
    = TRAINING(model, NUM_CLASSES=NUM_CLASSES, MAX_EPOCHS=MAX_EPOCHS, VAL_INTERVAL=VAL_INTERVAL,
               train_loader=train_loader,
               val_loader=val_loader,
               optimizer=optimizer,
               loss_function=loss_function,
               metrics=metrics,
               train_ds=train_ds,
               indicator='ANNA_256Epocs',
               device=device,
               )

### Outputs
DataViz.show_masks(model, val_loader, device)
DataViz.show_elbow_plot(epoch_loss_values=epoch_loss_values, val_interval=VAL_INTERVAL, metric_values=metric_values['dice'])

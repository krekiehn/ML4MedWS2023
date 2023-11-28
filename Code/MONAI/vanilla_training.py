from monai.utils import first, set_determinism
from monai.transforms import (
    AsDiscrete,
    AsDiscreted,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    SaveImaged,
    ScaleIntensityRanged,
    Spacingd,
    Invertd,
    RandAffined,
    RandGaussianNoised,
    DivisiblePadd
)
from monai.handlers.utils import from_engine
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch
from monai.config import print_config
from monai.apps import download_and_extract
import torch
import matplotlib.pyplot as plt
import tempfile
import shutil
import os
import glob
import numpy as np
from datetime import datetime

# from multiprocessing import Process, freeze_support, set_start_method
from Code.MONAI.CustomTransforms import ReplaceValuesNotInList
from Code.MONAI.DataLoader import get_data_dicts, check_transforms_in_dataloader
from Code.MONAI.TrainingLoop import TRAINING

print_config()

root_dir = r'C:\Users\Nicolai\PycharmProjects\ML4MedWS2023'

# CONFIG
set_determinism(seed=1)
SPATIAL_DIMS = 2
LABLES = [0, 1, 2, 3, 4, 5, 6]
NUM_CLASSES = len(LABLES)
LEARNING_RATE = 1e-4

debug_mode = True
if debug_mode:
    # Debug Mode
    BATCH_SIZE = 1
    MAX_EPOCHS = 1
    VAL_INTERVAL = 1
else:
    # User Mode
    BATCH_SIZE = 16
    MAX_EPOCHS = 600
    VAL_INTERVAL = 1

train_files, val_files = get_data_dicts(stop_index=BATCH_SIZE)

train_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="LP"),
        ReplaceValuesNotInList(keys=['label'], allowed_values=[0, 1, 2, 3, 4, 5, 6], replacement_value=0),
        DivisiblePadd(keys=["image", "label"], k=16)
    ]
)
val_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="LP"),
        ReplaceValuesNotInList(keys=['label'], allowed_values=[0, 1, 2, 3, 4, 5, 6], replacement_value=0),
        DivisiblePadd(keys=["image", "label"], k=16),
    ]
)


# Check transforms in DataLoader
check_transforms_in_dataloader(check_ds := Dataset(data=train_files, transform=train_transforms))

train_ds = Dataset(data=train_files, transform=train_transforms)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

val_ds = Dataset(data=val_files, transform=val_transforms)
val_loader = DataLoader(val_ds, batch_size=1)

# Model

# standard PyTorch program style: create UNet, DiceLoss and Adam optimizer
device = torch.device("cpu:0")
model = UNet(
    spatial_dims=SPATIAL_DIMS,
    in_channels=1,
    out_channels=NUM_CLASSES,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
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
model = TRAINING(model, NUM_CLASSES = NUM_CLASSES, MAX_EPOCHS = MAX_EPOCHS, VAL_INTERVAL = VAL_INTERVAL,
train_loader = train_loader,
val_loader = val_loader,
optimizer = optimizer,
loss_function = loss_function,
metrics = metrics,
# metric_name = 'DICE',
train_ds = train_ds,
device = torch.device("cpu:0"),
)

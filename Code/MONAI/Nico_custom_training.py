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
    RandAdjustContrastd,
    DivisiblePadd,
    Rand2DElasticd,
    SpatialPadd
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
import sys
sys.path.append(os.getcwd())

from Code.MONAI.CustomTransforms import ReplaceValuesNotInList, PadToMaxSize
from Code.MONAI.DataLoader import get_data_dicts, check_transforms_in_dataloader
from Code.MONAI.TrainingLoop import TRAINING

########################################################
# PLEASE CUSTOMIZE ME TO EXPERIMENT DIFFERENT SETTINGS #
########################################################

print_config()

if os.path.isdir(r'C:\Users\Nicolai\PycharmProjects\ML4MedWS2023'):
    root_dir = r'C:\Users\Nicolai\PycharmProjects\ML4MedWS2023'
else:
    root_dir = os.getcwd()

# CONFIG
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"{'':#<30}")
print(f"{' Used device is ' + str(device) + ' ':#^30}")
print(f"{'':#<30}")

set_determinism(seed=1)
SPATIAL_DIMS = 2
LABLES = [0, 1, 2, 3, 4, 5, 6]
NUM_CLASSES = len(LABLES)
LEARNING_RATE = 1e-4

# UNET ARCHITECTURE
UNET_CHANNELS = (16, 32, 64, 128, 256)
UNET_STRIDE = 2
UNET_STRIDES = tuple([UNET_STRIDE] * (len(UNET_CHANNELS) - 1))
K = 2**(len(UNET_CHANNELS)-1)

debug_mode = False
if debug_mode:
    # Debug Mode
    BATCH_SIZE = 1
    MAX_EPOCHS = 1
    VAL_INTERVAL = 1

    train_files, val_files = get_data_dicts(stop_index=BATCH_SIZE)
else:
    # User Mode
    BATCH_SIZE = 2*2
    MAX_EPOCHS = 600
    VAL_INTERVAL = 1

    train_files, val_files = get_data_dicts()


train_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="LP"),
        # PadToMaxSize(keys=['image', 'label']),
        ReplaceValuesNotInList(keys=['label'], allowed_values=LABLES, replacement_value=0),
        SpatialPadd(keys=["image", "label"], spatial_size=(2991, 2992)),
        DivisiblePadd(keys=["image", "label"], k=16),
        Rand2DElasticd(keys=['image', 'label'], spacing=(20, 20), magnitude_range=(0, 20),
                       rotate_range=(-np.pi, np.pi), translate_range=((-1000, 1000), (-1000, 1000)),
                       scale_range=((-0.5, 0.5), (-0.5, 0.5)),
                       padding_mode="zeros", mode=["bilinear", "nearest"], prob=1),
        # Rand2DElasticd(keys=['image'], spacing=(20), magnitude_range=(10, 20),
        #                padding_mode="zeros", mode=["bilinear"], prob=1),
        RandAdjustContrastd(keys=['image'], gamma=(0.5, 2), prob=1, retain_stats=True, invert_image=True),
    ]
)
val_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="LP"),
        ReplaceValuesNotInList(keys=['label'], allowed_values=LABLES, replacement_value=0),
        DivisiblePadd(keys=["image", "label"], k=16),
    ]
)


# Check transforms in DataLoader
check_transforms_in_dataloader(check_ds := Dataset(data=train_files, transform=train_transforms))

# train_ds = Dataset(data=train_files, transform=train_transforms)
# train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0, num_workers=0)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

# val_ds = Dataset(data=val_files, transform=val_transforms)
# val_loader = DataLoader(val_ds, batch_size=1)

val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=1, num_workers=0)

# Model

# standard PyTorch program style: create UNet, DiceLoss and Adam optimizer
# device = torch.device("cpu:0")
model = UNet(
    spatial_dims=SPATIAL_DIMS,
    in_channels=1,
    out_channels=NUM_CLASSES,
    channels=UNET_CHANNELS,
    strides=UNET_STRIDES,
    num_res_units=2,
    norm=Norm.BATCH,
).to(device)
if device == torch.device("cuda"):
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model, device_ids = list(range(torch.cuda.device_count())))

loss_function = DiceLoss(to_onehot_y=True, softmax=True)
optimizer = torch.optim.Adam(model.parameters(), LEARNING_RATE)
dice_metric = DiceMetric(include_background=False, reduction="mean")
HD_metric = HausdorffDistanceMetric(include_background=False, reduction="mean")

# Define multiple metrics
metrics = {
    'dice': DiceMetric(include_background=False, reduction='mean'),
    # 'hausdorff': HausdorffDistanceMetric(include_background=False, reduction='mean'),
}
# TRAINING
model = TRAINING(model, NUM_CLASSES = NUM_CLASSES, MAX_EPOCHS = MAX_EPOCHS, VAL_INTERVAL = VAL_INTERVAL,
train_loader = train_loader,
val_loader = val_loader,
optimizer = optimizer,
loss_function = loss_function,
metrics = metrics,
train_ds = train_ds,
device = device,
)
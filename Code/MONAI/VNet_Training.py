import os

import torch
from monai.config import print_config
from monai.data import DataLoader, Dataset, CacheDataset
from monai.losses import DiceLoss
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.networks.layers import Norm
from monai.networks.nets import UNet, VNet
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

from Code.MONAI.TrainingLoop_withoutSlidingWindow import TRAINING_withoutSlidingWindow
from Code.MONAI.TrainingLoop_withoutSlidingWindow_wScheduler import TRAINING_withoutSlidingWindow_withScheduler


print_config()

if os.path.isdir(r'.\ML4MedWS2023'):
    root_dir = r'.\ML4MedWS2023'
else:
    root_dir = os.getcwd()


# check if cuda or mps is available and set device accordingly
def set_device():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    #elif torch.backends.mps.is_available():
    #    device = torch.device("mps:0")
    else:
        device = torch.device("cpu:0")
    return device


# CONFIG
device = set_device()
print(f"{'':#<30}")
print(f"{' Used device is ' + str(device) + ' ':#^30}")
print(f"{'':#<30}")

set_determinism(seed=1)
SPATIAL_DIMS = 2
LABELS = [0, 1, 2, 3, 4, 5, 6]
NUM_CLASSES = len(LABELS)
LEARNING_RATE = 1e-3

# VNet ARCHITECTURE
# UNET_CHANNELS = (16, 32, 64, 128, 256)
#VNET_CHANNELS = (32, 64, 128, 256, 512)
VNET_STRIDE = 2
#VNET_STRIDES = tuple([VNET_STRIDE] * (len(VNET_CHANNELS) - 1))
#K = 2 ** (len(VNET_CHANNELS) - 1)

debug_mode = True
if debug_mode:
    # Debug Mode
    BATCH_SIZE = 8
    MAX_EPOCHS = 2
    VAL_INTERVAL = 1

    train_files, val_files = get_data_dicts(stop_index=BATCH_SIZE)
else:
    # User Mode
    BATCH_SIZE = 16
    MAX_EPOCHS = 128
    VAL_INTERVAL = 1

    train_files, val_files = get_data_dicts()

# Check transforms in DataLoader
check_ds = Dataset(data=train_files, transform=AppliedTransforms.train_transforms_NextTry4)
check_transforms_in_dataloader(check_ds)

train_ds = CacheDataset(data=train_files, transform=AppliedTransforms.train_transforms_NextTry4, cache_rate=1.0,
                        num_workers=2)
# train_ds = Dataset(data=train_files, transform=AppliedTransforms.train_transforms_NextTry2)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

val_ds = CacheDataset(data=val_files, transform=AppliedTransforms.val_transforms_BASELINE1_2, cache_rate=1.0,
                      num_workers=2)
# val_ds = Dataset(data=val_files, transform=AppliedTransforms.val_transforms_BASELINE1_2)
val_loader = DataLoader(val_ds, batch_size=1)

# Model

# standard PyTorch program style: create VNet, DiceLoss and Adam optimizer
model = VNet(
    spatial_dims=SPATIAL_DIMS,
    in_channels=1,
    out_channels=NUM_CLASSES,
    #channels=VNET_CHANNELS,
    #strides=VNET_STRIDES,
    #num_res_units=2,
    #norm=Norm.BATCH,
).to(device)
loss_function = DiceLoss(to_onehot_y=True, softmax=True, squared_pred=True)
optimizer = torch.optim.Adam(model.parameters(), LEARNING_RATE)
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.01)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min')

# metrics
dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=True)
HD_metric = HausdorffDistanceMetric(include_background=False, reduction="mean", get_not_nans=True)

# Define multiple metrics
metrics = {
    'dice': DiceMetric(include_background=False, reduction='mean'),
    'hausdorff': HausdorffDistanceMetric(include_background=False, reduction='mean'),
}
# TRAINING
model, epoch_loss_values, metric_values \
    = TRAINING_withoutSlidingWindow_withScheduler(model, NUM_CLASSES=NUM_CLASSES, MAX_EPOCHS=MAX_EPOCHS,
                                                  VAL_INTERVAL=VAL_INTERVAL,
                                                  train_loader=train_loader,
                                                  val_loader=val_loader,
                                                  optimizer=optimizer,
                                                  scheduler=scheduler,
                                                  loss_function=loss_function,
                                                  metrics=metrics,
                                                  train_ds=train_ds,
                                                  indicator='_BASE1_2_128Epochs_withoutSlidingWindow_Batchsize32_Ch32-512_schedulerOnPlateau',
                                                  device=device,
                                                  )

if not debug_mode:
    ### Outputs
    DataViz.show_masks_withoutSlidingWindow(model, val_loader, device)
    DataViz.show_elbow_plot(epoch_loss_values=epoch_loss_values, val_interval=VAL_INTERVAL,
                            metric_values=metric_values['dice'])

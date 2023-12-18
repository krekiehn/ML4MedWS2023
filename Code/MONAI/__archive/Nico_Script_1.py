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
from monai.metrics import DiceMetric
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
# from multiprocessing import Process, freeze_support, set_start_method
# from Code import datafile_folder

print_config()

# import numpy as np
import monai.transforms as T


class ReplaceValuesNotInList(T.MapTransform):
    def __init__(self, keys, allowed_values, replacement_value):
        super().__init__(keys)
        self.allowed_values = set(allowed_values)
        self.replacement_value = replacement_value

    def __call__(self, data):
        for key in self.keys:
            if key in data:
                data[key] = np.where(np.isin(data[key], list(self.allowed_values)),
                                     data[key],
                                     self.replacement_value)
        return data


# Example of usage:
# Replace values in the 'image' key that are not in [1, 2, 3] with a replacement value of 0.
# transform = ReplaceValuesNotInList(keys=['image'], allowed_values=[1, 2, 3], replacement_value=0)

# Apply the transform to a dictionary
# data_dict = {'image': np.array([[1, 2, 4], [5, 6, 7]])}
# transformed_data = transform(data_dict)

# Print the result
# print("Original data:")
# print(data_dict)
# print("\nTransformed data:")
# print(transformed_data)


# directory = os.environ.get("MONAI_DATA_DIRECTORY")
# root_dir = tempfile.mkdtemp() if directory is None else directory
# print(root_dir)

# data_dir = r'C:\Users\User\PycharmProjects\ML4Med\data\Converted Pelvis Nifti Dataset'
# r'Data/ConvertedPelvisNiftiDataset/Converted Pelvis Nifti Dataset'
# r'Data/ConvertedPelvisNiftiDataset/Converted Pelvis Nifti Dataset'
# data_dir = datafile_folder
data_dir = r'C:\Users\Nicolai\PycharmProjects\ML4MedWS2023\Data\ConvertedPelvisNiftiDataset\Converted Pelvis Nifti Dataset'
root_dir = data_dir

train_images = sorted(glob.glob(os.path.join(data_dir, "train", "imgs", "*.nii.gz")))
train_labels = sorted(glob.glob(os.path.join(data_dir, "train", "targets", "*.nii.gz")))

train_data_dicts = [{"image": image_name, "label": label_name} for image_name, label_name in zip(train_images, train_labels)]

val_images = sorted(glob.glob(os.path.join(data_dir, "valid", "imgs", "*.nii.gz")))
val_labels = sorted(glob.glob(os.path.join(data_dir, "valid", "targets", "*.nii.gz")))

val_data_dicts = [{"image": image_name, "label": label_name} for image_name, label_name in zip(train_images, train_labels)]

# train_files, val_files = train_data_dicts[:-9], val_data_dicts[-9:]
train_files, val_files = train_data_dicts, val_data_dicts
# train_files, val_files = train_data_dicts[:2], val_data_dicts[:2]

set_determinism(seed=1)


train_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="LP"),
        ReplaceValuesNotInList(keys=['label'], allowed_values=[0, 1, 2, 3, 4, 5, 6], replacement_value=0),
        DivisiblePadd(keys=["image", "label"], k=16),
        # ScaleIntensityRanged(
        #     keys=["image"],
        #     a_min=-57,
        #     a_max=164,
        #     b_min=0.0,
        #     b_max=1.0,
        #     clip=True,
        # ),
        # CropForegroundd(keys=["image", "label"], source_key="image"),
        # Orientationd(keys=["image", "label"]),
        # Orientationd(keys=["image", "label"], axcodes="RAS"),
        # Orientationd(keys=["image", "label"], axcodes="LP"),
        # Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5), mode=("bilinear", "nearest")),
        # Spacingd(keys=["image", "label"], pixdim=(10, 10), mode=("bilinear", "nearest")),
        # RandCropByPosNegLabeld(
        #     keys=["image", "label"],
        #     label_key="label",
        #     spatial_size=(1024, 1024),
        #     pos=1,
        #     neg=1,
        #     num_samples=4,
        #     image_key="image",
        #     image_threshold=0,
        # ),
        # user can also add other random transforms
        # RandGaussianNoised(
        #     # as_tensor_output=False,
        #     # keys=['image', 'label'],
        #     keys=['image'],
        #     # mode=('bilinear', 'nearest'),
        #     prob=1.0,
        #     mean=0.0,
        #     std=2,
        #     # rotate_range=(0, 0, np.pi/15),
        #     # scale_range=(0.1, 0.1, 0.1)),
        # )
    ]
)
val_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="LP"),
        ReplaceValuesNotInList(keys=['label'], allowed_values=[0, 1, 2, 3, 4, 5, 6], replacement_value=0),
        DivisiblePadd(keys=["image", "label"], k=16),
        # ScaleIntensityRanged(
        #     keys=["image"],
        #     a_min=-57,
        #     a_max=164,
        #     b_min=0.0,
        #     b_max=1.0,
        #     clip=True,
        # ),
        # CropForegroundd(keys=["image", "label"], source_key="image"),
        # Orientationd(keys=["image", "label"], axcodes="RAS"),
        # Orientationd(keys=["image", "label"], axcodes="LP"),
        # Orientationd(keys=["image", "label"]),
        # Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5), mode=("bilinear", "nearest")),
    ]
)


# Check transforms in DataLoader

# check_ds = Dataset(data=val_files, transform=val_transforms)
check_ds = Dataset(data=train_files, transform=train_transforms)
check_loader = DataLoader(check_ds, batch_size=1)
# print(check_loader)
check_data = first(check_loader)
image, label = (check_data["image"][0][0], check_data["label"][0][0])
print(f"image shape: {image.shape}, label shape: {label.shape}")
# plot the slice [:, :, 80]
plt.figure("check", (12, 6))
plt.subplot(1, 2, 1)
plt.title("image")
# plt.imshow(image[:, :, 80], cmap="gray")
plt.imshow(image[:, :], cmap="gray")
plt.subplot(1, 2, 2)
plt.title("label")
# plt.imshow(label[:, :, 80])
plt.imshow(label[:, :,])
plt.show()

# Cache

# train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0, num_workers=1)
train_ds = Dataset(data=train_files, transform=train_transforms)

# use batch_size=2 to load images and use RandCropByPosNegLabeld
# to generate 2 x 4 images for network training
# train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=1)
train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)

# val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0, num_workers=1)
val_ds = Dataset(data=val_files, transform=val_transforms)
# val_loader = DataLoader(val_ds, batch_size=1, num_workers=1)
val_loader = DataLoader(val_ds, batch_size=1)

# Model

# standard PyTorch program style: create UNet, DiceLoss and Adam optimizer
device = torch.device("cpu:0")
model = UNet(
    spatial_dims=2,
    in_channels=1,
    out_channels=7,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
    norm=Norm.BATCH,
).to(device)
loss_function = DiceLoss(to_onehot_y=True, softmax=True)
optimizer = torch.optim.Adam(model.parameters(), 1e-4)
dice_metric = DiceMetric(include_background=False, reduction="mean")

# TRAINING

max_epochs = 600
val_interval = 1
best_metric = -1
best_metric_epoch = -1
epoch_loss_values = []
metric_values = []
post_pred = Compose([AsDiscrete(argmax=True, to_onehot=7)])
post_label = Compose([AsDiscrete(to_onehot=7)])

# if __name__ == '__main__':
    # freeze_support()

for epoch in range(max_epochs):
    print("-" * 10)
    print(f"epoch {epoch + 1}/{max_epochs}")
    model.train()
    epoch_loss = 0
    step = 0
    for batch_data in train_loader:
        step += 1
        inputs, labels = (
            batch_data["image"].to(device),
            batch_data["label"].to(device),
        )
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        print(f"{step}/{len(train_ds) // train_loader.batch_size}, " f"train_loss: {loss.item():.4f}")
    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

    if (epoch + 1) % val_interval == 0:
        model.eval()
        with torch.no_grad():
            for val_data in val_loader:
                val_inputs, val_labels = (
                    val_data["image"].to(device),
                    val_data["label"].to(device),
                )
                roi_size = (160, 160)
                sw_batch_size = 4
                val_outputs = sliding_window_inference(val_inputs, roi_size, sw_batch_size, model)
                val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                val_labels = [post_label(i) for i in decollate_batch(val_labels)]
                # compute metric for current iteration
                dice_metric(y_pred=val_outputs, y=val_labels)

            # aggregate the final mean dice result
            metric = dice_metric.aggregate().item()
            # reset the status for next validation round
            dice_metric.reset()

            metric_values.append(metric)
            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), os.path.join(root_dir, "best_metric_model.pth"))
                print("saved new best metric model")
            print(
                f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                f"\nbest mean dice: {best_metric:.4f} "
                f"at epoch: {best_metric_epoch}"
            )
    #%%
print(f"train completed, best_metric: {best_metric:.4f} " f"at epoch: {best_metric_epoch}")
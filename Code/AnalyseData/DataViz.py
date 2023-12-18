import os

import matplotlib.pyplot as plt
import torch
from monai.data import CacheDataset, DataLoader
from monai.inferers import sliding_window_inference
from monai.networks.layers import Norm
from monai.networks.nets import UNet

from Code.MONAI import AppliedTransforms, SlidingWindowConfig
from Code.MONAI.DataLoader import get_data_dicts
import Code.MONAI.AppliedTransforms

########################
# show masks

def show_masks(model, val_loader, device):
    with torch.no_grad():
        for i, val_data in enumerate(val_loader):
            roi_size = (512, 512)
            sw_batch_size = 1
            val_outputs = sliding_window_inference(val_data["image"].to(device), SlidingWindowConfig.slidingWindow_roi_size,
                                                   SlidingWindowConfig.batch_size, model)
            #val_outputs = val_data["image"].to(device)
            plt.figure("check", (18, 6))
            plt.subplot(1, 3, 1)
            plt.title(f"image {i}")
            plt.imshow(val_data["image"][0, 0, :, :], cmap="gray")
            plt.subplot(1, 3, 2)
            plt.title(f"label {i}")
            plt.imshow(val_data["label"][0, 0, :, :])
            plt.subplot(1, 3, 3)
            plt.title(f"output {i}")
            plt.imshow(torch.argmax(val_outputs, dim=1).detach().cpu()[0, :, :])
            plt.show()
            if i == 2:
                break


########################
# show masks

def show_elbow_plot(epoch_loss_values, val_interval, metric_values):
    plt.figure("train", (12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Epoch Average Loss")
    x = [i + 1 for i in range(len(epoch_loss_values))]
    y = epoch_loss_values
    plt.xlabel("epoch")
    plt.plot(x, y)
    plt.subplot(1, 2, 2)
    plt.title("Val Mean Dice")
    x = [val_interval * (i + 1) for i in range(len(metric_values))]
    y = metric_values
    plt.xlabel("epoch")
    plt.plot(x, y)
    plt.show()


if __name__ == "__main__":
    #### MODEL
    SPATIAL_DIMS = 2
    LABELS = [0, 1, 2, 3, 4, 5, 6]
    NUM_CLASSES = len(LABELS)
    LEARNING_RATE = 1e-4

    # UNET ARCHITECTURE
    UNET_CHANNELS = (16, 32, 64, 128, 256)
    UNET_STRIDE = 2
    UNET_STRIDES = tuple([UNET_STRIDE] * (len(UNET_CHANNELS) - 1))

    model = UNet(
        spatial_dims=SPATIAL_DIMS,
        in_channels=1,
        out_channels=NUM_CLASSES,
        channels=UNET_CHANNELS,
        strides=UNET_STRIDES,
        num_res_units=2,
        norm=Norm.BATCH,
    ).to('cuda')

    filename = 'best_metric_hausdorff_model_2023-12-09_10-18-53_BASELINE.pth'
    path_to_pth = os.path.join('..', 'MONAI', 'Model', 'Save', filename)
    model.load_state_dict(torch.load(path_to_pth))
    model.eval()

    #### FILES
    train_files, val_files = get_data_dicts()

    val_ds = CacheDataset(data=val_files, transform=AppliedTransforms.val_transforms_BASELINE, cache_rate=1.0, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=1)


    ### Outputs
    show_masks(model, val_loader, 'cuda')

   # show_elbow_plot(epoch_loss_values=model.epoch_loss_values, val_interval=1, metric_values=model.metric_values)


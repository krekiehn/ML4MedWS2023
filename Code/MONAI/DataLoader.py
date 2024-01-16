import glob, os
import matplotlib.pyplot as plt
from monai.data import CacheDataset, DataLoader, Dataset
from monai.utils import first

relative_path = r'.\..\..\Data\ConvertedPelvisNiftiDataset\Converted Pelvis Nifti Dataset'
if os.path.isdir(relative_path):
    print(relative_path)
    data_dir = relative_path
else:
    data_dir = r'.\..\..\Data\ConvertedPelvisNiftiDataset\Converted Pelvis Nifti Dataset'


def get_data_dicts(data_dir=data_dir, stop_index=None):
    print("os:" + os.getcwd())
    print("datadir: " + data_dir)
    train_images = sorted(glob.glob(os.path.join(data_dir, "train", "imgs", "*.nii.gz")))
    train_labels = sorted(glob.glob(os.path.join(data_dir, "train", "targets", "*.nii.gz")))

    print(train_images)
    train_data_dicts = [{"image": image_name, "label": label_name} for image_name, label_name in
                        zip(train_images, train_labels)]

    val_images = sorted(glob.glob(os.path.join(data_dir, "valid", "imgs", "*.nii.gz")))
    val_labels = sorted(glob.glob(os.path.join(data_dir, "valid", "targets", "*.nii.gz")))

    val_data_dicts = [{"image": image_name, "label": label_name} for image_name, label_name in
                      zip(val_images, val_labels)]

    # train_files, val_files = train_data_dicts[:-9], val_data_dicts[-9:]
    if stop_index:
        train_files, val_files = train_data_dicts[:stop_index], val_data_dicts[:stop_index]
    else:
        train_files, val_files = train_data_dicts, val_data_dicts
    # train_files, val_files = train_data_dicts[:2], val_data_dicts[:2]

    print("train files size: " + str(len(train_files)))
    print("val files size: " + str(len(val_files)))
    return train_files, val_files


def check_transforms_in_dataloader(check_ds):
    # Check transforms in DataLoader

    # check_ds = Dataset(data=val_files, transform=val_transforms)
    # check_ds = Dataset(data=train_files, transform=train_transforms)
    check_loader = DataLoader(check_ds, batch_size=1)
    # print(check_loader)
    check_data = first(check_loader)
    image, label = (check_data["image"][0][0], check_data["label"][0][0])
    print(f"image shape: {image.shape}, label shape: {label.shape}")
    # plot the slice [:, :, 80]
    # plt.figure("check", (12, 6))
    # plt.subplot(1, 2, 1)
    # plt.title("image")
    # # plt.imshow(image[:, :, 80], cmap="gray")
    # plt.imshow(image[:, :], cmap="gray")
    # plt.subplot(1, 2, 2)
    # plt.title("label")
    # # plt.imshow(label[:, :, 80])
    # plt.imshow(label[:, :,])
    # plt.show()


import numpy as np
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Orientationd, DivisiblePadd, Resized, \
    RandGaussianNoised, Rand2DElasticd, SpatialPadd, Spacingd, RandRotated, ScaleIntensityRanged, RandZoomd, \
    PadListDataCollate, RandCropd, RandCropByPosNegLabeld, RandCropByLabelClassesd

from Code.MONAI.CustomTransforms import ReplaceValuesNotInList

LABELS = [0, 1, 2, 3, 4, 5, 6]

train_transforms_BASELINE = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="LP"),
        # SpatialPadd(keys=["image", "label"], spatial_size=(512, 512)),
        PadListDataCollate(keys=["image", "label"]),
        # PadToMaxSize(keys=["image", "label"]),
        ReplaceValuesNotInList(keys=['label'], allowed_values=LABELS, replacement_value=0),
        DivisiblePadd(keys=["image", "label"], k=16),
        Resized(keys=["image", "label"], spatial_size=(1024, 1024), mode="nearest")
    ]
)
val_transforms_BASELINE = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="LP"),
        ReplaceValuesNotInList(keys=['label'], allowed_values=LABELS, replacement_value=0),
        DivisiblePadd(keys=["image", "label"], k=16),
        Resized(keys=["image", "label"], spatial_size=(1024, 1024), mode="nearest"),
        ScaleIntensityRanged(
                    keys=["image"],
                    a_min=0,
                    a_max=225,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                ),
    ]
)



val_transforms_BASELINE2 = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="LP"),
        ReplaceValuesNotInList(keys=['label'], allowed_values=[0, 1, 2, 3, 4, 5, 6], replacement_value=0),
        SpatialPadd(keys=["image", "label"], spatial_size=(2991, 2992)),
        Spacingd(keys=["image", "label"], pixdim=(10, 10), mode=("bilinear", "nearest")),
        DivisiblePadd(keys=["image", "label"], k=16, method="symmetric"),
    ]
)

train_transforms_RandGaussianNoise = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="LP"),
        # SpatialPadd(keys=["image", "label"], spatial_size=(512, 512)),
        # PadListDataCollate(keys=["image", "label"]),
        # PadToMaxSize(keys=["image", "label"]),
        ReplaceValuesNotInList(keys=['label'], allowed_values=LABELS, replacement_value=0),
        DivisiblePadd(keys=["image", "label"], k=16),
        Resized(keys=["image", "label"], spatial_size=(1024, 1024), mode="nearest"),
        RandGaussianNoised(
            # as_tensor_output=False,
            keys=['image'],
            prob=1.0,
            mean=0.0,
            std=2
        )
    ]
)

train_transforms_Elastic = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="LP"),
        # SpatialPadd(keys=["image", "label"], spatial_size=(512, 512)),
        # PadListDataCollate(keys=["image", "label"]),
        # PadToMaxSize(keys=["image", "label"]),
        ReplaceValuesNotInList(keys=['label'], allowed_values=LABELS, replacement_value=0),
        DivisiblePadd(keys=["image", "label"], k=16),
        Resized(keys=["image", "label"], spatial_size=(1024, 1024), mode="nearest"),
        Rand2DElasticd(keys=["image", "label"],
                       prob=1,
                       spacing=(20, 20),
                       magnitude_range=(1, 2),
                       padding_mode='zeros',
                       mode=['bilinear', 'nearest'])
    ]
)


train_transforms_Anna = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="LP"),
        ReplaceValuesNotInList(keys=['label'], allowed_values=[0, 1, 2, 3, 4, 5, 6], replacement_value=0),
        SpatialPadd(keys=["image", "label"], spatial_size=(2991, 2992)),
        Spacingd(keys=["image", "label"], pixdim=(10, 10), mode=("bilinear", "nearest")),
        DivisiblePadd(keys=["image", "label"], k=16, method="symmetric"),
        RandRotated(keys=["image", "label"], range_x=np.pi, prob=0.8, keep_size=True, padding_mode="zeros"),
        Rand2DElasticd(keys=["image", "label"], prob=0.6, spacing=(30, 30), magnitude_range=(0.1, 0.3)),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=0,
            a_max=225,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        RandGaussianNoised(
            keys=['image'],
            prob=0.5,
            mean=0.0,
            std=0.1
        ),
        RandZoomd(keys=["image", "label"],
                  min_zoom=0.8,
                  max_zoom=1.5,
                  prob=0.4,
                  keep_size=True,
                  padding_mode="constant",
                  mode=("bilinear", "nearest")),
    ]
)

train_transforms_NextTry = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="LP"),
        ReplaceValuesNotInList(keys=['label'], allowed_values=LABELS, replacement_value=0),
        DivisiblePadd(keys=["image", "label"], k=16),
        Resized(keys=["image", "label"], spatial_size=(1024, 1024), mode="nearest"),
        ScaleIntensityRanged(
                    keys=["image"],
                    a_min=0,
                    a_max=225,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                ),
        Rand2DElasticd(keys=["image", "label"], prob=0.6, spacing=(30, 30), magnitude_range=(0.1, 0.3)),
        RandGaussianNoised(
                    keys=['image'],
                    prob=0.5,
                    mean=0.0,
                    std=0.1
                ),
    ]
)

val_transforms_BASELINE1_2 = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="LP"),
        ReplaceValuesNotInList(keys=['label'], allowed_values=LABELS, replacement_value=0),
        #SpatialPadd(keys=["image", "label"], spatial_size=(2991, 2991)),
        Resized(keys=["image", "label"], spatial_size=(1024, 1024), mode="nearest"),
        Spacingd(keys=["image", "label"], pixdim=(2, 2), mode=("bilinear", "nearest")),
        DivisiblePadd(keys=["image", "label"], k=16),
        ScaleIntensityRanged(
                    keys=["image"],
                    a_min=0,
                    a_max=225,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                ),
    ]
)
train_transforms_NextTry3 = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="LP"),
        ReplaceValuesNotInList(keys=['label'], allowed_values=LABELS, replacement_value=0),
        SpatialPadd(keys=["image", "label"], spatial_size=(2991, 2991)),
        Spacingd(keys=["image", "label"], pixdim=(2, 2), mode=("bilinear", "nearest")),
        DivisiblePadd(keys=["image", "label"], k=16),
        ScaleIntensityRanged(
                    keys=["image"],
                    a_min=0,
                    a_max=225,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                ),
        #RandCropByLabelClassesd(
        #    keys=["image", "label"],
        #    label_key="label",
        #    spatial_size=[3, 3],
        #    ratios=[1, 2, 3, 1],
        #    num_classes=7,
        #    num_samples=2,
        #),
        #Spacingd(keys=["image", "label"], pixdim=(10, 10), mode=("bilinear", "nearest")),
        #Resized(keys=["image", "label"], spatial_size=(1024, 1024), mode="nearest"),
        #PadListDataCollate(keys=["image", "label"]),

        Rand2DElasticd(keys=["image", "label"], prob=0.6, spacing=(30, 30), magnitude_range=(0.1, 0.3), mode="nearest"),
        RandGaussianNoised(
                    keys=['image'],
                    prob=0.5,
                    mean=0.0,
                    std=0.1
                ),
        #SpatialPadd(keys=["image", "label"], spatial_size=(2992, 2992)),

    ]
)

train_transforms_NextTry4 = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="LP"),
        ReplaceValuesNotInList(keys=['label'], allowed_values=LABELS, replacement_value=0),
        RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(1024, 1024),
                    pos=1,
                    neg=1,
                    num_samples=4,
                    image_key="image",
                    image_threshold=0,
                ),
        Spacingd(keys=["image", "label"], pixdim=(2, 2), mode=("bilinear", "nearest")),
        DivisiblePadd(keys=["image", "label"], k=16),
        ScaleIntensityRanged(
                    keys=["image"],
                    a_min=0,
                    a_max=225,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                ),
        Rand2DElasticd(keys=["image", "label"], prob=0.6, spacing=(30, 30), magnitude_range=(0.1, 0.3), mode="nearest"),
        RandGaussianNoised(
                    keys=['image'],
                    prob=0.5,
                    mean=0.0,
                    std=0.1
                ),
    ]
)

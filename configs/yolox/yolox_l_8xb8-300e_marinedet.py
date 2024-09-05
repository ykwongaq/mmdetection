_base_ = "./yolox_s_8xb8-300e_coco.py"

backend_args = None
img_scale = (640, 640)  # width, height

classes = (
    "Teleostei",
    "Malacostraca",
    "Elasmobranchii",
    "Gastropoda",
    "Actinopterygii",
    "Bivalvia",
    "Reptilia",
    "Hexacorallia",
    "Aves",
    "Echinoidea",
    "Octocorallia",
    "Polychaeta",
    "Hydrozoa",
    "Chondrichthyes",
    "Demospongiae",
    "Dipneusti",
    "Cubozoa",
    "Hexapoda",
    "Ophiuroidea",
    "Coelacanthi",
    "Nuda",
    "Petromyzonti",
    "Ascidiacea",
    "Florideophyceae",
)

work_dir = "/mnt/hdd/davidwong/models/yolox/"

# model settings
model = dict(
    backbone=dict(deepen_factor=1.0, widen_factor=1.0),
    neck=dict(in_channels=[256, 512, 1024], out_channels=256, num_csp_blocks=3),
    bbox_head=dict(in_channels=256, feat_channels=256, num_classes=24),
)

# dataset settings
dataset_type = "CocoDataset"
data_root = "/mnt/hdd/davidwong/data/marinedet/"

train_pipeline = [
    dict(type="Mosaic", img_scale=img_scale, pad_val=114.0),
    dict(
        type="RandomAffine",
        scaling_ratio_range=(0.1, 2),
        # img_scale is (width, height)
        border=(-img_scale[0] // 2, -img_scale[1] // 2),
    ),
    dict(type="MixUp", img_scale=img_scale, ratio_range=(0.8, 1.6), pad_val=114.0),
    dict(type="YOLOXHSVRandomAug"),
    dict(type="RandomFlip", prob=0.5),
    # According to the official implementation, multi-scale
    # training is not considered here but in the
    # 'mmdet/models/detectors/yolox.py'.
    # Resize and Pad are for the last 15 epochs when Mosaic,
    # RandomAffine, and MixUp are closed by YOLOXModeSwitchHook.
    dict(type="Resize", scale=img_scale, keep_ratio=True),
    dict(
        type="Pad",
        pad_to_square=True,
        # If the image is three-channel, the pad value needs
        # to be set separately for each channel.
        pad_val=dict(img=(114.0, 114.0, 114.0)),
    ),
    dict(type="FilterAnnotations", min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type="PackDetInputs"),
]

train_dataset = dict(
    # use MultiImageMixDataset wrapper to support mosaic and mixup
    type="MultiImageMixDataset",
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file="annotations/class_level_train_no_negative.json",
        data_prefix=dict(img=""),
        pipeline=[
            dict(type="LoadImageFromFile", backend_args=backend_args),
            dict(type="LoadAnnotations", with_bbox=True),
        ],
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        backend_args=backend_args,
        metainfo=dict(classes=classes),
    ),
    pipeline=train_pipeline,
)

test_pipeline = [
    dict(type="LoadImageFromFile", backend_args=backend_args),
    dict(type="Resize", scale=img_scale, keep_ratio=True),
    dict(type="Pad", pad_to_square=True, pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(
        type="PackDetInputs",
        meta_keys=("img_id", "img_path", "ori_shape", "img_shape", "scale_factor"),
    ),
]


train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=train_dataset,
)
val_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file="annotations/class_level_val_seen_no_negative.json",
        data_prefix=dict(img=""),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args,
        metainfo=dict(classes=classes),
    ),
)
test_dataloader = val_dataloader

val_evaluator = dict(
    type="CocoMetric",
    ann_file=data_root + "annotations/class_level_val_seen_no_negative.json",
    metric="bbox",
    backend_args=backend_args,
)
test_evaluator = val_evaluator

# training settings
max_epochs = 300
num_last_epochs = 15
interval = 50

train_cfg = dict(max_epochs=max_epochs, val_interval=interval)
default_hooks = dict(
    checkpoint=dict(
        interval=interval, max_keep_ckpts=3  # only keep latest 3 checkpoints
    )
)

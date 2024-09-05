_base_ = "./dino-5scale_swin-l_8xb2-12e_coco.py"
work_dir = "/mnt/hdd/davidwong/models/dino"

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

model = dict(bbox_head=dict(num_classes=len(classes)))

backend_args = None
train_pipeline = [
    dict(type="LoadImageFromFile", backend_args=backend_args),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type="Resize", scale=(480, 270), keep_ratio=True),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PackDetInputs"),
]

test_pipeline = [
    dict(type="LoadImageFromFile", backend_args=backend_args),
    dict(type="Resize", scale=(480, 270), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type="LoadAnnotations", with_bbox=True),
    dict(
        type="PackDetInputs",
        meta_keys=("img_id", "img_path", "ori_shape", "img_shape", "scale_factor"),
    ),
]

dataset_type = "CocoDataset"
data_root = "/mnt/hdd/davidwong/data/marinedet/"

train_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    batch_sampler=dict(type="AspectRatioBatchSampler"),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=dict(classes=classes),
        ann_file="annotations/class_level_train_no_negative.json",
        data_prefix=dict(img=""),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args,
        serialize_data=False,
    ),
)

val_dataloader = dict(
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=dict(classes=classes),
        ann_file="annotations/class_level_val_seen_no_negative.json",
        data_prefix=dict(img=""),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args,
        serialize_data=False,
    ),
)

test_dataloader = val_dataloader

val_evaluator = dict(
    type="CocoMetric",
    ann_file=data_root + "annotations/class_level_val_seen_no_negative.json",
    metric="bbox",
    format_only=False,
    backend_args=backend_args,
)
test_evaluator = val_evaluator

max_epochs = 36
train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=max_epochs, val_interval=12)
param_scheduler = [
    dict(
        type="MultiStepLR",
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[27, 33],
        gamma=0.1,
    )
]

vis_backends = [dict(type="LocalVisBackend"), dict(type="TensorboardVisBackend")]
visualizer = dict(
    type="DetLocalVisualizer", vis_backends=vis_backends, name="visualizer"
)

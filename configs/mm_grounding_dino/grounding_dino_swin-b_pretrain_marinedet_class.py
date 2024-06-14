_base_ = "grounding_dino_swin-t_pretrain_obj365.py"

# load_from = "https://download.openmmlab.com/mmdetection/v3.0/mm_grounding_dino/grounding_dino_swin-b_pretrain_obj365_goldg_v3det/grounding_dino_swin-b_pretrain_obj365_goldg_v3de-f83eef00.pth"  # noqa
load_from = "https://download.openmmlab.com/mmdetection/v3.0/mm_grounding_dino/grounding_dino_swin-b_pretrain_all/grounding_dino_swin-b_pretrain_all-f9818a7c.pth"
model = dict(
    use_autocast=True,
    backbone=dict(
        _delete_=True,
        type="SwinTransformer",
        pretrain_img_size=384,
        embed_dims=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=12,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.3,
        patch_norm=True,
        out_indices=(1, 2, 3),
        with_cp=True,
        convert_weights=True,
        frozen_stages=-1,
        init_cfg=None,
    ),
    neck=dict(in_channels=[256, 512, 1024]),
)

marindet_dataset = dict(
    type="ODVGDataset",
    data_root="data/marinedet",
    ann_file="annotations/class_level_train_grounding_filtered_processed.json",
    label_map_file=None,
    data_prefix=dict(img="."),
    filter_cfg=dict(filter_empty_gt=False),
    pipeline=_base_.train_pipeline,
    return_classes=True,
    backend_args=None,
)


# --------------------------- dataloader---------------------------
train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(datasets=[marindet_dataset]),
)

optim_wrapper = dict(optimizer=dict(lr=0.0001))

# learning policy
max_iter = 30000
train_cfg = dict(
    _delete_=True, type="IterBasedTrainLoop", max_iters=max_iter, val_interval=10000
)

param_scheduler = [
    dict(type="LinearLR", start_factor=0.1, by_epoch=False, begin=0, end=1000),
    dict(
        type="MultiStepLR",
        begin=0,
        end=max_iter,
        by_epoch=False,
        milestones=[228510],
        gamma=0.1,
    ),
]

default_hooks = dict(checkpoint=dict(by_epoch=False, interval=5000, max_keep_ckpts=20))
log_processor = dict(by_epoch=False)

test_pipeline = [
    dict(type="LoadImageFromFile", backend_args=None, imdecode_backend="pillow"),
    dict(
        type="FixScaleResize",
        # scale=(800, 1333),
        scale=(400, 600),
        keep_ratio=True,
        backend="pillow",
    ),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(
        type="PackDetInputs",
        meta_keys=(
            "img_id",
            "img_path",
            "ori_shape",
            "img_shape",
            "scale_factor",
            "text",
            "custom_entities",
            "phrase_ids",
            "tokens_positive",
            "phrases",
        ),
    ),
]

dataset_type = "MarineDetDataset"
data_root = "data/marinedet/"

dataset_class_seen = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file="annotations/class_level_val_seen_grounding.json",
    data_prefix=dict(img="."),
    pipeline=test_pipeline,
)

dataset_class_unseen = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file="annotations/class_level_val_unseen_grounding.json",
    data_prefix=dict(img="."),
    pipeline=test_pipeline,
)

val_evaluator_Flickr30k = dict(type="Flickr30kMetric")

test_evaluator_Flickr30k = dict(type="Flickr30kMetric")

# ----------Config---------- #
dataset_prefixes = ["MarineDet_Class_Seen", "MarineDet_Class_Unseen"]
datasets = [dataset_class_seen, dataset_class_unseen]
metrics = [val_evaluator_Flickr30k, test_evaluator_Flickr30k]

val_dataloader = dict(
    dataset=dict(_delete_=True, type="ConcatDataset", datasets=datasets)
)
test_dataloader = val_dataloader

val_evaluator = dict(
    _delete_=True,
    type="MultiDatasetsEvaluator",
    metrics=metrics,
    dataset_prefixes=dataset_prefixes,
)
test_evaluator = val_evaluator

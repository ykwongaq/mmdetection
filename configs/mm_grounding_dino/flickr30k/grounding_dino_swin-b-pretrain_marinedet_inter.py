_base_ = '../grounding_dino_swin-b_pretrain_all.py'

dataset_type = 'MarineDetDataset'
data_root = 'data/marinedet/'

test_pipeline = [
    dict(
        type='LoadImageFromFile', backend_args=None,
        imdecode_backend='pillow'),
    dict(
        type='FixScaleResize',
        # scale=(800, 1333),
        scale=(400, 600),
        keep_ratio=True,
        backend='pillow'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'text', 'custom_entities', 'phrase_ids', 'tokens_positive', 'phrases'))
]

dataset_class_seen = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file='annotations/inter_class_val_seen_grounding_filtered.json',
    data_prefix=dict(img='.'),
    pipeline=test_pipeline,
)

dataset_class_unseen = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file='annotations/inter_class_val_unseen_grounding_filtered.json',
    data_prefix=dict(img='.'),
    pipeline=test_pipeline,
)

val_evaluator_Flickr30k = dict(type='Flickr30kMetric')

test_evaluator_Flickr30k = dict(type='Flickr30kMetric')

# ----------Config---------- #
dataset_prefixes = ['MarineDet_Inter_Class_Seen', 'MarineDet_Inter_Class_Unseen']
datasets = [dataset_class_seen, dataset_class_unseen]
metrics = [val_evaluator_Flickr30k, test_evaluator_Flickr30k]

val_dataloader = dict(
    dataset=dict(_delete_=True, type='ConcatDataset', datasets=datasets))
test_dataloader = val_dataloader

val_evaluator = dict(
    _delete_=True,
    type='MultiDatasetsEvaluator',
    metrics=metrics,
    dataset_prefixes=dataset_prefixes)
test_evaluator = val_evaluator

# dataset settings
# CHEST_ROOT: directory of MedFMC ChestDR images. With official MedFMC release,
# the structure is `<root>/*.png`, where filenames match those in
# `data_meta/chest_*-shot_*_exp*.txt` and `data_meta/test_WithLabel.txt`.
CHEST_ROOT = 'data/medfmc/chest/images'

dataset_type = 'Chest19'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='RandomResizedCrop',
        size=256,
        backend='pillow',
        interpolation='bicubic'),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=256, backend='pillow', interpolation='bicubic'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_prefix=CHEST_ROOT,
        ann_file='data_meta/test_WithLabel.txt',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_prefix=CHEST_ROOT,
        ann_file='data_meta/test_WithLabel.txt',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_prefix=CHEST_ROOT,
        ann_file='data_meta/test_WithLabel.txt',
        pipeline=test_pipeline))

evaluation = dict(interval=1, metric='AUC_multilabel', save_best='auto')

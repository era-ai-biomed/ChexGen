# dataset settings
# Set MIMIC_TRAIN_ROOT / MIMIC_VAL_ROOT to your local MIMIC-CXR-JPG image roots.
# The ann_file paths under data_meta/ store relative jpg paths
# (`train/p<id>/s<id>/view*.jpg`), so the prefix should be the directory whose
# subtree mirrors that layout (typically the MIMIC-CXR-JPG `files/` dir).
MIMIC_TRAIN_ROOT = 'data/mimic-cxr-jpg/p10_p18'
MIMIC_VAL_ROOT = 'data/mimic-cxr-jpg/p19'

dataset_type = 'MIMIC'
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
    samples_per_gpu=128,
    workers_per_gpu=10,
    train=dict(
        type=dataset_type,
        data_prefix=MIMIC_TRAIN_ROOT,
        ann_file='data_meta/mimic_p10_18_findings_train.txt',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_prefix=MIMIC_VAL_ROOT,
        ann_file='data_meta/mimic_p19_findings_val.txt',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_prefix=MIMIC_VAL_ROOT,
        ann_file='data_meta/mimic_p19_findings_val.txt',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='AUC_multilabel', save_best='auto')

model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='DenseNet',
        arch='121',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='weights/densenet121_4xb256_in1k_20220426-07450f99.pth',
            prefix='backbone')),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='MultiLabelLinearClsHead', num_classes=14, in_channels=1024))
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
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=256, backend='pillow', interpolation='bicubic'),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=128,
    workers_per_gpu=10,
    train=dict(
        type='MIMIC',
        data_prefix=
        '/mnt/petrelfs/jiyuanfeng.p/data/mimic-cxr/mimic-cls-p10-p18-train',
        ann_file=
        'tools/downstream/medfm/data_backup/MedFMC/mimic_cxr/mimic_p10_18_findings_train.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='RandomResizedCrop',
                size=256,
                backend='pillow',
                interpolation='bicubic'),
            dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='ToTensor', keys=['gt_label']),
            dict(type='Collect', keys=['img', 'gt_label'])
        ]),
    val=dict(
        type='MIMIC',
        data_prefix='/mnt/petrelfs/jiyuanfeng.p/data/mimic-cxr/p19',
        ann_file=
        'tools/downstream/medfm/data_backup/MedFMC/mimic_cxr/mimic_p19_findings_val.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='Resize',
                size=256,
                backend='pillow',
                interpolation='bicubic'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ]),
    test=dict(
        type='MIMIC',
        data_prefix='/mnt/petrelfs/jiyuanfeng.p/data/mimic-cxr/p19',
        ann_file=
        'tools/downstream/medfm/data_backup/MedFMC/mimic_cxr/mimic_p19_findings_val.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='Resize',
                size=256,
                backend='pillow',
                interpolation='bicubic'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ]))
evaluation = dict(interval=1, metric='AUC_multilabel', save_best='auto')
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='CosineAnnealing', min_lr=0)
runner = dict(type='EpochBasedRunner', max_epochs=15)
checkpoint_config = dict(interval=1, max_keep_ckpts=1)
log_config = dict(interval=5, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
custom_imports = dict(
    imports=[
        'medfmc.models', 'medfmc.datasets.medical_datasets',
        'medfmc.core.evaluation'
    ],
    allow_failed_imports=False)
work_dir = 'work_dirs/downsteam/cls/mimic/baseline/2'
gpu_ids = [0]

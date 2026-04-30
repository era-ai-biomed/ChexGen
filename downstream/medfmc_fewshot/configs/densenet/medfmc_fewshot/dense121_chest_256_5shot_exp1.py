_base_ = [
    '../../_base_/models/densenet/densenet121_multilabel.py',
    '../../_base_/datasets/chest_256.py', '../../_base_/schedules/imagenet_dense.py',
    '../../_base_/default_runtime.py', '../../_base_/custom_imports.py'
]

model = dict(
    backbone=dict(
        init_cfg=dict(
            type='Pretrained',
            checkpoint='weights/densenet121_4xb256_in1k_20220426-07450f99.pth',
            prefix='backbone',
        )),
    head=dict(num_classes=19))


dataset = 'chest'
nshot = 5
exp = 1
data = dict(
    samples_per_gpu=4,
    train=dict(ann_file=f'data_meta/{dataset}_{nshot}-shot_train_exp{exp}.txt'),
    val=dict(ann_file=f'data_meta/{dataset}_{nshot}-shot_val_exp{exp}.txt'),
    test=dict(ann_file=f'data_meta/test_WithLabel.txt'))
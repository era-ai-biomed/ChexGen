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
nshot = 10
exp = 4
base_path = 'data_meta'
train_val_template = f"data_meta/{{dataset}}_{{nshot}}-shot_{{split}}_exp{{exp}}.txt"

data = dict(
    samples_per_gpu=4,
    train=dict(ann_file=train_val_template.format(dataset=dataset, nshot=nshot, split='train', exp=exp)),
    val=dict(ann_file=train_val_template.format(dataset=dataset, nshot=nshot, split='val', exp=exp)),
    test=dict(ann_file=f"data_meta/test_WithLabel.txt")
)

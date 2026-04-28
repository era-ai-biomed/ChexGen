#############################################################
# Example MIMIC-CXR fine-tuning config (text -> image, 512x512).
#
# Before training, run tools/preprocess/* to extract VAE image latents
# and T5 caption embeddings, then build a data list. Edit the fields
# below to point at those outputs:
#   - data_list_file: .txt where each line is `<image_npz>\t<text_npz>`
#     (relative paths under s3_bucket).
#   - s3_bucket: directory prefix prepended to those relative paths,
#     OR an s3:// URI when file_client_args.backend == "petrel".
#############################################################

model = dict(
    type='DiT_XL_2',
    input_size=512 // 8,
    class_dropout_prob=0.1,
    grad_checkpoint=True,
    grad_cal_steps=1,
    use_fp32_attn=True,
    token_num=120,
    shift_scale_gate_table=None,  # optional PixArt-style reparam table; point at a JSON to use it
    pos_embed_scale=2.0,
)

dataloader = dict(
    batch_size=32,
    num_workers=16,
    shuffle=True,
    pin_memory=False,
    drop_last=True,
    persistent_workers=True,
    dataset=dict(
        type='T2IDataset',
        transform=None,
        data_list_file='data/meta/second_stage_impression_512.txt',
        file_client_args={'backend': 'disk'},
        s3_bucket='data/mimic_cxr_512',
    ),
)

optimizer = dict(
    lr=2e-5,
    weight_decay=3e-2,
    eps=1e-10,
)

accelerator = dict(
    mixed_precision='bf16',
    gradient_accumulation_steps=1,
    gradient_clip=0.01,
)

runner = dict(
    epochs=800,
    log_every=100,
    ckpt_every=1000,
)

pretrained = 'weights/pretrained_256.pth'
resume = None

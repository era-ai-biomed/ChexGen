# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import torch
from mmengine import Config
from tqdm import tqdm
from torch.utils.data import DataLoader, DistributedSampler
from radiffuser.datasets.t2i import TextDataset
from radiffuser.diffusion import create_diffusion
from radiffuser.models.builder import build_model
from radiffuser.models.t5 import T5Embedder
from radiffuser.utils import find_model

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from diffusers.models import AutoencoderKL
import argparse
import os
import os.path as osp

import torch.distributed as dist


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to a config file.")
    parser.add_argument("ckpt", type=str, default=None)
    parser.add_argument("--work-dir", type=str, default="work_dirs/debug-sample-ddp")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--text-prompt", nargs='+')
    parser.add_argument("--text-prompt-file", type=str, default=None)
    parser.add_argument("--text-prompt-key", type=str, default="caption")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--local-rank", type=int, default=0)
    args = parser.parse_args()
    return args


def merge_args(cfg, args):
    # merge args into cfg, if each value in args is not None
    for k, v in vars(args).items():
        if v is not None:
            cfg[k] = v
    return cfg


def main(cfg):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)

    # setup ddp
    dist.init_process_group(backend="nccl", init_method="env://")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.seed
    torch.cuda.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank {rank}, seed {seed}, world size {dist.get_world_size()}")

    # setup model
    model_cfg = cfg.get("model")
    latent_size = model_cfg.get("input_size")
    model = build_model(model_cfg).to(device)

    # load model
    ckpt_path = cfg.get("ckpt")
    model.load_state_dict(find_model(ckpt_path, revised_keys={"module.": ""}), strict=True)
    print(f"loaded checkpoint from {ckpt_path}")

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device])
    model.float()
    model.eval()  # important!

    diffusion = create_diffusion(str(cfg.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{cfg.vae}").to(device)
    # cache_dir="/scratch/m000071/yfj/weights/t5"
    text_model = T5Embedder(device=device)
    token_nums = model_cfg.get("token_num", 120)

    if cfg.get("text_prompt_file", None) is not None:
        dataset = TextDataset(cfg.text_prompt_file, text_key=cfg.text_prompt_key, name_key="name", return_name=True)
    else:
        dataset = TextDataset(cfg.text_prompt, return_name=True)

    if rank == 0:
        if not osp.exists(args.work_dir):
            os.makedirs(args.work_dir)

    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        sampler=sampler,
        drop_last=False
    )

    dist.barrier()

    if rank == 0:
        print(f"Generating {len(dataset)} samples")

    for name, text in tqdm(dataloader):
        continue_flag = True

        for n in name:
            if not os.path.exists(os.path.join(args.work_dir, n)):
                continue_flag = False
                break

        if continue_flag:
            continue
        
        #TODO check the logic 
        caption_embs, emb_masks = text_model.get_text_embeddings(text, token_nums=token_nums)
        caption_embs = caption_embs[:, None]

        null_y = model.module.y_embedder.y_embedding[None].repeat(len(text), 1, 1)[:, None]

        # Setup classifier-free guidance
        n = len(text)
        z = torch.randn(n, 4, latent_size, latent_size, device=device).repeat(2, 1, 1, 1)
        model_kwargs = dict(y=torch.cat([caption_embs, null_y]), cfg_scale=args.cfg_scale, mask=emb_masks)

        samples = diffusion.p_sample_loop(
            model.module.forward_with_cfg, 
            z.shape,
            z,
            clip_denoised=False,
            model_kwargs=model_kwargs, 
            progress=False,
            device=device
        )

        samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
        samples = vae.decode(samples / 0.18215).sample

        for i, sample in enumerate(samples):
            if ".png" not in name[i] and ".jpg" not in name[i]:
                name[i] = f"{name[i]}.png"
            image_path = osp.join(args.work_dir, name[i])
            image_dir = osp.dirname(image_path)
            if not osp.exists(image_dir):
                os.makedirs(image_dir)
            save_image(samples[i], image_path, normalize=True, value_range=(-1, 1))

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg = merge_args(cfg, args)
    main(cfg)

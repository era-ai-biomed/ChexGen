# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP.
"""
import os.path as osp
import random
import torch
from mmengine.dist import get_world_size

from radiffuser.datasets.builder import build_dataset
from radiffuser.models.builder import build_model
from radiffuser.diffusion import create_diffusion
from radiffuser.models.dit_control import ControlT2IDiT
from radiffuser.utils import find_model

import glob

from mmengine import DictAction, Config

from radiffuser.utils.checkpoint import load_checkpoint, save_checkpoint

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.utils.data import DataLoader
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from time import time
import argparse
import logging
import os
import pynvml
from accelerate import Accelerator, DistributedDataParallelKwargs


#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        name = name.replace("module.", "")
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)
    return logger


def mask_feature(emb, mask):
    masked_feature = emb * mask[:, None, :, None]
    return masked_feature, emb.shape[2]



    # if emb.shape[0] == 1:
    #     keep_index = mask.sum().item()
    #     return emb[:, :, :keep_index, :], keep_index
    # else:
    #     masked_feature = emb * mask[:, None, :, None]
    #     return masked_feature, emb.shape[2]


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(cfg):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    
    # set random seed
    set_random_seed(cfg)

    # Setup accelerator:
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator_cfg = cfg.get("accelerator", {})

    accelerator = Accelerator(
        mixed_precision=accelerator_cfg.mixed_precision,
        kwargs_handlers=[ddp_kwargs]
    )
    device = accelerator.device

    # Setup an experiment folder:
    if accelerator.is_main_process:
        init_workspace(cfg)
        logger = create_logger(cfg.work_dir)
        logger.info(cfg.pretty_text)
        logger.info(f"Running with {get_world_size()} GPUs")

    # Create model:
    model = build_model(cfg.get("model"))

    if cfg.auto_resume and not cfg.resume:
        list_of_files = glob.glob(os.path.join(cfg.work_dir, '*.pth'))
        if list_of_files:
            latest_file = max(list_of_files, key=os.path.getctime)
            cfg.resume = latest_file
            cfg.pretrained = None

    if cfg.pretrained:
        missing, unexpected = load_checkpoint(cfg.pretrained, model, del_umap=True)
        if accelerator.is_main_process:
            logger.info(f"Missing keys: {missing}")
            logger.info(f"Unexpected keys: {unexpected}")

    model = ControlT2IDiT(model).to(device)
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule
    if accelerator.is_main_process:
        logger.info(f"Control DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt_cfg = cfg.get("optimizer")
    opt = torch.optim.AdamW(model.parameters(), **opt_cfg)

    dataset_cfg = cfg.get("dataloader").pop("dataset", None)
    dataset = build_dataset(dataset_cfg)
    loader = DataLoader(dataset=dataset, **cfg.get("dataloader"))

    if accelerator.is_main_process:
        logger.info(f"Dataset contains {len(dataset):,} images")

    # Prepare models for training:
    update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # resume training
    if cfg.resume:
        (train_steps, start_epoch), missing, unexpected = load_checkpoint(
            cfg.resume,
            model=model,
            model_ema=ema,
            optimizer=opt,
            load_ema=True,
            del_umap=False,
            resume_optimizer=True
        )
        if accelerator.is_main_process:
            logger.info(f"Resuming from {cfg.resume}, starting at iteration {train_steps} epoch {start_epoch}")

    else:
        start_epoch = 0
        train_steps = 0

    model, opt, loader = accelerator.prepare(model, opt, loader)

    # Variables for monitoring/logging purposes:
    runner_cfg = cfg.get("runner")
    epoches = runner_cfg.get("epochs")
    log_steps = 0
    running_loss = 0
    start_time = time()
    if accelerator.is_main_process:
        pynvml.nvmlInit()

    if accelerator.is_main_process:
        logger.info(f"Training for {epoches} epochs...")

    for epoch in range(start_epoch + 1, epoches + 1):
        if accelerator.is_main_process:
            logger.info(f"Beginning epoch {epoch}...")
        with accelerator.accumulate(model):
            for x, (y, y_mask), c in loader:
                x = x.to(device)
                t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
                y = y[:, None]
                masked_y, keep_index = mask_feature(y, y_mask)
                masked_y = masked_y.to(device)

                loss_dict = diffusion.training_losses(model, x, t, dict(y=masked_y, mask=y_mask, c=c))
                loss = loss_dict["loss"].mean()
                opt.zero_grad()
                accelerator.backward(loss)
                opt.step()
                update_ema(ema, model)

                # Log loss values:
                running_loss += loss.item()
                log_steps += 1
                train_steps += 1
                if train_steps % runner_cfg.log_every == 0:
                    log_steps += 1
                    # Measure training speed:
                    torch.cuda.synchronize()
                    end_time = time()
                    steps_per_sec = log_steps / (end_time - start_time)

                    # Reduce loss history over all processes:
                    avg_loss = torch.tensor(running_loss / log_steps, device=device)
                    avg_loss = avg_loss.item()

                    if accelerator.is_main_process:
                        handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # Assuming only one GPU is used
                        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        memory_allocated = memory_info.used / 1024 ** 3  # Convert to GB
                        memory_total = memory_info.total / 1024 ** 3  # Convert to GB
                        logger.info(
                            f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}, "
                            f"Memory Allocated: {memory_allocated:.2f} GB, Memory Total: {memory_total:.2f} GB")

                    # Reset monitoring variables:
                    running_loss = 0
                    log_steps = 0
                    start_time = time()

                # Save DiT checkpoint:
                if train_steps % runner_cfg.ckpt_every == 0 and train_steps > 0:
                    if accelerator.is_main_process:
                        save_checkpoint(
                            work_dir=cfg.work_dir,
                            epoch=epoch,
                            model=model,
                            model_ema=ema,
                            optimizer=opt,
                            step=train_steps,
                            keep_last=True,
                        )

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    if accelerator.is_main_process:
        pynvml.nvmlShutdown()
        logger.info("Done!")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='config file path')
    parser.add_argument('--work-dir', type=str, help='work directory')
    parser.add_argument('--cfg-options', nargs='+',
                        action=DictAction, help='override some settings in the used config')
    parser.add_argument('--random_seed', type=int, default=0, help='random seed')
    parser.add_argument('--resume', type=str, default=None, help='resume from the checkpoint')
    parser.add_argument('--auto-resume', action='store_true', default=True, help='auto resume from the latest checkpoint')    
    parser.add_argument('--pretrained', type=str, default=None, help='pretrained model')
    parser.add_argument('--local-rank', type=int, default=-1)
    args = parser.parse_args()
    return args


def merge_args(cfg, args):
    if args.work_dir:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(args.config))[0])
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    if args.random_seed:
        cfg.random_seed = args.random_seed
    if cfg.get('random_seed', None) is None:
        cfg.random_seed = 0
    if args.pretrained is not None:
        cfg.pretrained = args.pretrained
    if args.resume:
        cfg.resume = args.resume
        cfg.pretrained = None
    if args.auto_resume:
        cfg.auto_resume = args.auto_resume
        
    return cfg


def init_workspace(cfg):
    cfg_filename = osp.basename(cfg.filename)
    if not osp.exists(cfg.work_dir):
        os.makedirs(cfg.work_dir, exist_ok=True)
    cfg.dump(osp.join(cfg.work_dir, cfg_filename))


def set_random_seed(cfg):
    random.seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)
    torch.manual_seed(cfg.random_seed)


if __name__ == "__main__":
    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg = merge_args(cfg, args)

    main(cfg)

import argparse
import os
import os.path as osp
import sys
import warnings

ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), osp.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module=r"torch\.utils\.checkpoint")
warnings.filterwarnings("ignore", category=UserWarning, module=r"torch\._dynamo\.eval_frame")

from huggingface_hub.utils import disable_progress_bars
disable_progress_bars()

import pandas as pd
import torch
from mmengine import Config
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchvision import transforms as T
from radiffuser.diffusion import create_diffusion
from radiffuser.models.builder import build_model
from radiffuser.models.dit_control import ControlT2IDiT
from radiffuser.models.t5 import T5Embedder

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from diffusers.models import AutoencoderKL

import torch.distributed as dist


class TextCondDataset(Dataset):
    def __init__(self, csv_path, text_key, cond_key, name_key="name",
                 cond_dir=None, image_size=512):
        df = pd.read_csv(csv_path)
        for col in (text_key, cond_key):
            if col not in df.columns:
                raise ValueError(f"column '{col}' not found in {csv_path}")
        df = df[df[text_key].notna() & df[cond_key].notna()].reset_index(drop=True)
        self.names = df[name_key].tolist() if name_key in df.columns else [str(i) for i in range(len(df))]
        self.texts = df[text_key].tolist()
        self.cond_paths = df[cond_key].tolist()
        self.cond_dir = cond_dir
        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        name = self.names[idx]
        text = self.texts[idx]
        cond_path = self.cond_paths[idx]
        if self.cond_dir is not None and not osp.isabs(cond_path):
            cond_path = osp.join(self.cond_dir, cond_path)
        cond_img = Image.open(cond_path).convert("RGB")
        cond = self.transform(cond_img)
        return name, text, cond


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to a config file.")
    parser.add_argument("ckpt", type=str, default=None)
    parser.add_argument("--checkpoint-key", type=str, default="state_dict_ema")
    parser.add_argument("--work-dir", type=str, default="work_dirs/debug-sample-control")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--text-prompt-file", type=str, required=True)
    parser.add_argument("--text-prompt-key", type=str, default="impression")
    parser.add_argument("--cond-key", type=str, required=True,
                        help="CSV column with mask image path.")
    parser.add_argument("--cond-dir", type=str, default=None,
                        help="Base dir prepended to relative cond paths.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--local-rank", type=int, default=0)
    args = parser.parse_args()
    return args


def merge_args(cfg, args):
    for k, v in vars(args).items():
        if v is not None:
            cfg[k] = v
    return cfg


def load_state_dict_from_checkpoint(ckpt_path, checkpoint_key):
    if not osp.isfile(ckpt_path):
        raise FileNotFoundError(f"Could not find checkpoint at {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    if checkpoint_key not in checkpoint:
        available = ", ".join(checkpoint.keys()) if isinstance(checkpoint, dict) else f"<{type(checkpoint).__name__}>"
        raise KeyError(f"Checkpoint {ckpt_path} missing key {checkpoint_key!r}. Available: {available}")
    state_dict = checkpoint[checkpoint_key]
    return {
        key[len("module."):] if key.startswith("module.") else key: value
        for key, value in state_dict.items()
    }


def main(cfg):
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)

    dist.init_process_group(backend="nccl", init_method="env://")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    torch.cuda.manual_seed(args.seed)
    torch.cuda.set_device(device)
    print(f"Starting rank {rank}, seed {args.seed}, world size {dist.get_world_size()}")

    if args.batch_size > 1 and rank == 0:
        print(
            f"[warn] batch_size={args.batch_size}: cross-attention is routed through "
            f"xformers BlockDiagonalMask, whose reduction order differs from the single-"
            f"segment path. Per-step drift is ~1e-3 relative; over 100 sampling steps this "
            f"may visibly degrade fidelity vs batch_size=1."
        )

    # Build base DiT and wrap with ControlT2IDiT (force grad_checkpoint off for inference)
    model_cfg = cfg.get("model")
    model_cfg["grad_checkpoint"] = False
    latent_size = model_cfg.get("input_size")
    image_size = latent_size * 8

    base_model = build_model(model_cfg)
    model = ControlT2IDiT(base_model).to(device)

    ckpt_path = cfg.get("ckpt")
    checkpoint_key = cfg.get("checkpoint_key")
    state_dict = load_state_dict_from_checkpoint(ckpt_path, checkpoint_key)
    model.load_state_dict(state_dict, strict=True)
    print(f"loaded checkpoint key {checkpoint_key!r} from {ckpt_path}")

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device])
    model.float()
    model.eval()

    diffusion = create_diffusion(str(cfg.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{cfg.vae}").to(device)
    text_model = T5Embedder(device=device)
    token_nums = model_cfg.get("token_num", 120)

    dataset = TextCondDataset(
        csv_path=cfg.text_prompt_file,
        text_key=cfg.text_prompt_key,
        cond_key=cfg.cond_key,
        cond_dir=cfg.get("cond_dir", None),
        image_size=image_size,
    )

    if rank == 0 and not osp.exists(args.work_dir):
        os.makedirs(args.work_dir)

    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        sampler=sampler,
        drop_last=False,
    )

    dist.barrier()
    if rank == 0:
        print(f"Generating {len(dataset)} samples")

    local_entries = []
    for name, text, cond_img in tqdm(dataloader):
        # Skip already-rendered names in this batch
        if all(osp.exists(osp.join(args.work_dir, n if (n.endswith(".png") or n.endswith(".jpg")) else f"{n}.png"))
               for n in name):
            continue

        cond_img = cond_img.to(device)
        cond_latent = vae.encode(cond_img).latent_dist.sample().mul_(0.18215)
        # double for cfg-style batched forward (ControlT2IDiT.forward_with_cfg duplicates x's first half)
        cond_latent = torch.cat([cond_latent, cond_latent], dim=0)

        caption_embs, emb_masks = text_model.get_text_embeddings(list(text), token_nums=token_nums)
        caption_embs = caption_embs[:, None]

        null_y = model.module.y_embedder.y_embedding[None].repeat(len(text), 1, 1)[:, None]

        n = len(text)
        z = torch.randn(n, 4, latent_size, latent_size, device=device).repeat(2, 1, 1, 1)
        model_kwargs = dict(
            y=torch.cat([caption_embs, null_y]),
            c=cond_latent,
            mask=emb_masks,
            cfg_scale=args.cfg_scale,
        )

        samples = diffusion.p_sample_loop(
            model.module.forward_with_cfg,
            z.shape,
            z,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            progress=False,
            device=device,
        )

        samples, _ = samples.chunk(2, dim=0)
        samples = vae.decode(samples / 0.18215).sample

        for i in range(len(name)):
            fname = name[i] if (name[i].endswith(".png") or name[i].endswith(".jpg")) else f"{name[i]}.png"
            image_path = osp.join(args.work_dir, fname)
            image_dir = osp.dirname(image_path)
            if image_dir and not osp.exists(image_dir):
                os.makedirs(image_dir, exist_ok=True)
            save_image(samples[i], image_path, normalize=True, value_range=(-1, 1))
            # save the conditioning mask alongside so generated <-> input pairing is obvious
            stem, ext = osp.splitext(fname)
            mask_path = osp.join(args.work_dir, f"{stem}_mask{ext}")
            save_image(cond_img[i].cpu(), mask_path, normalize=True, value_range=(-1, 1))
            local_entries.append((fname, text[i].strip()))

    dist.barrier()
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local_entries)
    if rank == 0:
        all_entries = sorted({e for chunk in gathered for e in chunk})
        with open(osp.join(args.work_dir, "prompts.txt"), "w") as fp:
            for fname, prompt_text in all_entries:
                fp.write(f"{fname}\t{prompt_text}\n")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg = merge_args(cfg, args)
    main(cfg)

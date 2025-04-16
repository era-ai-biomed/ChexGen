# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
import json
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import xformers.ops as xops
from mmengine import to_2tuple
from timm.models.vision_transformer import PatchEmbed
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------
from torch.jit import Final

# from radiffuser.builder import MODELS
from .builder import MODELS
from radiffuser.models.block import PixArtMSBlock, T2IFinalLayer
from radiffuser.models.embedder import CaptionEmbedder, TimestepEmbedder
from radiffuser.models.utils import auto_grad_checkpoint, set_grad_checkpoint


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def mask_feature(emb, mask):
    if emb.shape[0] == 1:
        keep_index = mask.sum().item()
        return emb[:, :, :keep_index, :], keep_index
    else:
        masked_feature = emb * mask[:, None, :, None]
        return masked_feature, emb.shape[2]


#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(
            self,
            input_size=32,
            patch_size=2,
            in_channels=4,
            hidden_size=1152,
            depth=28,
            num_heads=16,
            mlp_ratio=4.0,
            class_dropout_prob=0,
            text_embed_dims=4096,
            learn_sigma=True,
            drop_path: float = 0.,
            window_size=0,
            window_block_indexes=[],
            use_rel_pos=False,
            grad_checkpoint=True,
            grad_cal_steps=1,
            use_fp32_attn=False,
            token_num=120,
            cap_embedd_file=None,
            shift_scale_gate_table=None,
            pos_embed_scale=1.0,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.pos_embed_scale = pos_embed_scale
        approx_gelu = lambda: nn.GELU(approximate="tanh")

        self.x_embedder = PatchEmbed(
            input_size, 
            patch_size, 
            in_channels, 
            hidden_size,
            bias=True
            )
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = CaptionEmbedder(
            in_channels=text_embed_dims,
            hidden_size=hidden_size,
            uncond_prob=class_dropout_prob, 
            act_layer=approx_gelu, 
            token_num=token_num,
            cap_embedd_file=cap_embedd_file
            )
        num_patches = self.x_embedder.num_patches
        self.base_size = input_size // self.patch_size
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        #
        if shift_scale_gate_table is not None:
            logger = logging.getLogger(__name__)
            logger.info("Loading shift_scale_gate_table from %s" % shift_scale_gate_table)

            with open(shift_scale_gate_table, "r") as f:
                shift_scale_gate_table = json.load(f)
        else:
            shift_scale_gate_table = {str(i): None for i in range(depth+1)}

        drop_path = [x.item() for x in torch.linspace(0, drop_path, depth)]
        self.blocks = nn.ModuleList([
            PixArtMSBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, drop_path=drop_path[i],
                          input_size=(input_size // patch_size, input_size // patch_size),
                          window_size=window_size if i in window_block_indexes else 0,
                          use_rel_pos=use_rel_pos if i in window_block_indexes else False,
                          shift_scale_gate_table=shift_scale_gate_table[str(i)], )
            for i in range(depth)
        ])
        self.final_layer = T2IFinalLayer(hidden_size, patch_size, self.out_channels, \
                                         shift_scale_table=shift_scale_gate_table[str(depth)])

        self.t_block = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

        self.initialize_weights()
        if grad_checkpoint:
            set_grad_checkpoint(self, use_fp32_attn, grad_cal_steps)

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5), \
                                            scale=self.pos_embed_scale, base_size=self.base_size)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.t_block[1].weight, std=0.02)

        # Initialize label y embedding MLP:
        nn.init.normal_(self.y_embedder.y_proj.fc1.weight, std=0.02)
        nn.init.normal_(self.y_embedder.y_proj.fc2.weight, std=0.02)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

        # # initialize the cross attention layer
        for block in self.blocks:
            nn.init.constant_(block.cross_attn.proj.weight, 0)
            nn.init.constant_(block.cross_attn.proj.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t, y, mask=None, **kwargs):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2 [B, 1024, 1152]
        t = self.t_embedder(t)  # (N, D) [B, 1152]
        y = self.y_embedder(y, self.training)  # (N, D)

        t0 = self.t_block(t)

        # if self.training:
        #     mask = mask.squeeze(1).squeeze(1)
        #     y = y.squeeze(1).masked_select(mask.unsqueeze(-1) != 0).view(1, -1, x.shape[-1])
        #     y_lens = mask.sum(dim=1).tolist()
        # else:
        #     y_lens = [y.shape[2]] * y.shape[0]
        #     y = y.squeeze(1).view(1, -1, x.shape[-1])
        if mask is not None:
            if mask.shape[0] != y.shape[0]: # [2, 120] [0,:] [1,:] -> 
                mask = mask.repeat(y.shape[0] // mask.shape[0], 1)
            mask = mask.squeeze(1).squeeze(1)
            y = y.squeeze(1).masked_select(mask.unsqueeze(-1) != 0).view(1, -1, x.shape[-1])
            y_lens = mask.sum(dim=1).tolist()
        else:
            y_lens = [y.shape[2]] * y.shape[0]
            y = y.squeeze(1).view(1, -1, x.shape[-1])

        for block in self.blocks:
            x = auto_grad_checkpoint(block, x, y, t0, y_lens, **kwargs)
        x = self.final_layer(x, t)  # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)  # (N, out_channels, H, W)
        return x

    # def forward_with_cfg(self, x, t, y, cfg_scale, **kwargs):
    #     """
    #     Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
    #     """
    #     # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
    #     half = x[: len(x) // 2]
    #     combined = torch.cat([half, half], dim=0)
    #     model_out = self.forward(combined, t, y, **kwargs)
    #     # For exact reproducibility reasons, we apply classifier-free guidance on only
    #     # three channels by default. The standard approach to cfg applies it to all channels.
    #     # This can be done by uncommenting the following line and commenting-out the line following that.
    #     # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
    #     eps, rest = model_out[:, :3], model_out[:, 3:]
    #     cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
    #     half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
    #     eps = torch.cat([half_eps, half_eps], dim=0)
    #     return torch.cat([eps, rest], dim=1)

    def forward_with_cfg(self, x, timestep, y, cfg_scale, mask=None, **kwargs):
        """
        Forward pass of PixArt, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, timestep, y, mask, **kwargs)
        model_out = model_out['x'] if isinstance(model_out, dict) else model_out
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0, scale=1.0, base_size=16):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    if isinstance(grid_size, int):
        grid_size = to_2tuple(grid_size)
    grid_h = np.arange(grid_size[0], dtype=np.float32) / (grid_size[0] / base_size) / scale
    grid_w = np.arange(grid_size[1], dtype=np.float32) / (grid_size[1] / base_size) / scale
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size[1], grid_size[0]])

    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   DiT Configs                                  #
#################################################################################
@MODELS.register_module()
def DiT_XL_2(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)


@MODELS.register_module()
def DiT_XL_4(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)


@MODELS.register_module()
def DiT_XL_8(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)


@MODELS.register_module()
def DiT_L_2(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)


@MODELS.register_module()
def DiT_L_4(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)


@MODELS.register_module()
def DiT_L_8(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)


@MODELS.register_module()
def DiT_B_2(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)


@MODELS.register_module()
def DiT_B_4(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)


@MODELS.register_module()
def DiT_B_8(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)


@MODELS.register_module()
def DiT_S_2(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)


@MODELS.register_module()
def DiT_S_4(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)


@MODELS.register_module()
def DiT_S_8(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)


DiT_models = {
    'DiT-XL/2': DiT_XL_2, 'DiT-XL/4': DiT_XL_4, 'DiT-XL/8': DiT_XL_8,
    'DiT-L/2': DiT_L_2, 'DiT-L/4': DiT_L_4, 'DiT-L/8': DiT_L_8,
    'DiT-B/2': DiT_B_2, 'DiT-B/4': DiT_B_4, 'DiT-B/8': DiT_B_8,
    'DiT-S/2': DiT_S_2, 'DiT-S/4': DiT_S_4, 'DiT-S/8': DiT_S_8,
}

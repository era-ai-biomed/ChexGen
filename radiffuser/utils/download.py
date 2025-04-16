# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Functions for downloading pre-trained DiT models
"""
from torchvision.datasets.utils import download_url
import torch
import os


pretrained_models = {'DiT-XL-2-512x512.pt', 'DiT-XL-2-256x256.pt'}


def find_model(model_name, revised_keys=None):
    """
    Finds a pre-trained DiT model, downloading it if necessary. Alternatively, loads a model from a local path.
    :param model_name: Name of the model or the path to the model checkpoint.
    :param revised_keys: A dictionary specifying the substrings in keys that should be replaced. For example,
                         {'module.': ''} would remove 'module.' from all keys.
    """
    if model_name in pretrained_models:  # Find/download our pre-trained DiT checkpoints
        checkpoint = download_model(model_name)
    else:  # Load a custom DiT checkpoint:
        assert os.path.isfile(model_name), f'Could not find DiT checkpoint at {model_name}'
        checkpoint = torch.load(model_name, map_location=lambda storage, loc: storage)
        if "state_dict" in checkpoint:  # supports checkpoints from train.py
            checkpoint = checkpoint["state_dict"]
        elif "ema" in checkpoint:  # supports checkpoints from train.py
            checkpoint = checkpoint["ema"]

    if revised_keys and isinstance(revised_keys, dict):
        new_checkpoint = {}
        for key, value in checkpoint.items():
            new_key = key
            for old_key, new_key_replacement in revised_keys.items():
                new_key = new_key.replace(old_key, new_key_replacement)
            new_checkpoint[new_key] = value
        checkpoint = new_checkpoint

    return checkpoint


def download_model(model_name):
    """
    Downloads a pre-trained DiT model from the web.
    """
    assert model_name in pretrained_models
    local_path = f'pretrained_models/{model_name}'
    if not os.path.isfile(local_path):
        os.makedirs('pretrained_models', exist_ok=True)
        web_path = f'https://dl.fbaipublicfiles.com/DiT/models/{model_name}'
        download_url(web_path, 'pretrained_models')
    model = torch.load(local_path, map_location=lambda storage, loc: storage)
    return model


if __name__ == "__main__":
    # Download all DiT checkpoints
    for model in pretrained_models:
        download_model(model)
    print('Done.')
import warnings
from mmengine.registry import Registry

MODELS = Registry('models')
DATASETS = Registry('datasets')

def build_model(cfg):
    """Build model."""
    return MODELS.build(cfg)

def build_dataset(cfg):
    """Build dataset."""
    return DATASETS.build(cfg)



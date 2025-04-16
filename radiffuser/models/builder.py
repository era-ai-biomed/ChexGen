import warnings
from mmengine.registry import Registry

MODELS = Registry('models')

def build_model(cfg):
    """Build model."""
    return MODELS.build(cfg)




"""
Models module for LookBench
Fashion Image Retrieval Benchmark
"""

from .registry import (
    ModelRegistry,
    registry,
    register_model,
    get_model,
    list_available_models,
    is_model_registered
)
from .factory import ModelFactory
from .base import BaseModel, BaseModelWrapper

# Import model implementations to register them
# These imports trigger the @register_model decorators
try:
    from . import clip_model, siglip_model, dinov2_model
except ImportError:
    # Models may not be available in all environments
    pass

__all__ = [
    'ModelRegistry',
    'registry',
    'register_model',
    'get_model',
    'list_available_models',
    'is_model_registered',
    'ModelFactory',
    'BaseModel',
    'BaseModelWrapper'
]


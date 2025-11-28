"""
Datasets module for LookBench
Fashion Image Retrieval Benchmark
Provides BEIR-style organization for fashion retrieval datasets
"""

from .base import BaseDataset, BaseDataLoader
from .registry import (
    DatasetRegistry,
    registry,
    register_dataset,
    get_dataset,
    list_available_datasets
)

__all__ = [
    'BaseDataset',
    'BaseDataLoader',
    'DatasetRegistry',
    'registry',
    'register_dataset',
    'get_dataset',
    'list_available_datasets'
]


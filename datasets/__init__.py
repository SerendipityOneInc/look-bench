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
from .zooclaw_dataset import (
    ZooClawImageDataset,
    ZooClawTextDataset,
    load_zooclaw_task,
    load_zooclaw_dataset,
)

__all__ = [
    'BaseDataset',
    'BaseDataLoader',
    'DatasetRegistry',
    'registry',
    'register_dataset',
    'get_dataset',
    'list_available_datasets',
    'ZooClawImageDataset',
    'ZooClawTextDataset',
    'load_zooclaw_task',
    'load_zooclaw_dataset',
]


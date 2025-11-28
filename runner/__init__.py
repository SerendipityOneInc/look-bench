"""
Runner module for LookBench
Provides flexible pipeline execution for evaluation and other tasks
"""

from .evaluator import Evaluator
from .runner import Runner
from .pipeline import (
    PipelineRegistry,
    get_pipeline,
    register_pipeline,
    list_available_pipelines,
    create_pipeline
)
from .base_pipeline import BasePipeline

# Import pipelines to register them
from . import evaluation_pipeline
from . import feature_extraction_pipeline

__all__ = [
    'Runner',
    'Evaluator',
    'PipelineRegistry',
    'get_pipeline',
    'register_pipeline',
    'list_available_pipelines',
    'create_pipeline',
    'BasePipeline'
]


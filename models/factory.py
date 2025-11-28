"""
Model Factory for LookBench
Fashion Image Retrieval Benchmark
"""

from typing import Any, Tuple, Optional, Dict, Union
import torch.nn as nn
import logging
from utils.logging import get_logger, log_structured, log_error_with_context
from .registry import ModelRegistry, get_model, list_available_models
from .base import BaseModel

logger = get_logger(__name__)


class ModelFactory:
    """Professional model factory for creating and managing model instances"""

    @staticmethod
    def create_model(model_type: str, model_name: str, model_path: Optional[str] = None, **kwargs) -> Tuple[Union[nn.Module, Any], BaseModel]:
        """
        Create model instance

        Args:
            model_type: Model type identifier (e.g., 'clip', 'siglip', 'dinov2')
            model_name: Model name for loading
            model_path: Optional path to model checkpoint
            **kwargs: Additional arguments to pass to model loading

        Returns:
            Tuple of (model_object, model_instance)

        Raises:
            ValueError: If model type is not found
            RuntimeError: If model creation fails
        """
        try:
            if not ModelRegistry.is_registered(model_type):
                available_models = list_available_models()
                raise ValueError(f"Model type '{model_type}' not found. Available models: {available_models}")

            model_class = get_model(model_type)
            log_structured(logger, logging.INFO, "Creating model",
                         model_type=model_type, model_name=model_name, model_path=model_path)

            model_object, model_instance = model_class.load_model(model_name, model_path, **kwargs)

            log_structured(logger, logging.INFO, "Model successfully created",
                         model_type=model_type, model_name=model_name)
            return model_object, model_instance

        except Exception as e:
            log_error_with_context(logger, f"Error creating model '{model_type}'", e,
                                model_type=model_type, model_name=model_name, model_path=model_path)
            raise RuntimeError(f"Failed to create model '{model_type}': {e}") from e

    @staticmethod
    def get_available_models() -> list:
        """Get all available model types"""
        return list_available_models()

    @staticmethod
    def get_available_models_with_metadata() -> Dict[str, Dict[str, Any]]:
        """Get all available models with their metadata"""
        return ModelRegistry.list_models_with_metadata()

    @staticmethod
    def get_transform(model_type: str, input_size: int):
        """
        Get transform for specified model type

        Args:
            model_type: Model type identifier
            input_size: Input image size

        Returns:
            Transform pipeline

        Raises:
            ValueError: If model type is not found
        """
        if not ModelRegistry.is_registered(model_type):
            available_models = list_available_models()
            raise ValueError(f"Model type '{model_type}' not found. Available models: {available_models}")

        model_class = get_model(model_type)
        return model_class.get_transform(input_size)

    @staticmethod
    def get_model_info(model_type: str) -> Dict[str, Any]:
        """
        Get information about a specific model type

        Args:
            model_type: Model type identifier

        Returns:
            Model metadata dictionary
        """
        if not ModelRegistry.is_registered(model_type):
            available_models = list_available_models()
            raise ValueError(f"Model type '{model_type}' not found. Available models: {available_models}")

        return ModelRegistry.get_model_metadata(model_type)

    @staticmethod
    def validate_model_type(model_type: str) -> bool:
        """
        Validate if a model type is registered

        Args:
            model_type: Model type identifier

        Returns:
            True if model type is valid, False otherwise
        """
        return ModelRegistry.is_registered(model_type)

    @staticmethod
    def get_model_count() -> int:
        """Get total number of available models"""
        return ModelRegistry.get_model_count()


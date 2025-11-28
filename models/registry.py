"""
Model Registry for LookBench
Fashion Image Retrieval Benchmark
"""

from typing import Dict, Type, Any, Optional, List
from abc import ABC
from utils.logging import get_logger, log_structured
import logging
from .base import BaseModel

logger = get_logger(__name__)


class ModelRegistry:
    """Professional model registry for managing all LookBench model types"""

    _models: Dict[str, Type['BaseModel']] = {}
    _metadata: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def register(cls, name: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Decorator to register a model class

        Args:
            name: Unique identifier for the model
            metadata: Optional metadata about the model
        """
        def decorator(model_class: Type['BaseModel']):
            if not issubclass(model_class, ABC):
                raise TypeError(f"Model class {model_class.__name__} must inherit from BaseModel")

            if name in cls._models:
                log_structured(logger, logging.WARNING, "Model already registered, overwriting",
                             model_name=name, previous_type=cls._models[name].__name__)

            cls._models[name] = model_class
            cls._metadata[name] = metadata or {}

            # Add model type to metadata if not present
            if 'model_type' not in cls._metadata[name]:
                cls._metadata[name]['model_type'] = model_class.get_model_type()

            log_structured(logger, logging.INFO, "Model successfully registered",
                         model_name=name, model_type=cls._metadata[name]['model_type'],
                         model_class=model_class.__name__)
            return model_class
        return decorator

    @classmethod
    def get_model(cls, name: str) -> Type['BaseModel']:
        """
        Get a registered model class by name

        Args:
            name: Model identifier

        Returns:
            Model class

        Raises:
            ValueError: If model is not found
        """
        if name not in cls._models:
            available_models = list(cls._models.keys())
            raise ValueError(f"Model '{name}' not found. Available models: {available_models}")
        return cls._models[name]

    @classmethod
    def list_models(cls) -> List[str]:
        """Get list of all registered model names"""
        return list(cls._models.keys())

    @classmethod
    def list_models_with_metadata(cls) -> Dict[str, Dict[str, Any]]:
        """Get all registered models with their metadata"""
        return cls._metadata.copy()

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if a model is registered"""
        return name in cls._models

    @classmethod
    def get_model_metadata(cls, name: str) -> Dict[str, Any]:
        """Get metadata for a specific model"""
        if name not in cls._models:
            raise ValueError(f"Model '{name}' not found")
        return cls._metadata.get(name, {})

    @classmethod
    def unregister(cls, name: str) -> bool:
        """
        Unregister a model

        Args:
            name: Model identifier

        Returns:
            True if model was unregistered, False if not found
        """
        if name in cls._models:
            del cls._models[name]
            if name in cls._metadata:
                del cls._metadata[name]
            log_structured(logger, logging.INFO, "Model unregistered",
                         model_name=name)
            return True
        return False

    @classmethod
    def clear(cls):
        """Clear all registered models"""
        count = len(cls._models)
        cls._models.clear()
        cls._metadata.clear()
        log_structured(logger, logging.INFO, "All registered models cleared",
                     previous_count=count)

    @classmethod
    def get_model_count(cls) -> int:
        """Get total number of registered models"""
        return len(cls._models)


# Global registry instance
registry = ModelRegistry()

# Convenience functions
def register_model(name: str, metadata: Optional[Dict[str, Any]] = None):
    """Convenience function to register a model"""
    return registry.register(name, metadata)

def get_model(name: str) -> Type['BaseModel']:
    """Convenience function to get a model"""
    return registry.get_model(name)

def list_available_models() -> List[str]:
    """Convenience function to list available models"""
    return registry.list_models()

def is_model_registered(name: str) -> bool:
    """Convenience function to check if a model is registered"""
    return registry.is_registered(name)


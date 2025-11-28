"""
Pipeline Registry for LookBench
Manages different pipeline types and allows easy extension
"""

from typing import Dict, Type, Optional
import logging

from .base_pipeline import BasePipeline
from utils.logging import get_logger, log_structured

logger = get_logger(__name__)


class PipelineRegistry:
    """Registry for managing different pipeline types"""

    _pipelines: Dict[str, Type[BasePipeline]] = {}

    @classmethod
    def register(cls, name: str):
        """
        Decorator to register a pipeline class

        Args:
            name: Unique identifier for the pipeline
        """
        def decorator(pipeline_class: Type[BasePipeline]):
            if not issubclass(pipeline_class, BasePipeline):
                raise TypeError(
                    f"Pipeline class {pipeline_class.__name__} must inherit from BasePipeline")

            if name in cls._pipelines:
                log_structured(logger, logging.WARNING, "Pipeline already registered, overwriting",
                             pipeline_name=name,
                             previous_type=cls._pipelines[name].__name__)

            cls._pipelines[name] = pipeline_class

            log_structured(logger, logging.INFO, "Pipeline successfully registered",
                         pipeline_name=name,
                         pipeline_class=pipeline_class.__name__)
            return pipeline_class
        return decorator

    @classmethod
    def get_pipeline(cls, name: str) -> Type[BasePipeline]:
        """
        Get a registered pipeline class by name

        Args:
            name: Pipeline identifier

        Returns:
            Pipeline class

        Raises:
            ValueError: If pipeline is not found
        """
        if name not in cls._pipelines:
            available_pipelines = list(cls._pipelines.keys())
            raise ValueError(
                f"Pipeline '{name}' not found. Available pipelines: {available_pipelines}")
        return cls._pipelines[name]

    @classmethod
    def list_pipelines(cls) -> list:
        """Get list of all registered pipeline names"""
        return list(cls._pipelines.keys())

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if a pipeline is registered"""
        return name in cls._pipelines

    @classmethod
    def create_pipeline(
        cls,
        name: str,
        config_manager,
        model_manager,
        data_manager
    ) -> BasePipeline:
        """
        Create a pipeline instance

        Args:
            name: Pipeline identifier
            config_manager: Configuration manager
            model_manager: Model manager
            data_manager: Data manager

        Returns:
            Pipeline instance
        """
        pipeline_class = cls.get_pipeline(name)
        return pipeline_class(config_manager, model_manager, data_manager)


# Global registry instance
registry = PipelineRegistry()

# Convenience functions
def register_pipeline(name: str):
    """Convenience function to register a pipeline"""
    return registry.register(name)

def get_pipeline(name: str) -> Type[BasePipeline]:
    """Convenience function to get a pipeline"""
    return registry.get_pipeline(name)

def list_available_pipelines() -> list:
    """Convenience function to list available pipelines"""
    return registry.list_pipelines()

def create_pipeline(name: str, config_manager, model_manager, data_manager) -> BasePipeline:
    """Convenience function to create a pipeline instance"""
    return registry.create_pipeline(name, config_manager, model_manager, data_manager)


"""
Dataset Registry for LookBench
Fashion Image Retrieval Benchmark
BEIR-style dataset organization
"""

from typing import Dict, Type, Any, Optional, List
from utils.logging import get_logger, log_structured
import logging

logger = get_logger(__name__)


class DatasetRegistry:
    """Registry for managing fashion retrieval datasets"""

    _datasets: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def register(cls, name: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Register a dataset configuration

        Args:
            name: Unique identifier for the dataset
            metadata: Metadata about the dataset
        """
        if name in cls._datasets:
            log_structured(logger, logging.WARNING, "Dataset already registered, overwriting",
                         dataset_name=name)

        cls._datasets[name] = metadata or {}
        
        log_structured(logger, logging.INFO, "Dataset successfully registered",
                     dataset_name=name, metadata=metadata)
        
        return metadata

    @classmethod
    def get_dataset(cls, name: str) -> Dict[str, Any]:
        """
        Get dataset configuration by name

        Args:
            name: Dataset identifier

        Returns:
            Dataset configuration

        Raises:
            ValueError: If dataset is not found
        """
        if name not in cls._datasets:
            available_datasets = list(cls._datasets.keys())
            raise ValueError(f"Dataset '{name}' not found. Available datasets: {available_datasets}")
        return cls._datasets[name]

    @classmethod
    def list_datasets(cls) -> List[str]:
        """Get list of all registered dataset names"""
        return list(cls._datasets.keys())

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if a dataset is registered"""
        return name in cls._datasets

    @classmethod
    def get_dataset_metadata(cls, name: str) -> Dict[str, Any]:
        """Get metadata for a specific dataset"""
        if name not in cls._datasets:
            raise ValueError(f"Dataset '{name}' not found")
        return cls._datasets.get(name, {})

    @classmethod
    def unregister(cls, name: str) -> bool:
        """
        Unregister a dataset

        Args:
            name: Dataset identifier

        Returns:
            True if dataset was unregistered, False if not found
        """
        if name in cls._datasets:
            del cls._datasets[name]
            log_structured(logger, logging.INFO, "Dataset unregistered",
                         dataset_name=name)
            return True
        return False

    @classmethod
    def clear(cls):
        """Clear all registered datasets"""
        count = len(cls._datasets)
        cls._datasets.clear()
        log_structured(logger, logging.INFO, "All registered datasets cleared",
                     previous_count=count)


# Global registry instance
registry = DatasetRegistry()

# Convenience functions
def register_dataset(name: str, metadata: Optional[Dict[str, Any]] = None):
    """Convenience function to register a dataset"""
    return registry.register(name, metadata)

def get_dataset(name: str) -> Dict[str, Any]:
    """Convenience function to get a dataset"""
    return registry.get_dataset(name)

def list_available_datasets() -> List[str]:
    """Convenience function to list available datasets"""
    return registry.list_datasets()


# Register standard fashion datasets
register_dataset("fashion200k", {
    "description": "Fashion200K dataset for fashion image retrieval",
    "num_categories": 200000,
    "tasks": ["image_retrieval", "fashion_search"],
    "splits": ["query", "gallery"]
})

register_dataset("deepfashion", {
    "description": "DeepFashion dataset for fashion understanding",
    "num_categories": 50,
    "tasks": ["image_retrieval", "attribute_prediction", "landmark_detection"],
    "splits": ["query", "gallery"]
})

register_dataset("deepfashion2", {
    "description": "DeepFashion2 dataset with detailed annotations",
    "num_categories": 13,
    "tasks": ["image_retrieval", "segmentation", "keypoint_detection"],
    "splits": ["query", "gallery"]
})

register_dataset("fashion_product", {
    "description": "Fashion Product Images dataset",
    "num_categories": 44,
    "tasks": ["image_retrieval", "classification"],
    "splits": ["query", "gallery"]
})

register_dataset("product10k", {
    "description": "Product10K dataset for product retrieval",
    "num_categories": 10000,
    "tasks": ["image_retrieval", "product_search"],
    "splits": ["query", "gallery"]
})


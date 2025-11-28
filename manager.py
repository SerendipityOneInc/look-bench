"""
Manager classes for LookBench
Fashion Image Retrieval Benchmark
Handles configuration, models, and data management
"""

import torch
import numpy as np
import yaml
import os
import logging
from typing import Dict, Any, Tuple, List
from tqdm import tqdm
from utils.logging import get_logger, log_structured, log_error_with_context
from models import ModelFactory, list_available_models
from datasets import BaseDataLoader


class ConfigManager:
    """Manages configuration loading and validation"""

    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self.load_config()

    def load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        logger = get_logger(__name__)
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            log_structured(logger, logging.INFO, "Configuration loaded successfully",
                           config_path=self.config_path, config_keys=list(config.keys()))
            return config
        except Exception as e:
            log_error_with_context(logger, "Failed to load configuration", e,
                                   config_path=self.config_path)
            raise

    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """Get configuration for a specific model"""
        if model_name not in self.config:
            raise ValueError(f"Model {model_name} not found in configuration")
        return self.config[model_name]

    def get_global_config(self) -> Dict[str, Any]:
        """Get global configuration"""
        return self.config.get('global', {})

    def get_evaluation_config(self) -> Dict[str, Any]:
        """Get evaluation configuration"""
        return self.config.get('evaluation', {})

    def get_dataset_config(self) -> Dict[str, Any]:
        """Get dataset configuration"""
        return self.config.get('datasets', {})

    def get_metric(self) -> str:
        """Get evaluation metric"""
        return self.config.get('evaluation', {}).get('metric', 'recall')


class ModelManager:
    """Manages model loading and operations based on configuration"""

    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.factory = ModelFactory()
        self.global_config = config_manager.get_global_config()

        # Set environment variables for cache directories
        self.setup_environment()

    def setup_environment(self):
        """Setup environment variables for cache directories"""
        logger = get_logger(__name__)
        cache_dir = self.global_config.get('cache_dir')
        if cache_dir:
            # Expand user path
            cache_dir = os.path.expanduser(cache_dir)
            os.environ["HF_HOME"] = cache_dir
            os.environ["TRANSFORMERS_CACHE"] = cache_dir
            os.environ["HF_DATASETS_CACHE"] = cache_dir
            os.environ["HF_HUB_CACHE"] = cache_dir
            os.environ["TORCH_HOME"] = cache_dir
            log_structured(logger, logging.INFO, "Cache directories configured",
                           cache_dir=cache_dir)

    def load_model(self, model_name: str) -> Tuple[Any, Any]:
        """Load a specific model based on configuration"""
        logger = get_logger(__name__)
        model_config = self.config_manager.get_model_config(model_name)

        if not model_config.get('enabled', True):
            raise ValueError(f"Model {model_name} is disabled in configuration")

        log_structured(logger, logging.INFO, "Loading model",
                       model_name=model_name, model_config=model_config)

        try:
            # Load model using factory
            model_object, model_instance = self.factory.create_model(
                model_type=model_name,
                model_name=model_config.get('model_name'),
                model_path=model_config.get('model_path')
            )

            # Move to device if specified
            device = model_config.get('device', 'cpu')
            if device == 'cuda' and torch.cuda.is_available():
                model_object = model_object.cuda()
                log_structured(logger, logging.INFO, "Model moved to CUDA",
                               model_name=model_name, device="cuda")
            elif device == 'auto':
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                model_object = model_object.to(device)
                log_structured(logger, logging.INFO, "Model moved to auto-selected device",
                               model_name=model_name, device=device)
            else:
                log_structured(logger, logging.INFO, "Model device configured",
                               model_name=model_name, device=device)

            return model_object, model_instance

        except Exception as e:
            log_error_with_context(logger, f"Failed to load model {model_name}", e,
                                   model_name=model_name, model_config=model_config)
            raise

    def get_transform(self, model_name: str):
        """Get transform for a specific model"""
        model_config = self.config_manager.get_model_config(model_name)
        input_size = model_config['input_size']
        return self.factory.get_transform(model_name, input_size)

    def get_available_models(self) -> list:
        """Get list of available models from registry"""
        return list_available_models()

    def is_model_enabled(self, model_name: str) -> bool:
        """Check if a model is enabled in configuration"""
        try:
            model_config = self.config_manager.get_model_config(model_name)
            return model_config.get('enabled', True)
        except:
            return False

    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get configuration information for a specific model"""
        return self.config_manager.get_model_config(model_name)


class DataManager:
    """Manages data loading based on configuration"""

    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.data_loader_manager = BaseDataLoader(config_manager.config)

    def load_dataset(self, dataset_type: str, transform, **kwargs) -> Dict[str, Any]:
        """Load dataset using configuration"""
        return self.data_loader_manager.load_data(dataset_type, transform, **kwargs)

    def create_dataloaders(self, datasets: Dict[str, Any], **kwargs) -> Dict[str, torch.utils.data.DataLoader]:
        """Create PyTorch DataLoaders with configuration parameters"""
        return self.data_loader_manager.create_dataloaders(datasets, **kwargs)

    def get_dataloader_config(self) -> Dict[str, Any]:
        """Get dataloader configuration parameters"""
        dataset_config = self.config_manager.get_dataset_config()
        return {
            'batch_size': dataset_config.get('batch_size', 32),
            'num_workers': dataset_config.get('num_workers', 4),
            'shuffle': dataset_config.get('shuffle', False),
            'pin_memory': dataset_config.get('pin_memory', True),
            'drop_last': dataset_config.get('drop_last', False)
        }

    def get_available_datasets(self) -> List[str]:
        """Get list of available dataset types"""
        return self.data_loader_manager.get_available_datasets()

    def get_dataset_info(self, dataset_type: str) -> Dict[str, Any]:
        """Get information about a specific dataset type"""
        return self.data_loader_manager.get_dataset_info(dataset_type)

    def extract_features(
        self,
        model: torch.nn.Module,
        dataset,
        batch_size: int,
        num_workers: int,
        **kwargs
    ) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Extract features from dataset using the model

        Args:
            model: The model to extract features from
            dataset: Dataset to extract features from
            batch_size: Batch size for processing
            num_workers: Number of workers for data loading

        Returns:
            Tuple of (features, labels)
        """
        if isinstance(model, torch.nn.Module):
            model.eval()

        dataloader_kwargs = {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "drop_last": False,
            "shuffle": False
        }

        dataloader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)

        features = None
        labels = []
        idx = 0

        with torch.no_grad():
            for batch_data in tqdm(dataloader, desc="Extracting features", unit="batch"):
                if isinstance(batch_data, (list, tuple)):
                    images, batch_labels = batch_data
                else:
                    images = batch_data
                    batch_labels = None

                # Move to GPU if available
                if torch.cuda.is_available():
                    images = images.cuda()

                embeddings = model(images)
                embedding_size = embeddings.size(1)

                if features is None:
                    size = [len(dataset), embedding_size]
                    features = torch.zeros(*size, device=embeddings.device)

                features[idx:idx + embeddings.size(0)] = embeddings

                if batch_labels is not None:
                    labels.append(np.array(batch_labels))
                idx += embeddings.size(0)

        features = features.cpu()
        if labels:
            labels = np.concatenate(labels, axis=0)
        else:
            labels = np.arange(len(dataset))

        return features, labels


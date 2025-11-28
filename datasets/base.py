"""
Base dataset classes for LookBench
Fashion Image Retrieval Benchmark
"""

from typing import Dict, Any, List, Tuple, Optional, Iterator
from PIL import Image
import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import pyarrow.parquet as pq
from utils.logging import get_logger, log_structured
import logging

logger = get_logger(__name__)


class BaseDataset(Dataset):
    """Base dataset class for fashion image retrieval"""
    
    def __init__(
        self,
        root: str,
        mode: str,
        transform=None,
        parquet_path: Optional[str] = None,
        image_column: str = "image",
        label_column: str = "label",
        streaming: bool = False,
        label_mapping: Optional[Dict[int, int]] = None
    ):
        """
        Initialize base dataset
        
        Args:
            root: Root directory for the dataset
            mode: Dataset mode ('query' or 'gallery')
            transform: Image transformation pipeline
            parquet_path: Path to parquet file containing dataset metadata
            image_column: Name of column containing image paths
            label_column: Name of column containing labels
            streaming: Whether to use streaming mode
            label_mapping: Optional mapping for label transformation
        """
        super().__init__()
        
        self.root = root
        self.mode = mode
        self.transform = transform
        self.parquet_path = parquet_path
        self.image_column = image_column
        self.label_column = label_column
        self.streaming = streaming
        self.label_mapping = label_mapping
        
        # Load dataset
        if not streaming:
            self.dataset = self._load_parquet_data()
            self._num_rows = len(self.dataset)
        else:
            self.dataset = None
            self._num_rows = None
    
    def _load_parquet_data(self) -> pd.DataFrame:
        """Load data from parquet file"""
        if not self.parquet_path:
            raise ValueError("parquet_path must be specified")
        
        if not os.path.exists(self.parquet_path):
            self.parquet_path = os.path.join(self.root, self.mode, self.parquet_path)
            if not os.path.exists(self.parquet_path):
                raise FileNotFoundError(f"Parquet file not found: {self.parquet_path}")

        log_structured(logger, logging.INFO, f"Loading {self.mode} parquet data", 
                     data_type=self.mode, parquet_path=self.parquet_path)
        df = pq.read_table(self.parquet_path).to_pandas()
        log_structured(logger, logging.INFO, f"Parquet {self.mode} data loaded successfully", 
                     data_type=self.mode, row_count=len(df))
        return df
    
    def _decode_image(self, image_data) -> Image.Image:
        """Decode image from various formats"""
        if isinstance(image_data, str):
            # Image path
            image_path = os.path.join(self.root, image_data)
            if not os.path.exists(image_path):
                # Try without root
                if os.path.exists(image_data):
                    image_path = image_data
                else:
                    raise FileNotFoundError(f"Image not found: {image_path}")
            return Image.open(image_path).convert('RGB')
        elif isinstance(image_data, bytes):
            # Binary image data
            import io
            return Image.open(io.BytesIO(image_data)).convert('RGB')
        else:
            # Assume it's already a PIL Image
            return image_data
    
    def _process_label(self, label) -> int:
        """Process label with optional mapping"""
        if self.label_mapping:
            return self.label_mapping.get(label, label)
        return label
    
    def __len__(self) -> int:
        if self.streaming:
            raise TypeError("Streaming dataset has no static length")
        return self._num_rows
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        if self.streaming:
            raise TypeError("__getitem__ is not supported in streaming mode; iterate instead")
        
        row = self.dataset.iloc[index]
        img = self._decode_image(row[self.image_column])
        
        if self.transform is not None:
            img = self.transform(img)
        
        target = self._process_label(row[self.label_column])
        return img, target
    
    def __iter__(self) -> Iterator:
        if not self.streaming:
            return super().__iter__()
        
        # Create streaming iterator
        df = self._load_parquet_data()
        
        for _, row in df.iterrows():
            img = self._decode_image(row[self.image_column])
            if self.transform is not None:
                img = self.transform(img)
            target = self._process_label(row[self.label_column])
            yield img, target
    
    def nb_classes(self) -> int:
        """Get number of unique classes"""
        if self.dataset is not None:
            return self.dataset[self.label_column].nunique()
        return 0


class BaseDataLoader:
    """Data loader manager for creating datasets and dataloaders"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize data loader
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.dataset_config = config.get('datasets', {})
    
    def load_data(self, dataset_type: str, transform, **kwargs) -> Dict[str, Any]:
        """
        Load dataset based on dataset_type
        
        Args:
            dataset_type: Type of dataset (e.g., 'fashion200k', 'deepfashion')
            transform: Image transformation pipeline
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing query and gallery datasets
        """
        # Get dataset configuration
        if dataset_type not in self.dataset_config:
            available_datasets = [k for k in self.dataset_config.keys() 
                                if k not in ['batch_size', 'num_workers', 'shuffle', 'pin_memory', 'drop_last']]
            raise ValueError(f"Dataset type '{dataset_type}' not found. Available: {available_datasets}")
        
        dataset_config = self.dataset_config[dataset_type]
        data_root = dataset_config.get('data_root', 'data')
        splits = dataset_config.get('splits', {'query': 'query', 'gallery': 'gallery'})
        
        # Get parquet file paths
        parquet_files = dataset_config.get('parquet_files', {})
        query_parquet = parquet_files.get('query')
        gallery_parquet = parquet_files.get('gallery')
        
        # Create datasets
        query = BaseDataset(
            root=data_root,
            mode=splits['query'],
            transform=transform,
            parquet_path=query_parquet,
            **kwargs
        )
        
        gallery = BaseDataset(
            root=data_root,
            mode=splits['gallery'], 
            transform=transform,
            parquet_path=gallery_parquet,
            **kwargs
        )
        
        return {
            "query": query,
            "gallery": gallery
        }
    
    def get_available_datasets(self) -> List[str]:
        """Get list of available dataset types"""
        return [k for k in self.dataset_config.keys() 
                if k not in ['batch_size', 'num_workers', 'shuffle', 'pin_memory', 'drop_last']]
    
    def get_dataset_info(self, dataset_type: str) -> Dict[str, Any]:
        """Get information about a specific dataset type"""
        if dataset_type not in self.dataset_config:
            available_datasets = self.get_available_datasets()
            raise ValueError(f"Dataset type '{dataset_type}' not found. Available: {available_datasets}")
        
        dataset_config = self.dataset_config[dataset_type]
        return {
            'data_root': dataset_config.get('data_root'),
            'splits': dataset_config.get('splits'),
            'parquet_files': dataset_config.get('parquet_files', {}),
            'type': dataset_type
        }
    
    def create_dataloaders(self, datasets: Dict[str, Any], **kwargs) -> Dict[str, DataLoader]:
        """
        Create PyTorch DataLoaders from datasets
        
        Args:
            datasets: Dictionary containing query and gallery datasets
            **kwargs: Override config parameters
            
        Returns:
            Dictionary containing query and gallery DataLoaders
        """
        # Get dataloader parameters
        batch_size = kwargs.get('batch_size', self.dataset_config.get('batch_size', 32))
        num_workers = kwargs.get('num_workers', self.dataset_config.get('num_workers', 4))
        shuffle = kwargs.get('shuffle', self.dataset_config.get('shuffle', False))
        pin_memory = kwargs.get('pin_memory', self.dataset_config.get('pin_memory', True))
        drop_last = kwargs.get('drop_last', self.dataset_config.get('drop_last', False))
        
        query_loader = DataLoader(
            datasets['query'], 
            batch_size=batch_size, 
            num_workers=num_workers, 
            shuffle=shuffle,
            pin_memory=pin_memory,
            drop_last=drop_last
        )
        
        gallery_loader = DataLoader(
            datasets['gallery'], 
            batch_size=batch_size, 
            num_workers=num_workers, 
            shuffle=shuffle,
            pin_memory=pin_memory,
            drop_last=drop_last
        )
        
        return {
            "query": query_loader,
            "gallery": gallery_loader
        }


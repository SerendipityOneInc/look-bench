"""
Hugging Face Dataset Loader for LookBench
Handles loading datasets from Hugging Face Hub
"""

import logging
from typing import Optional, Union, Dict, Any
from .logging import get_logger, log_structured

logger = get_logger(__name__)


def load_lookbench_dataset(
    config_name: str,
    split: Optional[str] = None,
    cache_dir: Optional[str] = None,
    **kwargs
) -> Any:
    """
    Load LookBench dataset from Hugging Face Hub
    
    This function properly handles the import of the Hugging Face datasets library
    to avoid shadowing by the local datasets module.
    
    Args:
        config_name: Config name for the dataset. One of:
            - 'real_studio_flat': Real studio flat-lay photos
            - 'aigen_studio': AI-generated lifestyle images
            - 'real_streetlook': Real street outfit photos
            - 'aigen_streetlook': AI-generated street outfits
            - 'noise': Noise/distractor images
        split: Optional split name ('query', 'gallery', etc.)
        cache_dir: Optional cache directory path
        **kwargs: Additional arguments to pass to load_dataset
    
    Returns:
        Dataset or DatasetDict from Hugging Face
    
    Example:
        >>> dataset = load_lookbench_dataset('real_studio_flat')
        >>> query_data = dataset['query']
        >>> gallery_data = dataset['gallery']
    
    Example with specific split:
        >>> query_data = load_lookbench_dataset('real_studio_flat', split='query')
    """
    # Import from the correct datasets package (Hugging Face)
    try:
        # Use importlib to ensure we get the HF datasets, not local one
        import importlib.util
        import sys
        
        # Find the HF datasets package
        spec = importlib.util.find_spec('datasets')
        if spec is None:
            raise ImportError("Hugging Face datasets package not found. Install with: pip install datasets")
        
        # Import load_dataset from the HF package
        datasets_module = importlib.util.module_from_spec(spec)
        sys.modules.setdefault('_hf_datasets', datasets_module)
        spec.loader.exec_module(datasets_module)
        load_dataset = datasets_module.load_dataset
        
    except Exception as e:
        logger.error(f"Failed to import Hugging Face datasets library: {e}")
        raise ImportError(
            "Failed to import Hugging Face datasets library. "
            "Please install it with: pip install datasets>=2.14.0"
        ) from e
    
    # LookBench repository on Hugging Face
    repo_id = "srpone/look-bench"
    
    # Validate config name
    valid_configs = ['real_studio_flat', 'aigen_studio', 'real_streetlook', 
                     'aigen_streetlook', 'noise']
    if config_name not in valid_configs:
        raise ValueError(
            f"Invalid config_name: {config_name}. "
            f"Must be one of: {valid_configs}"
        )
    
    try:
        log_structured(logger, logging.INFO, "Loading LookBench dataset",
                      config_name=config_name, split=split)
        
        # Load dataset
        dataset = load_dataset(
            repo_id,
            config_name,
            split=split,
            cache_dir=cache_dir,
            **kwargs
        )
        
        log_structured(logger, logging.INFO, "LookBench dataset loaded successfully",
                      config_name=config_name, split=split)
        
        return dataset
        
    except Exception as e:
        log_structured(logger, logging.ERROR, "Failed to load LookBench dataset",
                      config_name=config_name, split=split, error=str(e))
        raise RuntimeError(
            f"Failed to load LookBench dataset with config '{config_name}'. "
            f"Error: {e}"
        ) from e


def load_all_lookbench_configs(
    cache_dir: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Load all LookBench dataset configs
    
    Args:
        cache_dir: Optional cache directory path
        **kwargs: Additional arguments to pass to load_dataset
    
    Returns:
        Dictionary mapping config names to datasets
    
    Example:
        >>> datasets = load_all_lookbench_configs()
        >>> real_studio = datasets['real_studio_flat']
        >>> query_data = real_studio['query']
    """
    configs = ['real_studio_flat', 'aigen_studio', 'real_streetlook', 
               'aigen_streetlook', 'noise']
    
    log_structured(logger, logging.INFO, "Loading all LookBench configs",
                  configs=configs)
    
    datasets = {}
    for config_name in configs:
        try:
            datasets[config_name] = load_lookbench_dataset(
                config_name=config_name,
                cache_dir=cache_dir,
                **kwargs
            )
        except Exception as e:
            logger.warning(f"Failed to load config '{config_name}': {e}")
            continue
    
    log_structured(logger, logging.INFO, "Loaded LookBench configs",
                  loaded_configs=list(datasets.keys()))
    
    return datasets


def get_available_configs() -> list:
    """
    Get list of available LookBench dataset configs
    
    Returns:
        List of config names
    """
    return ['real_studio_flat', 'aigen_studio', 'real_streetlook', 
            'aigen_streetlook', 'noise']


def get_config_info() -> Dict[str, Dict[str, Any]]:
    """
    Get information about each LookBench config
    
    Returns:
        Dictionary mapping config names to metadata
    """
    return {
        'real_studio_flat': {
            'description': 'Real studio flat-lay photos',
            'difficulty': 'Easy',
            'image_type': 'real',
            'setting': 'studio'
        },
        'aigen_studio': {
            'description': 'AI-generated lifestyle images',
            'difficulty': 'Medium',
            'image_type': 'ai_generated',
            'setting': 'studio'
        },
        'real_streetlook': {
            'description': 'Real street outfit photos',
            'difficulty': 'Hard',
            'image_type': 'real',
            'setting': 'street'
        },
        'aigen_streetlook': {
            'description': 'AI-generated street outfits',
            'difficulty': 'Hard',
            'image_type': 'ai_generated',
            'setting': 'street'
        },
        'noise': {
            'description': 'Noise/distractor images',
            'difficulty': 'N/A',
            'image_type': 'noise',
            'setting': 'N/A'
        }
    }

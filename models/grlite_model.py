"""
GR-Lite Model for LookBench
Fashion Image Search Model
"""

import torch
import torch.nn as nn
from typing import Tuple, Any
from torchvision import transforms
import logging
from utils.logging import get_logger, log_structured
from .base import BaseModel
from .registry import register_model

logger = get_logger(__name__)


@register_model("gr-lite", metadata={
    "description": "GR-Lite: Fashion image search model based on DINOv3",
    "framework": "PyTorch",
    "input_size": 518,
    "embedding_dim": 256,
    "model_family": "vision_transformer",
    "huggingface_repo": "srpone/gr-lite"
})
class GRLiteModel(BaseModel):
    """
    GR-Lite fashion image search model
    Based on DINOv3 with custom training for fashion domain
    """

    @classmethod
    def load_model(cls, model_name: str = "srpone/gr-lite", model_path: str = None, **kwargs) -> Tuple[nn.Module, 'GRLiteModel']:
        """
        Load GR-Lite model from Hugging Face

        Args:
            model_name: Model name or path (default: srpone/gr-lite)
            model_path: Optional local path to model checkpoint
            **kwargs: Additional arguments

        Returns:
            Tuple of (model, model_instance)
        """
        from transformers import AutoModel, AutoConfig

        try:
            log_structured(logger, logging.INFO, "Loading GR-Lite model",
                         model_name=model_name, model_path=model_path)

            # Load config first
            config = AutoConfig.from_pretrained(
                model_path or model_name,
                trust_remote_code=True
            )

            # Set config options
            if hasattr(config, 'is_crop'):
                config.is_crop = False

            # Load model
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = AutoModel.from_pretrained(
                model_path or model_name,
                config=config,
                trust_remote_code=True,
                device_map=device
            )

            model.eval()

            # Create wrapper that uses model's search method
            class GRLiteWrapper(nn.Module):
                def __init__(self, base_model):
                    super().__init__()
                    self.model = base_model

                def forward(self, x):
                    """
                    Forward pass for feature extraction
                    
                    Args:
                        x: Input tensor (batch of images) or list of PIL images
                    
                    Returns:
                        Embeddings tensor (batch_size, 256)
                    """
                    # If input is a tensor, we assume it's already preprocessed
                    # GR-Lite expects PIL images via .search() method
                    # This wrapper is for tensor-based pipelines
                    
                    if isinstance(x, torch.Tensor):
                        # For tensor input, use the underlying model directly
                        with torch.no_grad():
                            outputs = self.model.model(x)
                            # Extract features from DINOv3 output
                            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                                features = outputs.pooler_output
                            elif hasattr(outputs, 'last_hidden_state'):
                                # Use CLS token
                                features = outputs.last_hidden_state[:, 0]
                            elif isinstance(outputs, dict):
                                features = outputs.get('pooler_output', outputs.get('last_hidden_state'))
                                if len(features.shape) == 3:
                                    features = features[:, 0]
                            else:
                                features = outputs
                                if len(features.shape) == 3:
                                    features = features[:, 0]
                        return features
                    else:
                        # For PIL images, use the search method
                        _, embeddings = self.model.search(image_paths=x, feature_dim=256)
                        if isinstance(embeddings, torch.Tensor):
                            return embeddings
                        return torch.tensor(embeddings)

                def extract_from_images(self, images):
                    """
                    Extract features from PIL images using model's search method
                    
                    Args:
                        images: List of PIL images
                    
                    Returns:
                        numpy array of embeddings
                    """
                    # Convert to RGB if needed
                    rgb_images = [img.convert("RGB") if img.mode != "RGB" else img for img in images]
                    
                    with torch.no_grad():
                        _, batch_embeddings = self.model.search(image_paths=rgb_images, feature_dim=256)
                    
                    if isinstance(batch_embeddings, torch.Tensor):
                        return batch_embeddings.cpu().numpy()
                    return batch_embeddings

            wrapped_model = GRLiteWrapper(model)

            log_structured(logger, logging.INFO, "GR-Lite model loaded successfully",
                         model_name=model_name, device=device)

            return wrapped_model, cls()

        except Exception as e:
            log_structured(logger, logging.ERROR, "Failed to load GR-Lite model",
                         model_name=model_name, error=str(e))
            raise RuntimeError(f"Failed to load GR-Lite model: {e}") from e

    @classmethod
    def get_transform(cls, input_size: int = 518):
        """
        Get preprocessing transform for GR-Lite model
        
        Note: GR-Lite has built-in preprocessing via .search() method.
        This transform is for tensor-based pipelines.
        
        Args:
            input_size: Input image size (default: 518)
        
        Returns:
            Transform pipeline
        """
        return transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    @classmethod
    def get_model_type(cls) -> str:
        """Get model type identifier"""
        return "gr-lite"

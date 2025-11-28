"""
SigLIP Model Implementation for LookBench
Fashion Image Retrieval Benchmark
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple
from transformers import AutoModel, AutoProcessor
from torchvision import transforms
import logging

from .base import BaseModel, BaseModelWrapper
from .registry import register_model
from utils.logging import get_logger, log_structured

logger = get_logger(__name__)


@register_model("siglip", metadata={
    "description": "SigLIP (Sigmoid Loss for Language-Image Pre-training) model for fashion image retrieval",
    "framework": "PyTorch",
    "default_input_size": 224,
    "default_embedding_dim": 768,
    "architecture": "Vision Transformer"
})
class SigLIPEmbeddingModel(BaseModel):
    """SigLIP model for fashion image embedding extraction"""

    @classmethod
    def load_model(cls, model_name: str, model_path: Optional[str] = None) -> Tuple[nn.Module, 'SigLIPEmbeddingModel']:
        """
        Load SigLIP model

        Args:
            model_name: HuggingFace SigLIP model name (e.g., 'google/siglip-base-patch16-224')
            model_path: Optional path to checkpoint (not used for SigLIP)

        Returns:
            Tuple of (wrapped_model, model_instance)
        """
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()
        
        wrapped_model = SigLIPModelWrapper(model, processor.image_processor)
        
        log_structured(logger, logging.INFO, "SigLIP model loaded successfully",
                     model_name=model_name, device=device)
        
        return wrapped_model, cls()

    @classmethod
    def get_transform(cls, input_size: int, is_train: bool = False):
        """Get transform for SigLIP model"""
        transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                               std=[0.5, 0.5, 0.5])
        ])
        return transform


class SigLIPModelWrapper(BaseModelWrapper):
    """Wrapper for SigLIP vision model"""
    
    def get_embeddings(self, inputs) -> torch.Tensor:
        """Extract embeddings using SigLIP vision model"""
        with torch.no_grad():
            if isinstance(inputs, dict) and 'pixel_values' in inputs:
                embeddings = self.model.get_image_features(inputs['pixel_values'])
            else:
                embeddings = self.model.get_image_features(**inputs)
            
            # Normalize embeddings
            import torch.nn.functional as F
            embeddings = F.normalize(embeddings, p=2, dim=1)
            return embeddings


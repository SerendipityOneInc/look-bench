"""
CLIP Model Implementation for LookBench
Fashion Image Retrieval Benchmark
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple
from transformers import CLIPModel, CLIPProcessor
from torchvision import transforms
import logging

from .base import BaseModel, BaseModelWrapper
from .registry import register_model
from utils.logging import get_logger, log_structured

logger = get_logger(__name__)


@register_model("clip", metadata={
    "description": "CLIP (Contrastive Language-Image Pre-training) model for fashion image retrieval",
    "framework": "PyTorch",
    "default_input_size": 224,
    "default_embedding_dim": 512,
    "architecture": "Vision Transformer"
})
class CLIPEmbeddingModel(BaseModel):
    """CLIP model for fashion image embedding extraction"""

    @classmethod
    def load_model(cls, model_name: str, model_path: Optional[str] = None) -> Tuple[nn.Module, 'CLIPEmbeddingModel']:
        """
        Load CLIP model

        Args:
            model_name: HuggingFace CLIP model name (e.g., 'openai/clip-vit-base-patch16')
            model_path: Optional path to checkpoint (not used for CLIP)

        Returns:
            Tuple of (wrapped_model, model_instance)
        """
        model = CLIPModel.from_pretrained(model_name)
        processor = CLIPProcessor.from_pretrained(model_name)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()
        
        wrapped_model = CLIPModelWrapper(model, processor.image_processor)
        
        log_structured(logger, logging.INFO, "CLIP model loaded successfully",
                     model_name=model_name, device=device)
        
        return wrapped_model, cls()

    @classmethod
    def get_transform(cls, input_size: int, is_train: bool = False):
        """Get transform for CLIP model"""
        transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                               std=[0.26862954, 0.26130258, 0.27577711])
        ])
        return transform


class CLIPModelWrapper(BaseModelWrapper):
    """Wrapper for CLIP vision model"""
    
    def get_embeddings(self, inputs) -> torch.Tensor:
        """Extract embeddings using CLIP vision model"""
        with torch.no_grad():
            if isinstance(inputs, dict) and 'pixel_values' in inputs:
                embeddings = self.model.get_image_features(inputs['pixel_values'])
            else:
                embeddings = self.model.get_image_features(**inputs)
            
            # CLIP already normalizes embeddings
            return embeddings


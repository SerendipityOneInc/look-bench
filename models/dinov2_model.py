"""
DINOv2 Model Implementation for LookBench
Fashion Image Retrieval Benchmark
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple
from transformers import AutoModel, AutoImageProcessor
from torchvision import transforms
import logging

from .base import BaseModel, BaseModelWrapper
from .registry import register_model
from utils.logging import get_logger, log_structured

logger = get_logger(__name__)


@register_model("dinov2", metadata={
    "description": "DINOv2 self-supervised vision transformer for fashion image retrieval",
    "framework": "PyTorch",
    "default_input_size": 224,
    "default_embedding_dim": 768,
    "architecture": "Vision Transformer"
})
class DINOv2EmbeddingModel(BaseModel):
    """DINOv2 model for fashion image embedding extraction"""

    @classmethod
    def load_model(cls, model_name: str, model_path: Optional[str] = None) -> Tuple[nn.Module, 'DINOv2EmbeddingModel']:
        """
        Load DINOv2 model

        Args:
            model_name: HuggingFace DINOv2 model name (e.g., 'facebook/dinov2-base')
            model_path: Optional path to checkpoint (not used for DINOv2)

        Returns:
            Tuple of (wrapped_model, model_instance)
        """
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        processor = AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()
        
        wrapped_model = DINOv2ModelWrapper(model, processor)
        
        log_structured(logger, logging.INFO, "DINOv2 model loaded successfully",
                     model_name=model_name, device=device)
        
        return wrapped_model, cls()

    @classmethod
    def get_transform(cls, input_size: int, is_train: bool = False):
        """Get transform for DINOv2 model"""
        transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        return transform


class DINOv2ModelWrapper(BaseModelWrapper):
    """Wrapper for DINOv2 model"""
    
    def get_embeddings(self, inputs) -> torch.Tensor:
        """Extract CLS token embeddings from DINOv2"""
        with torch.no_grad():
            outputs = self.model(**inputs) if isinstance(inputs, dict) else self.model(inputs)
            
            # Extract CLS token (first token in the sequence)
            if hasattr(outputs, 'last_hidden_state'):
                embeddings = outputs.last_hidden_state[:, 0, :]
            elif hasattr(outputs, 'pooler_output'):
                embeddings = outputs.pooler_output
            else:
                raise ValueError("Cannot extract embeddings from DINOv2 output")
            
            # Normalize embeddings
            embeddings = F.normalize(embeddings, p=2, dim=1)
            return embeddings


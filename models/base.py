"""
Base model classes for LookBench
Fashion Image Retrieval Benchmark
"""

from abc import ABC
from typing import Dict, Any, Optional, Tuple, Union
import torch
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F


class BaseModelWrapper(nn.Module):
    """Base wrapper class for models"""
    
    def __init__(self, model: nn.Module, processor: Optional[AutoImageProcessor] = None):
        super().__init__()
        self.model = model
        self.processor = processor

    def forward(self, image: Union[str, Image.Image, torch.Tensor]) -> torch.Tensor:
        """
        Extract normalized embeddings from the model

        Args:
            image: PIL Image, image path, or pre-processed tensor (single or batch)

        Returns:
            The embeddings of the image
        """
        if isinstance(image, str):
            # Image path - load and process
            image = Image.open(image).convert('RGB')
            if self.processor:
                inputs = self.processor(images=image, return_tensors="pt")
            else:
                raise ValueError("Processor required for image path input")
                
        elif isinstance(image, Image.Image):
            # PIL Image - process with processor
            if self.processor:
                inputs = self.processor(images=image, return_tensors="pt")
            else:
                raise ValueError("Processor required for PIL image input")
                
        elif isinstance(image, torch.Tensor):
            # Already processed tensor [B, C, H, W]
            if self.processor:
                batch_size = image.size(0)
                image_pils = []
                for i in range(batch_size):
                    image_pil = transforms.ToPILImage()(image[i])
                    image_pils.append(image_pil)
                inputs = self.processor(images=image_pils, return_tensors="pt")
            else:
                # Direct tensor input
                inputs = {'pixel_values': image}
        else:
            raise ValueError(f"Unsupported input type: {type(image)}")

        # Move inputs to model device
        device = next(self.model.parameters()).device
        if isinstance(inputs, dict):
            inputs = {k: v.to(device) for k, v in inputs.items()}
        else:
            inputs = inputs.to(device)

        return self.get_embeddings(inputs)

    def get_embeddings(self, inputs) -> torch.Tensor:
        """
        Extract and return normalized embeddings from the model

        Args:
            inputs: Inputs to the model

        Returns:
            Normalized embeddings
        """
        with torch.no_grad():
            if hasattr(self.model, 'get_image_features'):
                # CLIP/SigLIP style models
                if isinstance(inputs, dict) and 'pixel_values' in inputs:
                    embeddings = self.model.get_image_features(inputs['pixel_values'])
                else:
                    embeddings = self.model.get_image_features(**inputs)
            else:
                # Generic model
                outputs = self.model(**inputs) if isinstance(inputs, dict) else self.model(inputs)
                
                if hasattr(outputs, 'last_hidden_state'):
                    # Use CLS token
                    embeddings = outputs.last_hidden_state[:, 0, :]
                elif hasattr(outputs, 'pooler_output'):
                    embeddings = outputs.pooler_output
                elif isinstance(outputs, torch.Tensor):
                    embeddings = outputs
                else:
                    raise ValueError(f"Cannot extract embeddings from model output type: {type(outputs)}")

            # Normalize embeddings
            embeddings = F.normalize(embeddings, p=2, dim=1)
            return embeddings

    def __call__(self, image: Union[str, Image.Image, torch.Tensor]) -> torch.Tensor:
        """Make the wrapper callable"""
        return self.forward(image)


class BaseModel(ABC):
    """Abstract base class for image retrieval models"""

    @classmethod
    def load_model(cls, model_name: str, model_path: Optional[str] = None) -> Tuple[Union[nn.Module, Any], 'BaseModel']:
        """
        Load model and return model wrapper and model instance

        Args:
            model_name: Model name or path (e.g., HuggingFace repo path)
            model_path: Optional path to model checkpoint

        Returns:
            Tuple of (wrapped_model, model_instance)
        """
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        model.to("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()
        processor = AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True)
        wrapped_model = BaseModelWrapper(model, processor)
        return wrapped_model, cls()

    def get_model_info(self, model: Union[nn.Module, Any]) -> Dict[str, Any]:
        """
        Get model information

        Args:
            model: The loaded model instance

        Returns:
            Dictionary containing model information
        """
        if hasattr(model, 'model'):  # If it's our wrapper
            base_model = model.model
        else:
            base_model = model

        total_params = sum(p.numel() for p in base_model.parameters())

        info = {
            "total_params": total_params,
            "device": str(next(base_model.parameters()).device),
            "model_type": self.get_model_type(),
            "framework": "PyTorch"
        }

        return info

    @classmethod
    def get_transform(cls, input_size: int, is_train: bool = False):
        """
        Get transform for batched images

        Args:
            input_size: Input image size
            is_train: Whether this is for training

        Returns:
            Transform pipeline
        """
        transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor()
        ])
        return transform

    @classmethod
    def get_model_type(cls) -> str:
        """Get the model type identifier"""
        return cls.__name__.lower().replace('model', '').replace('embedding', '')

    @classmethod
    def get_model_name(cls) -> str:
        """Get the human-readable model name"""
        return cls.__name__


#!/usr/bin/env python3
"""
LookBench Custom Model Integration Example
Demonstrates how to integrate custom models into LookBench
"""

import torch
import torch.nn as nn
from torchvision import transforms, models
from datasets import load_dataset
import numpy as np
import sys
sys.path.append('..')

from models.base import BaseModel
from models.registry import register_model, list_available_models


# Example 1: Simple ResNet-based Model
@register_model("resnet50", metadata={
    "description": "ResNet-50 pretrained on ImageNet",
    "framework": "PyTorch",
    "input_size": 224,
    "embedding_dim": 2048
})
class ResNet50Model(BaseModel):
    """ResNet-50 model for image embedding extraction"""
    
    @classmethod
    def load_model(cls, model_name: str = "resnet50", model_path: str = None):
        """Load ResNet-50 model"""
        # Load pretrained ResNet-50
        model = models.resnet50(pretrained=True)
        
        # Remove classification head, keep feature extractor
        model = nn.Sequential(*list(model.children())[:-1])
        
        # Load custom weights if provided
        if model_path:
            state_dict = torch.load(model_path, map_location='cpu')
            model.load_state_dict(state_dict)
        
        model.eval()
        
        # Create a wrapper that flattens the output
        class ModelWrapper(nn.Module):
            def __init__(self, backbone):
                super().__init__()
                self.backbone = backbone
            
            def forward(self, x):
                features = self.backbone(x)
                return features.squeeze(-1).squeeze(-1)  # Flatten spatial dimensions
        
        wrapped_model = ModelWrapper(model)
        return wrapped_model, cls()
    
    @classmethod
    def get_transform(cls, input_size: int = 224):
        """Get image preprocessing transform"""
        return transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])


# Example 2: Custom Architecture
@register_model("custom_fashion_model", metadata={
    "description": "Custom fashion embedding model",
    "framework": "PyTorch",
    "input_size": 256,
    "embedding_dim": 512
})
class CustomFashionModel(BaseModel):
    """Custom fashion model trained on fashion data"""
    
    @classmethod
    def load_model(cls, model_name: str = "custom_fashion_model", model_path: str = None):
        """Load custom model"""
        if not model_path:
            raise ValueError("model_path is required for custom model")
        
        # Define your model architecture
        class YourCustomArchitecture(nn.Module):
            def __init__(self, embedding_dim=512):
                super().__init__()
                # Your model architecture here
                self.backbone = models.resnet34(pretrained=False)
                self.backbone.fc = nn.Linear(self.backbone.fc.in_features, embedding_dim)
            
            def forward(self, x):
                return self.backbone(x)
        
        # Instantiate and load weights
        model = YourCustomArchitecture(embedding_dim=512)
        
        # Load trained weights
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        return model, cls()
    
    @classmethod
    def get_transform(cls, input_size: int = 256):
        """Custom preprocessing pipeline"""
        return transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])


def test_custom_model():
    """Test the custom model"""
    
    print("="*60)
    print("Testing Custom Model Integration")
    print("="*60)
    
    # List all available models
    print("\nAvailable models:", list_available_models())
    
    # Load dataset
    print("\nLoading dataset...")
    dataset = load_dataset("srpone/look-bench")
    sample_img = dataset['real_studio_flat']['query'][0]['image']
    
    # Test ResNet-50
    print("\n[1/2] Testing ResNet-50 model...")
    model, wrapper = ResNet50Model.load_model()
    transform = ResNet50Model.get_transform()
    
    img_tensor = transform(sample_img).unsqueeze(0)
    if torch.cuda.is_available():
        model = model.cuda()
        img_tensor = img_tensor.cuda()
    
    with torch.no_grad():
        features = model(img_tensor)
    
    print(f"  Feature shape: {features.shape}")
    print(f"  Feature dimension: {features.shape[1]}")
    print(f"  Feature norm: {torch.norm(features).item():.4f}")
    print("  ✓ ResNet-50 model works correctly!")
    
    # Test similarity computation
    print("\n[2/2] Testing similarity computation...")
    gallery_imgs = [dataset['real_studio_flat']['gallery'][i]['image'] for i in range(3)]
    
    def extract_features(images, model, transform):
        features = []
        for img in images:
            img_tensor = transform(img).unsqueeze(0)
            if torch.cuda.is_available():
                img_tensor = img_tensor.cuda()
            with torch.no_grad():
                feat = model(img_tensor)
            features.append(feat.cpu().numpy())
        return np.vstack(features)
    
    query_feat = extract_features([sample_img], model, transform)
    gallery_feats = extract_features(gallery_imgs, model, transform)
    
    # Compute cosine similarity
    query_norm = query_feat / np.linalg.norm(query_feat, axis=1, keepdims=True)
    gallery_norm = gallery_feats / np.linalg.norm(gallery_feats, axis=1, keepdims=True)
    similarities = np.dot(query_norm, gallery_norm.T)[0]
    
    print("  Similarity scores:")
    for i, sim in enumerate(similarities):
        print(f"    Gallery image {i}: {sim:.4f}")
    print("  ✓ Similarity computation works correctly!")
    
    print("\n" + "="*60)
    print("Custom Model Integration Test Passed!")
    print("="*60)
    
    print("\nNext steps:")
    print("  1. Add your model configuration to configs/config.yaml:")
    print("     ```yaml")
    print("     resnet50:")
    print("       enabled: true")
    print("       model_name: \"resnet50\"")
    print("       model_path: null")
    print("       input_size: 224")
    print("       embedding_dim: 2048")
    print("       device: \"cuda\"")
    print("     ```")
    print("\n  2. Use it in evaluation:")
    print("     python main.py --model resnet50 --dataset fashion200k")
    print("\n  3. Compare with baselines using 02_model_evaluation.py")


def main():
    """Main function"""
    test_custom_model()


if __name__ == "__main__":
    main()

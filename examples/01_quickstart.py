#!/usr/bin/env python3
"""
LookBench Quickstart Example
Demonstrates basic usage of LookBench for fashion image retrieval
"""

import torch
import numpy as np
from datasets import load_dataset
import matplotlib.pyplot as plt
import sys
sys.path.append('..')

from manager import ConfigManager, ModelManager, DataManager


def main():
    print("="*60)
    print("LookBench Quickstart Example")
    print("="*60)
    
    # 1. Load LookBench Dataset from Hugging Face
    print("\n[1/5] Loading LookBench dataset from Hugging Face...")
    dataset = load_dataset("srpone/look-bench")
    
    print(f"Available subsets: {list(dataset.keys())}")
    print(f"\nDataset structure:")
    for subset_name in dataset.keys():
        print(f"  {subset_name}:")
        for split_name in dataset[subset_name].keys():
            print(f"    {split_name}: {len(dataset[subset_name][split_name])} samples")
    
    # 2. Explore a sample
    print("\n[2/5] Exploring a sample...")
    sample = dataset['real_studio_flat']['query'][0]
    print(f"Sample keys: {list(sample.keys())}")
    print(f"Category: {sample['category']}")
    print(f"Main attribute: {sample['main_attribute']}")
    print(f"Other attributes: {sample['other_attributes']}")
    print(f"Task: {sample['task']}")
    print(f"Difficulty: {sample['difficulty']}")
    
    # 3. Load Model
    print("\n[3/5] Loading CLIP model...")
    config_manager = ConfigManager('../configs/config.yaml')
    model_manager = ModelManager(config_manager)
    
    model, model_wrapper = model_manager.load_model('clip')
    transform = model_manager.get_transform('clip')
    
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
        print("Model moved to CUDA")
    
    # 4. Extract features from a sample
    print("\n[4/5] Extracting features from sample image...")
    sample_image = dataset['real_studio_flat']['query'][0]['image']
    image_tensor = transform(sample_image).unsqueeze(0)
    
    if torch.cuda.is_available():
        image_tensor = image_tensor.cuda()
    
    with torch.no_grad():
        features = model(image_tensor)
    
    print(f"Feature shape: {features.shape}")
    print(f"Feature norm: {torch.norm(features).item():.4f}")
    
    # 5. Compute similarity with gallery images
    print("\n[5/5] Computing similarity with gallery images...")
    
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
    
    # Get query and gallery images
    query_img = dataset['real_studio_flat']['query'][0]['image']
    gallery_imgs = [dataset['real_studio_flat']['gallery'][i]['image'] for i in range(5)]
    
    # Extract features
    query_feat = extract_features([query_img], model, transform)
    gallery_feats = extract_features(gallery_imgs, model, transform)
    
    # Compute cosine similarity
    query_norm = query_feat / np.linalg.norm(query_feat, axis=1, keepdims=True)
    gallery_norm = gallery_feats / np.linalg.norm(gallery_feats, axis=1, keepdims=True)
    similarities = np.dot(query_norm, gallery_norm.T)[0]
    
    print("\nSimilarity scores:")
    for i, sim in enumerate(similarities):
        print(f"  Gallery image {i}: {sim:.4f}")
    
    print("\n" + "="*60)
    print("Quickstart completed successfully!")
    print("="*60)
    print("\nNext steps:")
    print("  - Run 02_model_evaluation.py for full evaluation")
    print("  - Run 03_custom_model.py to integrate your own model")
    print("  - Check out the paper: https://arxiv.org/abs/2601.14706")


if __name__ == "__main__":
    main()

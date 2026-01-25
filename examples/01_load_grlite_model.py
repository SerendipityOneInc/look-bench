#!/usr/bin/env python3
"""
LookBench GR-Lite Model Loading Example
Demonstrates how to load and use the GR-Lite model for fashion image retrieval
"""

import torch
import numpy as np
import sys
sys.path.append('..')

# Use look-bench's dataset loader to avoid import conflicts
from utils.hf_loader import load_lookbench_dataset


def main():
    print("="*60)
    print("LookBench GR-Lite Model Loading Example")
    print("="*60)
    
    # 1. Load LookBench Dataset from Hugging Face
    print("\n[1/5] Loading LookBench dataset from Hugging Face...")
    # Load real_studio_flat config using look-bench's loader
    dataset = load_lookbench_dataset("real_studio_flat")
    
    print(f"Dataset splits: {list(dataset.keys())}")
    print(f"\nDataset structure:")
    for split_name in dataset.keys():
        print(f"  {split_name}: {len(dataset[split_name])} samples")
    
    # 2. Explore a sample
    print("\n[2/5] Exploring a sample...")
    sample = dataset['query'][0]
    print(f"Sample keys: {list(sample.keys())}")
    print(f"Category: {sample['category']}")
    print(f"Main attribute: {sample['main_attribute']}")
    print(f"Other attributes: {sample['other_attributes']}")
    print(f"Task: {sample['task']}")
    print(f"Difficulty: {sample['difficulty']}")
    
    # 3. Load Model
    print("\n[3/5] Loading GR-Lite model from Hugging Face...")
    from huggingface_hub import hf_hub_download
    
    # Load GR-Lite model (fashion image search model)
    model_name = "srpone/gr-lite"
    
    try:
        # Download the model checkpoint from Hugging Face
        print(f"  Downloading model from {model_name}...")
        model_path = hf_hub_download(
            repo_id=model_name,
            filename="gr_lite.pt"
        )
        
        # Load the PyTorch model
        print(f"  Loading model checkpoint...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = torch.load(model_path, map_location=device)
        
        # Set to evaluation mode
        if hasattr(model, 'eval'):
            model.eval()
        
        print(f"  Model loaded on {device}")
        print(f"✅ GR-Lite model loaded from {model_name}")
        print(f"   Model type: {type(model)}")
        
        # Check if model has expected methods
        if hasattr(model, 'search'):
            print(f"   Model has .search() method for feature extraction")
        elif hasattr(model, 'forward') or hasattr(model, '__call__'):
            print(f"   Model has forward/call method")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print(f"\nTroubleshooting:")
        print(f"  1. Check if the model exists: https://huggingface.co/{model_name}")
        print(f"  2. Ensure huggingface_hub is installed: pip install huggingface_hub")
        print(f"  3. Model file: gr_lite.pt")
        raise
    
    # 4. Inspect model structure
    print("\n[4/5] Inspecting model structure...")
    print(f"Model type: {type(model)}")
    
    # Check available methods
    if hasattr(model, '__dict__'):
        print(f"Model attributes: {list(model.__dict__.keys())[:10]}")
    
    # Try to understand the model interface
    if hasattr(model, 'search'):
        print("✓ Model has .search() method")
        print("  Usage: model.search(image_paths=images, feature_dim=256)")
    elif hasattr(model, 'encode'):
        print("✓ Model has .encode() method")
    elif hasattr(model, 'forward'):
        print("✓ Model has .forward() method")
    else:
        print("⚠ Model interface unclear - may need custom preprocessing")
    
    # 5. Example usage (if model has search method)
    print("\n[5/5] Example usage...")
    if hasattr(model, 'search'):
        print("Extracting features from sample image...")
        sample_image = dataset['query'][0]['image']
        
        # Use model's search method
        with torch.no_grad():
            _, embeddings = model.search(image_paths=[sample_image], feature_dim=256)
        
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()
        
        print(f"✓ Feature shape: {embeddings.shape}")
        print(f"✓ Feature norm: {np.linalg.norm(embeddings[0]):.4f}")
    else:
        print("⚠ Skipping feature extraction - model interface needs clarification")
        print("  Please refer to model documentation for usage")
    
    print("\n" + "="*60)
    print("GR-Lite model loading and testing completed successfully!")
    print("="*60)
    print("\nModel used: GR-Lite (srpone/gr-lite)")
    print("  - Hugging Face: https://huggingface.co/srpone/gr-lite")
    print("\nNext steps:")
    print("  - Run 02_model_evaluation.py for full evaluation")
    print("  - Run 03_custom_model.py to integrate your own model")
    print("  - Check out the paper: https://arxiv.org/abs/2601.14706")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
LookBench Model Evaluation Example
Demonstrates how to evaluate models on the LookBench dataset
"""

import torch
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
import sys
sys.path.append('..')

from manager import ConfigManager, ModelManager
from metrics import RankEvaluator, MRREvaluator, NDCGEvaluator, MAPEvaluator


def extract_features_from_dataset(data, model, transform, batch_size=32):
    """Extract features from a dataset"""
    features = []
    labels = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(data), batch_size), desc="Extracting features"):
            batch_data = data[i:i+batch_size]
            
            # Prepare batch images
            batch_images = []
            batch_labels = []
            
            for sample in batch_data:
                img = sample['image']
                img_tensor = transform(img)
                batch_images.append(img_tensor)
                
                # Use item_ID as label if available
                label = sample.get('item_ID', i)
                batch_labels.append(label)
            
            # Stack batch
            batch_tensor = torch.stack(batch_images)
            if torch.cuda.is_available():
                batch_tensor = batch_tensor.cuda()
            
            # Extract features
            batch_features = model(batch_tensor)
            features.append(batch_features.cpu())
            labels.extend(batch_labels)
    
    features = torch.cat(features, dim=0)
    return features.numpy(), np.array(labels)


def evaluate_model(model_name='clip', subset_name='real_studio_flat'):
    """Evaluate a model on a specific LookBench subset"""
    
    print("="*60)
    print(f"Evaluating {model_name} on {subset_name}")
    print("="*60)
    
    # Load dataset
    print("\n[1/5] Loading dataset...")
    dataset = load_dataset("srpone/look-bench")
    query_data = dataset[subset_name]['query']
    gallery_data = dataset[subset_name]['gallery']
    
    print(f"Query samples: {len(query_data)}")
    print(f"Gallery samples: {len(gallery_data)}")
    
    # Load model
    print(f"\n[2/5] Loading {model_name} model...")
    config_manager = ConfigManager('../configs/config.yaml')
    model_manager = ModelManager(config_manager)
    
    model, _ = model_manager.load_model(model_name)
    transform = model_manager.get_transform(model_name)
    
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
        print("Model moved to CUDA")
    
    # Extract features
    print("\n[3/5] Extracting features...")
    query_features, query_labels = extract_features_from_dataset(
        query_data, model, transform, batch_size=32
    )
    gallery_features, gallery_labels = extract_features_from_dataset(
        gallery_data, model, transform, batch_size=32
    )
    
    # L2 normalize features
    query_features = query_features / np.linalg.norm(query_features, axis=1, keepdims=True)
    gallery_features = gallery_features / np.linalg.norm(gallery_features, axis=1, keepdims=True)
    print("Features normalized")
    
    # Compute similarity matrix
    print("\n[4/5] Computing similarity matrix...")
    similarity_matrix = np.dot(query_features, gallery_features.T)
    sorted_indices = np.argsort(-similarity_matrix, axis=1)
    
    # Evaluate metrics
    print("\n[5/5] Computing metrics...")
    rank_evaluator = RankEvaluator(top_k=[1, 5, 10, 20])
    mrr_evaluator = MRREvaluator()
    ndcg_evaluator = NDCGEvaluator(k=5)
    map_evaluator = MAPEvaluator()
    
    results = {}
    
    # Recall@K
    for k in [1, 5, 10, 20]:
        scores = []
        for i in range(len(query_labels)):
            score = rank_evaluator.metric_eval(
                sorted_indices[i],
                k,
                query_labels[i],
                gallery_labels
            )
            scores.append(score)
        results[f'Recall@{k}'] = np.mean(scores) * 100
    
    # MRR
    mrr_scores = []
    for i in range(len(query_labels)):
        score = mrr_evaluator.metric_eval(
            sorted_indices[i],
            None,
            query_labels[i],
            gallery_labels
        )
        mrr_scores.append(score)
    results['MRR'] = np.mean(mrr_scores) * 100
    
    # NDCG@5
    ndcg_scores = []
    for i in range(len(query_labels)):
        score = ndcg_evaluator.metric_eval(
            sorted_indices[i],
            5,
            query_labels[i],
            gallery_labels
        )
        ndcg_scores.append(score)
    results['NDCG@5'] = np.mean(ndcg_scores) * 100
    
    # mAP
    map_scores = []
    for i in range(len(query_labels)):
        score = map_evaluator.metric_eval(
            sorted_indices[i],
            None,
            query_labels[i],
            gallery_labels
        )
        map_scores.append(score)
    results['mAP'] = np.mean(map_scores) * 100
    
    # Print results
    print(f"\n{'='*60}")
    print(f"Evaluation Results on {subset_name}")
    print(f"Model: {model_name}")
    print(f"{'='*60}")
    for metric, value in results.items():
        print(f"{metric:15s}: {value:6.2f}%")
    print(f"{'='*60}\n")
    
    return results


def main():
    """Main evaluation function"""
    
    # Evaluate CLIP on different subsets
    subsets = ['real_studio_flat', 'aigen_studio', 'real_streetlook', 'aigen_streetlook']
    
    print("LookBench Model Evaluation")
    print("This will evaluate CLIP on all LookBench subsets\n")
    
    all_results = {}
    
    for subset in subsets:
        try:
            results = evaluate_model(model_name='clip', subset_name=subset)
            all_results[subset] = results
        except Exception as e:
            print(f"Error evaluating on {subset}: {e}")
            continue
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"{'Subset':<20} {'Recall@1':>10} {'Recall@5':>10} {'Recall@10':>10} {'MRR':>10}")
    print("-"*60)
    for subset, results in all_results.items():
        print(f"{subset:<20} {results['Recall@1']:>9.2f}% {results['Recall@5']:>9.2f}% "
              f"{results['Recall@10']:>9.2f}% {results['MRR']:>9.2f}%")
    print("="*60)


if __name__ == "__main__":
    main()

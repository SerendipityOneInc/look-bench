"""
Multi-modal evaluator for LookBench.

Supports text-to-image retrieval with explicit ground truth mappings
(ZooClaw-Fashion format).

Follows the same encoding approach as fine-tune-vit (InferenceEngine):
- Images: processor(images=..., return_tensors="pt") → model.get_image_features(**inputs)
- Text:   processor(text=..., return_tensors="pt", padding=True, truncation=True) → model.get_text_features(**inputs)
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


def extract_image_features(
    model, dataset, processor, batch_size: int = 128, num_workers: int = 4,
) -> tuple:
    """Extract image features using processor + model.get_image_features.

    Args:
        model: The VLE model (e.g. SiglipModel, CLIPModel)
        dataset: Dataset yielding (PIL.Image, item_id) pairs
        processor: The model's processor (AutoProcessor or compatible)
        batch_size: Batch size
        num_workers: DataLoader workers

    Returns:
        (features [N, D], ids [N])
    """
    from PIL import Image

    def _collate(batch):
        images, ids = zip(*batch)
        return list(images), list(ids)

    loader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers,
        shuffle=False, drop_last=False, collate_fn=_collate,
    )
    all_features = []
    all_ids = []

    model.eval()
    device = next(model.parameters()).device

    with torch.no_grad():
        for images, ids in tqdm(loader, desc="Encoding images"):
            inputs = processor(images=images, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            features = model.get_image_features(**inputs)
            if not isinstance(features, torch.Tensor):
                features = features.pooler_output
            features = F.normalize(features.float(), p=2, dim=1)
            all_features.append(features.cpu())
            all_ids.extend(ids if isinstance(ids, list) else ids.tolist())

    return torch.cat(all_features, dim=0), all_ids


def extract_text_features(
    model, dataset, processor, batch_size: int = 128, max_length: int = 64,
) -> tuple:
    """Extract text features using processor + model.get_text_features.

    Args:
        model: The VLE model
        dataset: Dataset yielding (text_string, item_id) pairs
        processor: The model's processor
        batch_size: Batch size
        max_length: Max token length

    Returns:
        (features [N, D], ids [N])
    """
    def _collate(batch):
        texts, ids = zip(*batch)
        return list(texts), list(ids)

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        drop_last=False, collate_fn=_collate,
    )
    all_features = []
    all_ids = []

    model.eval()
    device = next(model.parameters()).device

    with torch.no_grad():
        for texts, ids in tqdm(loader, desc="Encoding text"):
            inputs = processor(
                text=texts, return_tensors="pt",
                padding="max_length", truncation=True, max_length=max_length,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            features = model.get_text_features(**inputs)
            if not isinstance(features, torch.Tensor):
                features = features.pooler_output
            features = F.normalize(features.float(), p=2, dim=1)
            all_features.append(features.cpu())
            all_ids.extend(ids)

    return torch.cat(all_features, dim=0), all_ids


def evaluate_retrieval(
    query_features: torch.Tensor,
    query_ids: List[int],
    corpus_features: torch.Tensor,
    corpus_ids: List[int],
    ground_truth: Dict[str, List[int]],
    top_k: List[int] = None,
) -> Dict[str, float]:
    """Evaluate retrieval with explicit ground truth mapping.

    Args:
        query_features: [N_query, D]
        query_ids: List of query IDs
        corpus_features: [N_corpus, D]
        corpus_ids: List of corpus IDs
        ground_truth: {query_id_str: [corpus_id, ...]}
        top_k: List of K values for Recall@K

    Returns:
        Dict with recall@K and MRR values
    """
    if top_k is None:
        top_k = [1, 5, 10, 20]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    query_features = F.normalize(query_features.to(device), p=2, dim=1)
    corpus_features = F.normalize(corpus_features.to(device), p=2, dim=1)

    # Compute similarity matrix
    logger.info("Computing similarity matrix...")
    similarity = torch.mm(query_features, corpus_features.t())  # [N_q, N_c]

    # Evaluate
    recall_scores = {k: [] for k in top_k}
    mrr_scores = []

    max_k = max(top_k)

    for i, qid in enumerate(tqdm(query_ids, desc="Evaluating")):
        gt_cids = set(ground_truth.get(str(qid), []))
        if not gt_cids:
            continue

        # Get top-K corpus indices
        _, topk_indices = torch.topk(similarity[i], min(max_k, len(corpus_ids)))
        topk_indices = topk_indices.cpu().tolist()

        # Map indices back to corpus IDs
        topk_cids = [corpus_ids[idx] for idx in topk_indices]

        # Recall@K
        for k in top_k:
            hit = any(cid in gt_cids for cid in topk_cids[:k])
            recall_scores[k].append(1.0 if hit else 0.0)

        # MRR
        rr = 0.0
        for rank, cid in enumerate(topk_cids, start=1):
            if cid in gt_cids:
                rr = 1.0 / rank
                break
        mrr_scores.append(rr)

    results = {}
    for k in top_k:
        if recall_scores[k]:
            results[f"recall@{k}"] = round(np.mean(recall_scores[k]), 4)
    if mrr_scores:
        results["mrr"] = round(np.mean(mrr_scores), 4)

    for key, val in results.items():
        logger.info(f"  {key}: {val:.4f}")

    return results


def evaluate_zooclaw_task(
    model,
    task_data: Dict,
    processor,
    batch_size: int = 128,
    num_workers: int = 4,
    top_k: List[int] = None,
    max_length: int = 64,
) -> Dict[str, float]:
    """Run evaluation for a single ZooClaw task.

    Args:
        model: The VLE model (SiglipModel, CLIPModel, etc.)
        task_data: Output from load_zooclaw_task()
        processor: Model processor (AutoProcessor or compatible)
        batch_size: Batch size for feature extraction
        num_workers: DataLoader workers
        top_k: K values for Recall@K
        max_length: Max token length for text encoding

    Returns:
        Dict with recall@K and MRR values
    """
    query_modality = task_data["query_modality"]
    corpus_modality = task_data["corpus_modality"]

    # Extract query features
    if query_modality == "image":
        q_features, q_ids = extract_image_features(
            model, task_data["query_dataset"], processor,
            batch_size=batch_size, num_workers=num_workers,
        )
    else:
        q_features, q_ids = extract_text_features(
            model, task_data["query_dataset"], processor,
            batch_size=batch_size, max_length=max_length,
        )

    # Extract corpus features
    if corpus_modality == "image":
        c_features, c_ids = extract_image_features(
            model, task_data["corpus_dataset"], processor,
            batch_size=batch_size, num_workers=num_workers,
        )
    else:
        c_features, c_ids = extract_text_features(
            model, task_data["corpus_dataset"], processor,
            batch_size=batch_size, max_length=max_length,
        )

    return evaluate_retrieval(
        q_features, q_ids, c_features, c_ids,
        task_data["ground_truth"], top_k=top_k,
    )

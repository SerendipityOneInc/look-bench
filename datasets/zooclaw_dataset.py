"""
ZooClaw-Fashion dataset loader for LookBench.

Loads BEIR-style JSON datasets (queries.json, corpus.json, ground_truth.json)
with support for multi-modal retrieval tasks (text2image, image2text, text2text).
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset

from utils.logging import get_logger

logger = get_logger(__name__)


class ZooClawImageDataset(Dataset):
    """Dataset that yields (PIL.Image, item_id) pairs."""

    def __init__(self, items: List[dict], image_key: str, id_key: str,
                 transform=None, data_root: str = ""):
        self.items = items
        self.image_key = image_key
        self.id_key = id_key
        self.transform = transform
        self.data_root = data_root

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, int]:
        item = self.items[idx]
        image_path = item[self.image_key]
        if not os.path.isabs(image_path):
            image_path = os.path.join(self.data_root, image_path)
        img = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, item[self.id_key]


class ZooClawTextDataset(Dataset):
    """Dataset that yields (text_string, item_id) pairs."""

    def __init__(self, items: List[dict], text_key: str, id_key: str):
        self.items = items
        self.text_key = text_key
        self.id_key = id_key

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx) -> Tuple[str, int]:
        item = self.items[idx]
        return item[self.text_key], item[self.id_key]


def load_zooclaw_task(
    task_dir: str,
    task_type: str,
    transform=None,
    data_root: str = "",
    use_long_queries: bool = False,
) -> Dict:
    """
    Load a single ZooClaw-Fashion retrieval task.

    Args:
        task_dir: Path to task directory (e.g., "data/zooclaw-fashion/text2image")
        task_type: One of "text2image", "image2text", "text2text"
        transform: Image transform (required for tasks involving images)
        data_root: Root directory for resolving relative image paths
        use_long_queries: If True and available, use queries_long.json for text queries

    Returns:
        Dict with keys: "query_dataset", "corpus_dataset", "ground_truth",
                        "query_modality", "corpus_modality"
    """
    task_dir = Path(task_dir)

    # Load ground truth
    gt_path = task_dir / "ground_truth.json"
    with open(gt_path) as f:
        ground_truth = json.load(f)
    # Ensure keys are strings mapping to list of ints
    ground_truth = {str(k): [int(x) for x in v] for k, v in ground_truth.items()}

    if task_type == "text2image":
        # Query = text, Corpus = images
        if use_long_queries and (task_dir / "queries_long.json").exists():
            queries_file = "queries_long.json"
            text_key = "long_query_llm"
        else:
            queries_file = "queries.json"
            text_key = "short_query"

        with open(task_dir / queries_file) as f:
            queries = json.load(f)
        with open(task_dir / "corpus.json") as f:
            corpus = json.load(f)

        query_dataset = ZooClawTextDataset(queries, text_key=text_key, id_key="query_id")
        corpus_dataset = ZooClawImageDataset(
            corpus, image_key="image_path", id_key="corpus_id",
            transform=transform, data_root=data_root,
        )
        return {
            "query_dataset": query_dataset,
            "corpus_dataset": corpus_dataset,
            "ground_truth": ground_truth,
            "query_modality": "text",
            "corpus_modality": "image",
        }

    elif task_type == "image2text":
        # Query = images, Corpus = text
        with open(task_dir / "queries.json") as f:
            queries = json.load(f)
        with open(task_dir / "corpus.json") as f:
            corpus = json.load(f)

        query_dataset = ZooClawImageDataset(
            queries, image_key="image_path", id_key="query_id",
            transform=transform, data_root=data_root,
        )
        corpus_dataset = ZooClawTextDataset(corpus, text_key="short_text", id_key="corpus_id")
        return {
            "query_dataset": query_dataset,
            "corpus_dataset": corpus_dataset,
            "ground_truth": ground_truth,
            "query_modality": "image",
            "corpus_modality": "text",
        }

    elif task_type == "text2text":
        # Query = text, Corpus = text
        with open(task_dir / "queries.json") as f:
            queries = json.load(f)
        with open(task_dir / "corpus.json") as f:
            corpus = json.load(f)

        query_dataset = ZooClawTextDataset(queries, text_key="short_query", id_key="query_id")
        corpus_dataset = ZooClawTextDataset(corpus, text_key="short_text", id_key="corpus_id")
        return {
            "query_dataset": query_dataset,
            "corpus_dataset": corpus_dataset,
            "ground_truth": ground_truth,
            "query_modality": "text",
            "corpus_modality": "text",
        }

    else:
        raise ValueError(f"Unknown task type: {task_type}. Expected: text2image, image2text, text2text")


def load_zooclaw_dataset(
    dataset_dir: str,
    tasks: Optional[List[str]] = None,
    transform=None,
    data_root: Optional[str] = None,
    use_long_queries: bool = False,
) -> Dict[str, Dict]:
    """
    Load all tasks from a ZooClaw-Fashion dataset directory.

    Args:
        dataset_dir: Root dataset directory containing task subdirectories
        tasks: List of tasks to load (default: all available)
        transform: Image transform
        data_root: Root for image paths (defaults to dataset_dir)
        use_long_queries: Use long queries where available

    Returns:
        Dict mapping task_name -> task_data (from load_zooclaw_task)
    """
    dataset_dir = Path(dataset_dir)
    if data_root is None:
        data_root = str(dataset_dir)

    all_tasks = ["text2image", "image2text", "text2text"]
    if tasks is None:
        tasks = [t for t in all_tasks if (dataset_dir / t).exists()]

    result = {}
    for task in tasks:
        task_dir = dataset_dir / task
        if not task_dir.exists():
            logger.warning(f"Task directory not found: {task_dir}, skipping")
            continue
        logger.info(f"Loading task: {task}")
        result[task] = load_zooclaw_task(
            str(task_dir), task, transform=transform,
            data_root=data_root, use_long_queries=use_long_queries,
        )

    return result

#!/usr/bin/env python3
"""
ZooClaw-Fashion Evaluation Example for LookBench.

Evaluates a model on the ZooClaw-Fashion benchmark (text2image retrieval).

Usage:
    # From look-bench root directory
    python examples/04_zooclaw_eval.py \
        --dataset-dir /path/to/zooclaw-fashion \
        --model-name google/siglip2-base-patch16-384 \
        --batch-size 128

    # With long queries
    python examples/04_zooclaw_eval.py \
        --dataset-dir /path/to/zooclaw-fashion \
        --model-name google/siglip2-base-patch16-384 \
        --long-queries

    # With HuggingFace dataset
    python examples/04_zooclaw_eval.py \
        --hf-dataset srpone/zooclaw-fashion \
        --model-name google/siglip2-base-patch16-384
"""

import argparse
import importlib.util
import json
import sys
import os

_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _root)

import torch
from transformers import AutoModel


def _load_module(name, filepath):
    """Load a module directly from file to avoid package __init__ circular imports."""
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_zooclaw_ds = _load_module("zooclaw_dataset", os.path.join(_root, "datasets", "zooclaw_dataset.py"))
load_zooclaw_dataset = _zooclaw_ds.load_zooclaw_dataset

_mm_eval = _load_module("multimodal_evaluator", os.path.join(_root, "runner", "multimodal_evaluator.py"))
evaluate_zooclaw_task = _mm_eval.evaluate_zooclaw_task


def load_processor(model_name: str):
    """Load model processor, with fallback for SigLIP2 tokenizer compatibility."""
    from transformers import AutoProcessor
    try:
        return AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    except Exception:
        # Older transformers may fail on SigLIP2 (GemmaTokenizer).
        # Build a composite processor from image_processor + tokenizer.
        from transformers import AutoImageProcessor, AutoTokenizer
        image_processor = AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True)
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        except Exception:
            from transformers import GemmaTokenizer
            tokenizer = GemmaTokenizer.from_pretrained(model_name, trust_remote_code=True)

        class _CompositeProcessor:
            def __init__(self, img_proc, tok):
                self.image_processor = img_proc
                self.tokenizer = tok

            def __call__(self, text=None, images=None, **kwargs):
                result = {}
                if images is not None:
                    result.update(self.image_processor(images=images, return_tensors=kwargs.get("return_tensors")))
                if text is not None:
                    result.update(self.tokenizer(
                        text,
                        return_tensors=kwargs.get("return_tensors"),
                        padding=kwargs.get("padding", False),
                        truncation=kwargs.get("truncation", False),
                        max_length=kwargs.get("max_length"),
                    ))
                return result

        return _CompositeProcessor(image_processor, tokenizer)


def main():
    parser = argparse.ArgumentParser(description="ZooClaw-Fashion evaluation")
    parser.add_argument("--dataset-dir", type=str, default=None,
                        help="Path to ZooClaw-Fashion dataset directory")
    parser.add_argument("--hf-dataset", type=str, default=None,
                        help="HuggingFace dataset ID (e.g., srpone/zooclaw-fashion)")
    parser.add_argument("--model-name", type=str,
                        default="google/siglip2-base-patch16-384")
    parser.add_argument("--tasks", nargs="+",
                        default=["text2image"],
                        choices=["text2image", "image2text", "text2text"])
    parser.add_argument("--long-queries", action="store_true",
                        help="Use long-form queries for text2image")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--top-k", nargs="+", type=int, default=[1, 5, 10, 20])
    parser.add_argument("--output", type=str, default=None,
                        help="Save results to JSON file")
    args = parser.parse_args()

    # Resolve dataset directory
    if args.hf_dataset:
        from huggingface_hub import snapshot_download
        args.dataset_dir = snapshot_download(
            repo_id=args.hf_dataset, repo_type="dataset",
        )
        print(f"Downloaded dataset to: {args.dataset_dir}")

    if not args.dataset_dir:
        parser.error("Either --dataset-dir or --hf-dataset is required")

    # Load model + processor
    print(f"Loading model: {args.model_name}")
    model = AutoModel.from_pretrained(args.model_name, trust_remote_code=True)
    processor = load_processor(args.model_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    # Load dataset (no transform needed — processor handles preprocessing)
    print(f"Loading ZooClaw-Fashion dataset from: {args.dataset_dir}")
    task_data = load_zooclaw_dataset(
        args.dataset_dir,
        tasks=args.tasks,
        transform=None,  # raw PIL images — processor handles it
        use_long_queries=args.long_queries,
    )

    # Evaluate each task
    all_results = {}
    for task_name, data in task_data.items():
        print(f"\n{'='*60}")
        print(f"Evaluating: {task_name}")
        print(f"  Queries: {len(data['query_dataset'])} ({data['query_modality']})")
        print(f"  Corpus:  {len(data['corpus_dataset'])} ({data['corpus_modality']})")
        print(f"{'='*60}")

        results = evaluate_zooclaw_task(
            model=model,
            task_data=data,
            processor=processor,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            top_k=args.top_k,
            max_length=args.max_length,
        )
        all_results[task_name] = results

        print(f"\nResults for {task_name}:")
        for k, v in results.items():
            print(f"  {k}: {v:.4f}")

    # Summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    for task_name, results in all_results.items():
        print(f"\n{task_name}:")
        for k, v in results.items():
            print(f"  {k}: {v:.4f}")

    # Save results
    if args.output:
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()

# LookBench: A Live and Holistic Open Benchmark for Fashion Image Retrieval

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyPI](https://img.shields.io/pypi/dm/look-bench?label=PyPI%20downloads)](https://pypi.org/project/look-bench/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2601.14706-b31b1b.svg)](https://arxiv.org/abs/2601.14706)
[![Project Page](https://img.shields.io/badge/Project-Page-green.svg)](https://serendipityoneinc.github.io/look-bench-page/)
[![Dataset](https://img.shields.io/badge/🤗-Dataset-yellow.svg)](https://huggingface.co/datasets/srpone/look-bench)
[![Model](https://img.shields.io/badge/🤗-GR--Lite-orange.svg)](https://huggingface.co/srpone/gr-lite)

**LookBench** is a live, holistic, and challenging benchmark for fashion image retrieval in real e-commerce settings. This repository provides the official evaluation code and model implementations.

## 📰 News

- **[2026-01]** [LookBench](https://arxiv.org/abs/2601.14706) paper released on arXiv
- **[2026-01]** [GR-Lite](https://huggingface.co/srpone/gr-lite) open-source model released
- **[2026-01]** [Initial benchmark dataset](https://huggingface.co/datasets/srpone/look-bench) released

## 📖 Overview

LookBench addresses the limitations of existing fashion retrieval benchmarks by providing:

- **🔄 Continuously Refreshing Samples**: Mitigates data contamination with time-stamped, periodically updated test sets
- **🎯 Diverse Retrieval Tasks**: Covers single-item and multi-item retrieval across real studio, AI-generated studio, real street-look, and AI-generated street-look scenarios
- **📊 Attribute-Supervised Evaluation**: Fine-grained evaluation based on 100+ fashion attributes across categories
- **🏆 Challenging Benchmarks**: Many strong baselines achieve below 60% Recall@1

### Benchmark Subsets

| Dataset | Image Source | # Retrieval Items | Difficulty | # Queries / Corpus |
|---------|--------------|-------------------|------------|-------------------|
| **RealStudioFlat** | Real studio flat-lay product photos | Single | Easy | 1,011 / 62,226 |
| **AIGen-Studio** | AI-generated lifestyle studio images | Single | Medium | 192 / 59,254 |
| **RealStreetLook** | Real street outfit photos | Multi | Hard | 1,000 / 61,553 |
| **AIGen-StreetLook** | AI-generated street outfit compositions | Multi | Hard | 160 / 58,846 |

### Reranking Track: ShopRank-Bench

LookBench also hosts a **text reranking** track. Where the retrieval track ranks a gallery
against a query image, the reranking track asks a cross-encoder which of two candidate
products a shopper would prefer.

**ShopRank-Bench** is a contamination-limited e-commerce preference benchmark: 10,511 pairs
over 2,991 queries drawn from live shopping traffic, labelled by a cross-family LLM judge
panel and tiered by how many families committed a verdict (1,843 gold / 4,445 silver /
4,223 bronze). Every pair ships in two formats — the canonical structured attribute schema
and a natural-language rendering — carrying the same label, so a per-format accuracy gap
measures sensitivity to serialization rather than a difference in labels.

| Track | Task | Metric | Size |
|-------|------|--------|------|
| Image retrieval | query image → gallery | Recall@K, MRR, NDCG, MAP | 4 subsets |
| **ShopRank-Bench** | query + two candidates → preferred | Pairwise accuracy, by tier | 10,511 pairs / 2,991 queries |

- 🤗 Dataset: [srpone/zoowork-shoprank-bench](https://huggingface.co/datasets/srpone/zoowork-shoprank-bench) _(release pending)_
- 🤗 Models: [ZooWork-ShopRanker (0.6B / 4B / 8B)](https://huggingface.co/collections/srpone/rerankers-in-e-commerce-69c4a9acb3eb3f8284d6c0c8)

```python
from datasets import ShopRankPairs
from models import get_model
from metrics import PairwiseAccuracyEvaluator

# from_hub() once the dataset is published; from_jsonl() for a local release file
pairs = ShopRankPairs.from_jsonl("shoprank_bench.jsonl", text_format="structured")
_, reranker = get_model("qwen3-reranker").load_model(
    "Qwen/Qwen3-Reranker-4B", model_path="srpone/zoowork-shopranker-4b")

results = PairwiseAccuracyEvaluator().evaluate_reranker(
    reranker, list(pairs), preferred_key="preferred", rejected_key="rejected")
print(results["pairwise_accuracy"], results["pairwise_accuracy_gold"])
```

Reranking needs one extra dependency: `pip install look-bench[reranking]`.

## 🚀 Quick Start

### Installation

**Option 1: Install from PyPI (Recommended)**

```bash
pip install look-bench
```

**Option 2: Install from Source**

```bash
# Clone the repository
git clone https://github.com/SerendipityOneInc/look-bench.git
cd look-bench

# Install in development mode
pip install -e .

# Or install dependencies only
pip install -r requirements.txt
```

**Optional: Install with Examples Support**

For running example notebooks and scripts that require matplotlib:

```bash
pip install look-bench[examples]
```

### Load Dataset from Hugging Face

The LookBench dataset is hosted on Hugging Face and can be loaded directly:

**Option 1: Using look-bench utility (Recommended)**

```python
from look_bench.utils import load_lookbench_dataset

# Load a specific config
dataset = load_lookbench_dataset("real_studio_flat")

# Access query and gallery splits
query_data = dataset['query']
gallery_data = dataset['gallery']

print(f"Query samples: {len(query_data)}")
print(f"Gallery samples: {len(gallery_data)}")
```

**Option 2: Using Hugging Face datasets directly**

```python
from datasets import load_dataset

# Load a specific config
dataset = load_dataset("srpone/look-bench", "real_studio_flat")

# Access query and gallery splits
query_data = dataset['query']
gallery_data = dataset['gallery']

print(f"Query samples: {len(query_data)}")
print(f"Gallery samples: {len(gallery_data)}")
```

### Quick Evaluation

```python
import torch
from manager import ConfigManager, ModelManager

# Load model
config_manager = ConfigManager('configs/config.yaml')
model_manager = ModelManager(config_manager)

model, _ = model_manager.load_model('clip')
transform = model_manager.get_transform('clip')

# Extract features from an image
sample = dataset['real_studio_flat']['query'][0]
image_tensor = transform(sample['image']).unsqueeze(0)

if torch.cuda.is_available():
    model = model.cuda()
    image_tensor = image_tensor.cuda()

with torch.no_grad():
    features = model(image_tensor)

print(f"Feature shape: {features.shape}")
```

### Run Full Evaluation

```bash
# Run evaluation with default configuration
python main.py

# Run with specific model
python main.py --pipeline evaluation --model clip

# Use custom configuration
python main.py --config configs/config.yaml
```

### Example Scripts & Notebooks

We provide both **Python scripts** and **Google Colab notebooks** for easy experimentation:

#### 📓 Colab Notebooks (Run in Browser)

- **[01_quickstart.ipynb](notebooks/01_quickstart.ipynb)** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SerendipityOneInc/look-bench/blob/main/notebooks/01_quickstart.ipynb) - Basic usage and dataset exploration
- **[02_model_evaluation.ipynb](notebooks/02_model_evaluation.ipynb)** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SerendipityOneInc/look-bench/blob/main/notebooks/02_model_evaluation.ipynb) - Complete evaluation pipeline
- **[03_custom_model.ipynb](notebooks/03_custom_model.ipynb)** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SerendipityOneInc/look-bench/blob/main/notebooks/03_custom_model.ipynb) - Integrate custom models

#### 🐍 Python Scripts (Run Locally)

- **[examples/00_data_exploration.py](examples/00_data_exploration.py)** - Dataset exploration and statistics
- **[examples/01_load_grlite_model.py](examples/01_load_grlite_model.py)** - Load and test GR-Lite model
- **[examples/02_model_evaluation.py](examples/02_model_evaluation.py)** - Complete model evaluation pipeline
- **[examples/03_custom_model.py](examples/03_custom_model.py)** - Integrate your own custom models

```bash
# Run examples locally
python examples/00_data_exploration.py
python examples/01_load_grlite_model.py
python examples/02_model_evaluation.py
python examples/03_custom_model.py
```

## 🏗️ Architecture

```
look-bench/
├── main.py                 # Main entry point (config-driven)
├── manager.py              # Configuration, model, and data managers
├── runner/                 # Pipeline execution framework
│   ├── base_pipeline.py   # Base pipeline class
│   ├── evaluator.py       # Core evaluation logic
│   ├── pipeline.py        # Pipeline registry
│   ├── evaluation_pipeline.py      # Standard evaluation pipeline
│   └── feature_extraction_pipeline.py  # Feature extraction pipeline
├── models/                 # Model implementations and registry
│   ├── base.py            # Base model interface
│   ├── registry.py        # Model registration system
│   ├── factory.py         # Model factory
│   ├── clip_model.py      # CLIP model
│   ├── siglip_model.py    # SigLIP model
│   └── dinov2_model.py    # DINOv2 model
├── datasets/               # Dataset loading (BEIR-style)
│   ├── base.py            # Base dataset implementation
│   └── registry.py        # Dataset registry
├── metrics/                # Evaluation metrics
│   ├── rank.py            # Recall@K
│   ├── mrr.py             # Mean Reciprocal Rank
│   ├── ndcg.py            # Normalized Discounted Cumulative Gain
│   └── map.py             # Mean Average Precision
├── configs/                # Configuration files
│   └── config.yaml        # Main configuration
└── utils/                  # Utilities and logging
```

## 🎯 Supported Models

| Model | Architecture | Input Size | Embedding Dim | Framework |
|-------|--------------|------------|---------------|-----------|
| **CLIP** | Vision Transformer | 224×224 | 512 | PyTorch |
| **SigLIP** | Vision Transformer | 224×224 | 768 | PyTorch |
| **DINOv2** | Vision Transformer | 224×224 | 768 | PyTorch |
| **GR-Lite** | Vision Transformer | 336×336 | 1024 | PyTorch |

Reranking models score a (query, candidate) pair directly rather than producing an
embedding, so they register through `BaseReranker` instead of `BaseModel`:

| Model | Architecture | Scoring | Adapter |
|-------|--------------|---------|---------|
| **Qwen3-Reranker** | Causal LM cross-encoder | P("yes") vs P("no") at final position | optional LoRA |

## ⚙️ Configuration

Edit `configs/config.yaml` to configure models and evaluation settings:

```yaml
# Pipeline configuration
pipeline:
  name: "evaluation"  # evaluation, feature_extraction
  model: "clip"
  dataset: "fashion200k"
  args: {}

# Model configuration
clip:
  enabled: true
  model_name: "openai/clip-vit-base-patch16"
  input_size: 224
  embedding_dim: 512
  device: "cuda"

# Evaluation settings
evaluation:
  metric: "recall"
  top_k: [1, 5, 10, 20]
  l2norm: true
```

## 📊 Evaluation Metrics

LookBench supports multiple evaluation metrics:

- **Recall@K**: Top-K retrieval accuracy (K=1, 5, 10, 20)
- **MRR**: Mean Reciprocal Rank
- **NDCG@K**: Normalized Discounted Cumulative Gain
- **MAP**: Mean Average Precision

### Fine-Grained Evaluation

All metrics are computed with attribute-level matching:
- **Fine Recall@1**: Requires exact category and all attributes to match
- **Coarse Recall@1**: Only requires category to match
- **nDCG@K**: Graded relevance based on attribute overlap

## 🔧 Advanced Usage

### Custom Model Integration

LookBench makes it easy to integrate your own models using the registry pattern. Here's a quick example:

```python
from models.base import BaseModel
from models.registry import register_model
import torch.nn as nn
from torchvision import models, transforms

@register_model("resnet50", metadata={
    "description": "ResNet-50 for fashion retrieval",
    "framework": "PyTorch",
    "input_size": 224,
    "embedding_dim": 2048
})
class ResNet50Model(BaseModel):
    @classmethod
    def load_model(cls, model_name: str, model_path: str = None):
        model = models.resnet50(pretrained=True)
        model = nn.Sequential(*list(model.children())[:-1])  # Remove FC layer
        
        # Wrapper to flatten output
        class Wrapper(nn.Module):
            def __init__(self, backbone):
                super().__init__()
                self.backbone = backbone
            def forward(self, x):
                return self.backbone(x).squeeze(-1).squeeze(-1)
        
        return Wrapper(model), cls()
    
    @classmethod
    def get_transform(cls, input_size: int = 224):
        return transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
```

Then add your model to `configs/config.yaml`:

```yaml
resnet50:
  enabled: true
  model_name: "resnet50"
  model_path: null  # or path to your weights
  input_size: 224
  embedding_dim: 2048
  device: "cuda"
```

**For complete examples, see [examples/03_custom_model.py](examples/03_custom_model.py)**

### Custom Pipeline

Create custom evaluation pipelines:

```python
from runner.base_pipeline import BasePipeline
from runner.pipeline import register_pipeline

@register_pipeline("custom_pipeline")
class CustomPipeline(BasePipeline):
    def get_pipeline_name(self) -> str:
        return "custom_pipeline"
    
    def run(self, **kwargs):
        # Your custom logic here
        model_name = kwargs.get('model_name', 'clip')
        dataset_type = kwargs.get('dataset_type', 'fashion200k')
        
        # Load model and data
        model, _ = self.model_manager.load_model(model_name)
        # ... your evaluation logic
        
        return {"status": "success", "results": results}
```

## 📈 Results

### Fine Recall@1 Performance

Our GR-Lite model achieves state-of-the-art performance on LookBench. Fine Recall@1 requires exact category and all attributes to match:

| Model | Resolution / Emb. | AIGen-StreetLook | AIGen-Studio | RealStreetLook | RealStudioFlat | Overall |
|-------|-------------------|------------------|--------------|----------------|----------------|---------|
| **GR-Pro** (Ours) | 336 / 1024 | **63.67** | **54.88** | **44.75** | 51.55 | **49.80** |
| **GR-Lite** (Ours, Open) | 336 / 1024 | 62.47 | 52.08 | 43.84 | **51.70** | 49.18 |
| Marqo-FashionSigLIP | 224 / 768 | 66.27 | 58.53 | 42.43 | 51.86 | 49.44 |
| Marqo-FashionCLIP | 224 / 512 | 63.22 | 54.93 | 41.87 | 51.68 | 48.63 |
| SigLIP2-B/16 | 384 / 768 | 57.83 | 54.97 | 39.35 | 49.12 | 46.10 |
| SigLIP2-L/16 | 384 / 1024 | 51.89 | 48.57 | 35.91 | 44.78 | 41.86 |
| PP-ShiTuV2 | 224 / 512 | 30.06 | 33.69 | 32.77 | 43.22 | 37.17 |
| DINOv3-ViT-L | 224 / 1024 | 20.24 | 27.66 | 26.27 | 39.85 | 31.83 |
| DINOv2-ViT-L | 224 / 1024 | 24.29 | 25.05 | 22.99 | 37.66 | 29.57 |
| CLIP-L/14 | 336 / 768 | 25.28 | 25.95 | 21.09 | 40.35 | 30.08 |
| CLIP-B/16 | 224 / 512 | 17.86 | 13.75 | 16.80 | 34.75 | 24.36 |

### Coarse Recall@1 Performance

Coarse Recall@1 only requires category match (more lenient):

| Model | Resolution / Emb. | AIGen-StreetLook | AIGen-Studio | RealStreetLook | RealStudioFlat | Overall |
|-------|-------------------|------------------|--------------|----------------|----------------|---------|
| **GR-Pro** (Ours) | 336 / 1024 | **92.50** | **92.75** | **79.82** | **94.16** | **87.93** |
| **GR-Lite** (Ours, Open) | 336 / 1024 | 88.75 | 90.16 | 76.76 | 92.68 | 85.54 |
| Marqo-FashionSigLIP | 224 / 768 | 90.00 | 93.78 | 73.39 | 88.63 | 82.77 |
| Marqo-FashionCLIP | 224 / 512 | 84.38 | 87.05 | 75.33 | 88.72 | 82.68 |
| SigLIP2-B/16 | 384 / 768 | 86.25 | 90.67 | 72.17 | 88.33 | 81.62 |
| SigLIP2-L/16 | 384 / 1024 | 80.62 | 90.67 | 68.20 | 84.97 | 78.12 |
| CLIP-L/14 | 336 / 768 | 46.88 | 56.48 | 45.26 | 76.85 | 59.91 |
| CLIP-B/16 | 224 / 512 | 35.62 | 32.12 | 33.54 | 67.26 | 48.11 |

### nDCG@5 Performance

nDCG@5 evaluates ranking quality with graded relevance based on attribute overlap:

| Model | Resolution / Emb. | AIGen-StreetLook | AIGen-Studio | RealStreetLook | RealStudioFlat | Overall |
|-------|-------------------|------------------|--------------|----------------|----------------|---------|
| **GR-Pro** (Ours) | 336 / 1024 | **63.67** | **54.88** | **44.75** | 51.55 | **49.80** |
| **GR-Lite** (Ours, Open) | 336 / 1024 | 62.47 | 52.08 | 43.84 | **51.70** | 49.18 |
| Marqo-FashionSigLIP | 224 / 768 | 66.27 | 58.53 | 42.43 | 51.86 | 49.44 |
| Marqo-FashionCLIP | 224 / 512 | 63.22 | 54.93 | 41.87 | 51.68 | 48.63 |
| SigLIP2-B/16 | 384 / 768 | 57.83 | 54.97 | 39.35 | 49.12 | 46.10 |

*See our [paper](https://arxiv.org/abs/2601.14706) for complete results including MRR and additional models.*

## 📄 Citation

If you use LookBench in your research, please cite:

```bibtex
@article{gao2026lookbench,
  title={LookBench: A Live and Holistic Open Benchmark for Fashion Image Retrieval}, 
  author={Chao Gao and Siqiao Xue and Yimin Peng and Jiwen Fu and Tingyi Gu and Shanshan Li and Fan Zhou},
  year={2026},
  url={https://arxiv.org/abs/2601.14706}, 
  journal={arXiv preprint arXiv:2601.14706},
}
```


## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

The GR-Lite model weights are distributed under the DINOv3 License as they are derived from Meta's DINOv3 model.


# LookBench: A Live and Holistic Open Benchmark for Fashion Image Retrieval

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2601.14706-b31b1b.svg)](https://arxiv.org/abs/2601.14706)
[![Project Page](https://img.shields.io/badge/Project-Page-green.svg)](https://serendipityoneinc.github.io/look-bench-page/)
[![Dataset](https://img.shields.io/badge/🤗-Dataset-yellow.svg)](https://huggingface.co/datasets/srpone/look-bench)
[![Model](https://img.shields.io/badge/🤗-GR--Lite-orange.svg)](https://huggingface.co/srpone/gr-lite)

**LookBench** is a live, holistic, and challenging benchmark for fashion image retrieval in real e-commerce settings. This repository provides the official evaluation code and model implementations.

## 📰 News

- **[2026-01]** LookBench paper released on arXiv
- **[2026-01]** GR-Lite open-source model released
- **[2026-01]** Initial benchmark dataset released

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

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/SerendipityOneInc/look-bench.git
cd look-bench

# Install dependencies
pip install -r requirements.txt
```

### Download Dataset

Download the LookBench dataset from [Hugging Face](https://huggingface.co/datasets/srpone/look-bench) and organize it in the `data/` directory.

### Run Evaluation

```bash
# Run evaluation with default configuration
python main.py

# Run with specific model and dataset
python main.py --model clip --dataset fashion200k

# Use custom configuration
python main.py --config configs/config.yaml
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

```python
from models.base import BaseModel
from models.registry import register_model

@register_model("custom_model", metadata={
    "description": "Custom fashion embedding model",
    "framework": "PyTorch",
    "input_size": 224,
    "embedding_dim": 512
})
class CustomModel(BaseModel):
    @classmethod
    def load_model(cls, model_name: str, model_path: str = None):
        # Load your model
        model = YourModel()
        return model, cls()
    
    @classmethod
    def get_transform(cls, input_size: int):
        # Define preprocessing
        return your_transform
```

### Custom Pipeline

```python
from runner.base_pipeline import BasePipeline
from runner.pipeline import register_pipeline

@register_pipeline("custom_pipeline")
class CustomPipeline(BasePipeline):
    def get_pipeline_name(self) -> str:
        return "custom_pipeline"
    
    def run(self, **kwargs):
        # Your pipeline logic
        return results
```

## 📈 Results

Our GR-Lite model achieves state-of-the-art performance on LookBench:

| Model | RealStudioFlat | AIGen-Studio | RealStreetLook | AIGen-StreetLook | Overall |
|-------|----------------|--------------|----------------|------------------|---------|
| **GR-Lite** | 51.70 | 52.08 | 43.84 | 62.47 | 49.18 |
| Marqo-FashionSigLIP | 51.86 | 58.53 | 42.43 | 66.27 | 49.44 |
| SigLIP2-B/16 | 49.12 | 54.97 | 39.35 | 57.83 | 46.10 |
| CLIP-L/14 | 40.35 | 25.95 | 21.09 | 25.28 | 30.08 |

*Fine Recall@1 scores. See [paper](https://arxiv.org/abs/2601.14706) for complete results.*

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

## 🔗 Links

- **Paper**: [https://arxiv.org/abs/2601.14706](https://arxiv.org/abs/2601.14706)
- **Project Page**: [https://serendipityoneinc.github.io/look-bench-page/](https://serendipityoneinc.github.io/look-bench-page/)
- **Dataset**: [https://huggingface.co/datasets/srpone/look-bench](https://huggingface.co/datasets/srpone/look-bench)
- **GR-Lite Model**: [https://huggingface.co/srpone/gr-lite](https://huggingface.co/srpone/gr-lite)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

The GR-Lite model weights are distributed under the DINOv3 License as they are derived from Meta's DINOv3 model.

## 🙏 Acknowledgments

- **BEIR**: Inspiration for benchmark organization
- **Meta AI**: DINOv3 foundation model
- **HuggingFace**: Model implementations and hosting
- **PyTorch**: Deep learning framework

## 📞 Contact

For questions and issues:
- Create an issue on [GitHub](https://github.com/SerendipityOneInc/look-bench/issues)
- Visit our [project page](https://serendipityoneinc.github.io/look-bench-page/)

---

**LookBench** is designed for research and production use in fashion image retrieval. We welcome contributions and feedback!

# LookBench: Fashion Image Retrieval Benchmark

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**LookBench** is a comprehensive, production-ready benchmarking framework for evaluating fashion image retrieval models. Inspired by BEIR's organization style, LookBench provides a unified interface for multiple state-of-the-art vision models with standardized evaluation metrics and professional logging systems.

## 🚀 Features

- **Multi-Model Support**: Integrated support for CLIP, SigLIP, DINOv2, and more via registry pattern
- **BEIR-Style Organization**: Clean dataset organization similar to BEIR benchmark
- **Standardized Evaluation**: Consistent evaluation pipeline with multiple metrics (Recall@K, MRR, NDCG, MAP)
- **Registry Pattern**: Easy model and dataset registration for extensibility
- **Professional Architecture**: Clean, modular design with factory classes and registry pattern
- **Comprehensive Logging**: Structured logging with performance monitoring and error tracking
- **Flexible Configuration**: YAML-based configuration for easy model and dataset management
- **Production Ready**: Error handling, device management, and batch processing optimization

## 🏗️ Architecture

```
look-bench/
├── main.py                 # Main entry point and benchmark runner
├── manager.py              # Configuration, model, and data managers
├── runner/                 # Pipeline execution framework
│   ├── __init__.py        # Runner module exports
│   ├── base_pipeline.py   # Base pipeline class
│   ├── evaluator.py       # Core evaluation logic
│   ├── pipeline.py        # Pipeline registry
│   ├── evaluation_pipeline.py      # Standard evaluation pipeline
│   └── feature_extraction_pipeline.py  # Feature extraction pipeline
├── models/                 # Model implementations and registry
│   ├── base.py            # Base model interface
│   ├── registry.py        # Model registration system
│   ├── factory.py         # Model factory for instantiation
│   ├── clip_model.py      # CLIP model implementation
│   ├── siglip_model.py    # SigLIP model implementation
│   └── dinov2_model.py    # DINOv2 model implementation
├── datasets/               # Dataset loading (BEIR-style)
│   ├── base.py            # Base dataset implementation
│   └── registry.py        # Dataset registry
├── metrics/                # Evaluation metrics
│   ├── base.py            # Base evaluator interface
│   ├── rank.py            # Recall@K evaluation
│   ├── mrr.py             # Mean Reciprocal Rank
│   ├── ndcg.py            # Normalized Discounted Cumulative Gain
│   ├── map.py             # Mean Average Precision
│   └── factory.py         # Evaluator factory
├── configs/                # Configuration files
│   └── config.yaml        # Main configuration file
├── scripts/                # Utility scripts
│   ├── example_dataset_converter.py
│   └── download_datasets.sh
├── utils/                  # Utility functions and logging
└── logs/                   # Log files and outputs
```

## 🎯 Supported Models

| Model | Architecture | Input Size | Embedding Dim | Framework |
|-------|--------------|------------|---------------|-----------|
| **CLIP** | Vision Transformer | 224×224 | 512 | PyTorch |
| **SigLIP** | Vision Transformer | 224×224 | 768 | PyTorch |
| **DINOv2** | Vision Transformer | 224×224 | 768 | PyTorch |

## 📊 Supported Datasets (BEIR-Style)

LookBench supports multiple fashion retrieval datasets:

- **Fashion200K**: Large-scale fashion retrieval dataset
- **DeepFashion**: Fashion understanding dataset with attribute prediction
- **DeepFashion2**: Advanced fashion dataset with detailed annotations
- **Fashion Product Images**: Product classification and retrieval
- **Product10K**: Large-scale product retrieval dataset

## 📦 Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- 8GB+ RAM

### Install Dependencies

```bash
# Clone the repository
git clone <repository-url>
cd look-bench

# Install required packages
pip install -r requirements.txt

# For GPU support (optional)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Environment Setup

The framework automatically configures cache directories for:
- HuggingFace models and datasets
- PyTorch models
- Transformers cache

## ⚙️ Configuration

### Basic Configuration

Edit `configs/config.yaml` to configure your models and evaluation settings:

```yaml
# Global settings
global:
  cache_dir: "~/.cache"
  log_level: "INFO"

# Model configuration
clip:
  enabled: true
  model_name: "openai/clip-vit-base-patch16"
  input_size: 224
  embedding_dim: 512
  device: "cuda"

# Evaluation settings
evaluation:
  metric: "recall"  # recall, mrr, ndcg, map
  top_k: [1, 5, 10, 20]
  l2norm: true

# Dataset settings
datasets:
  fashion200k:
    data_root: "data/fashion200k"
    splits:
      query: "query"
      gallery: "gallery"
    parquet_files:
      query: "query.parquet"
      gallery: "gallery.parquet"
  batch_size: 128
  num_workers: 8
```

## 🚀 Usage

### Quick Start

```python
from main import LookBench

# Initialize benchmark runner
benchmark = LookBench("configs/config.yaml")

# Run evaluation
metrics = benchmark.run_evaluation(
    model_name="clip",
    dataset_type="fashion200k"
)

print(f"Recall@1: {metrics['recall@1']:.2f}%")
```

### Command Line Usage

```bash
# Run benchmark with default configuration (evaluation pipeline)
python main.py

# Run specific model evaluation
python main.py --model clip --dataset fashion200k

# Run feature extraction pipeline
python main.py --pipeline feature_extraction --model clip --dataset fashion200k --save-features features.pt

# Use custom config
python main.py --config configs/custom_config.yaml --model siglip

# List available pipelines
python -c "from runner import list_available_pipelines; print(list_available_pipelines())"
```

### Pipeline System

LookBench uses a flexible pipeline system that allows you to run different types of operations:

#### Available Pipelines

- **evaluation**: Standard evaluation pipeline (default)
  - Runs full evaluation: model loading → feature extraction → evaluation
- **feature_extraction**: Extract features without evaluation
  - Useful for extracting features once and reusing them

#### Using Pipelines

```python
from main import LookBench

benchmark = LookBench("configs/config.yaml")

# Run evaluation pipeline
metrics = benchmark.run_evaluation(model_name="clip", dataset_type="fashion200k")

# Run feature extraction pipeline
results = benchmark.run_pipeline(
    pipeline_name="feature_extraction",
    model_name="clip",
    dataset_type="fashion200k",
    save_path="features.pt"
)
```

#### Creating Custom Pipelines

```python
from runner.base_pipeline import BasePipeline
from runner.pipeline import register_pipeline

@register_pipeline("custom_pipeline")
class CustomPipeline(BasePipeline):
    def get_pipeline_name(self) -> str:
        return "custom_pipeline"

    def run(self, **kwargs) -> Dict[str, Any]:
        # Your custom pipeline logic
        return {"result": "custom_output"}
```

### Custom Model Integration

```python
from models.base import BaseModel
from models.registry import register_model

@register_model("custom_model", metadata={
    "description": "Custom fashion image embedding model",
    "framework": "PyTorch",
    "input_size": 224,
    "embedding_dim": 512
})
class CustomEmbeddingModel(BaseModel):

    @classmethod
    def load_model(cls, model_name: str, model_path: str = None):
        # Load your custom model
        model = YourCustomModel()
        return model, cls()

    @classmethod
    def get_transform(cls, input_size: int):
        # Define your preprocessing pipeline
        return your_transform_pipeline
```

### Custom Dataset Integration

```python
from datasets.registry import register_dataset

# Register a new dataset
register_dataset("my_fashion_dataset", {
    "description": "My custom fashion dataset",
    "num_categories": 100,
    "tasks": ["image_retrieval"],
    "splits": ["query", "gallery"]
})
```

## 📊 Evaluation Metrics

LookBench supports multiple evaluation metrics:

- **Recall@K**: Top-K retrieval accuracy (Recall@1, Recall@5, Recall@10, Recall@20)
- **MRR**: Mean Reciprocal Rank
- **NDCG**: Normalized Discounted Cumulative Gain
- **MAP**: Mean Average Precision

### Feature Analysis

- **L2 Normalization**: Optional feature normalization for fair comparison
- **Embedding Dimensions**: Configurable output dimensions
- **Batch Processing**: Optimized for large-scale evaluation

## 🔧 Advanced Usage

### Custom Evaluators

```python
from metrics.base import BaseEvaluator
from metrics.factory import register_evaluator

class CustomEvaluator(BaseEvaluator):
    def get_metric_name(self) -> str:
        return "CustomMetric"

    def metric_eval(self, sorted_indices, rank_val, query_label, gallery_label):
        # Implement your custom evaluation logic
        return score

# Register the evaluator
register_evaluator("custom", CustomEvaluator)
```

### Performance Optimization

```python
# Configure batch processing
config = {
    'batch_size': 256,      # Larger batches for GPU
    'num_workers': 16,      # More workers for I/O
    'pin_memory': True,     # Faster GPU transfer
    'drop_last': False      # Keep all samples
}
```

## 📈 Performance Monitoring

### Logging Features

- **Structured Logging**: JSON-formatted logs with context
- **Performance Metrics**: Timing and memory usage tracking
- **Error Handling**: Comprehensive error logging with context
- **Progress Tracking**: Real-time progress bars for long operations

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size in config
   - Use gradient checkpointing
   - Enable mixed precision

2. **Model Loading Errors**
   - Check model paths in config
   - Verify model compatibility
   - Check cache directory permissions

3. **Performance Issues**
   - Increase num_workers for I/O bound operations
   - Use pin_memory for GPU operations
   - Optimize batch size for your hardware

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable detailed logging
benchmark = LookBench("configs/config.yaml")
benchmark.run_evaluation("clip", "fashion200k")
```

## 🤝 Contributing

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt

# Code formatting
black .
isort .

# Linting
flake8 .
mypy .
```

### Code Style

- Follow PEP 8 guidelines
- Use type hints
- Add comprehensive docstrings
- Include unit tests for new features

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **BEIR**: Inspiration for dataset organization style
- **HuggingFace Transformers**: Model implementations
- **PyTorch**: Deep learning framework
- Fashion dataset creators and contributors

## 📞 Support

For questions and support:
- Create an issue on GitHub
- Check the documentation
- Review the configuration examples

---

**Note**: LookBench is designed for production use and research purposes. Ensure you have appropriate licenses for any models you use in production environments.

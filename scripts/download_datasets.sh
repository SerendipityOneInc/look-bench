#!/bin/bash
# Download fashion datasets for benchmarking
# This is a template script - adjust URLs and paths as needed

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DATA_DIR="${PROJECT_ROOT}/data"

echo "Creating data directory: ${DATA_DIR}"
mkdir -p "${DATA_DIR}"

# Function to download and extract dataset
download_dataset() {
    local dataset_name=$1
    local dataset_url=$2
    local output_dir="${DATA_DIR}/${dataset_name}"
    
    echo "Downloading ${dataset_name}..."
    mkdir -p "${output_dir}"
    
    # Example: Download using wget or curl
    # wget -O "${output_dir}/dataset.zip" "${dataset_url}"
    # unzip "${output_dir}/dataset.zip" -d "${output_dir}"
    
    echo "${dataset_name} download complete"
}

# Fashion200K
echo "=== Fashion200K Dataset ==="
echo "Please download Fashion200K from: https://github.com/xthan/fashion-200k"
echo "Extract to: ${DATA_DIR}/fashion200k"

# DeepFashion
echo ""
echo "=== DeepFashion Dataset ==="
echo "Please download DeepFashion from: http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html"
echo "Extract to: ${DATA_DIR}/deepfashion"

# DeepFashion2
echo ""
echo "=== DeepFashion2 Dataset ==="
echo "Please download DeepFashion2 from: https://github.com/switchablenorms/DeepFashion2"
echo "Extract to: ${DATA_DIR}/deepfashion2"

# Product10K
echo ""
echo "=== Product10K Dataset ==="
echo "Please download Product10K from the official source"
echo "Extract to: ${DATA_DIR}/product10k"

echo ""
echo "=== Dataset Download Instructions ==="
echo "1. Download datasets from the URLs above"
echo "2. Extract them to the corresponding directories in ${DATA_DIR}"
echo "3. Convert them to parquet format using scripts/example_dataset_converter.py"
echo ""
echo "Example conversion command:"
echo "python scripts/example_dataset_converter.py --dataset fashion200k --data_root ${DATA_DIR}/fashion200k --output_root ${DATA_DIR}/fashion200k"


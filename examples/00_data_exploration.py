#!/usr/bin/env python3
"""
LookBench Data Exploration Example
Download the dataset and explore its structure, statistics, and samples
"""

import matplotlib.pyplot as plt
from datasets import load_dataset
from collections import Counter
import numpy as np


def print_section(title):
    """Print a formatted section header"""
    print("\n" + "="*70)
    print(f" {title}")
    print("="*70)


def explore_dataset_structure(dataset):
    """Explore and print dataset structure"""
    print_section("Dataset Structure")
    
    print(f"\nAvailable subsets: {list(dataset.keys())}")
    print(f"Total subsets: {len(dataset.keys())}\n")
    
    for subset_name in dataset.keys():
        print(f"📁 {subset_name}:")
        # Each config returns a DatasetDict with splits
        if hasattr(dataset[subset_name], 'keys'):
            for split_name in dataset[subset_name].keys():
                num_samples = len(dataset[subset_name][split_name])
                print(f"   ├─ {split_name}: {num_samples:,} samples")
        print()


def analyze_subset_statistics(dataset, subset_name):
    """Analyze and print statistics for a specific subset"""
    print_section(f"Statistics for '{subset_name}'")
    
    if subset_name not in dataset:
        print(f"Subset '{subset_name}' not found!")
        return
    
    # Query statistics
    # Each config is a DatasetDict, access splits directly
    subset_data = dataset[subset_name]
    if 'query' in subset_data:
        query_data = subset_data['query']
        print(f"\n📊 Query Split ({len(query_data):,} samples):")
        
        # Category distribution
        categories = [sample['category'] for sample in query_data]
        category_counts = Counter(categories)
        print(f"\n  Categories ({len(category_counts)} unique):")
        for cat, count in category_counts.most_common(10):
            print(f"    • {cat}: {count}")
        
        # Task distribution
        if 'task' in query_data[0]:
            tasks = [sample['task'] for sample in query_data]
            task_counts = Counter(tasks)
            print(f"\n  Tasks:")
            for task, count in task_counts.items():
                print(f"    • {task}: {count} ({count/len(query_data)*100:.1f}%)")
        
        # Difficulty distribution
        if 'difficulty' in query_data[0]:
            difficulties = [sample['difficulty'] for sample in query_data]
            diff_counts = Counter(difficulties)
            print(f"\n  Difficulty levels:")
            for diff, count in diff_counts.items():
                print(f"    • {diff}: {count} ({count/len(query_data)*100:.1f}%)")
        
        # Attribute statistics
        if 'main_attribute' in query_data[0]:
            main_attrs = [sample['main_attribute'] for sample in query_data]
            attr_counts = Counter(main_attrs)
            print(f"\n  Main attributes ({len(attr_counts)} unique):")
            for attr, count in attr_counts.most_common(5):
                print(f"    • {attr}: {count}")
    
    # Gallery statistics
    if 'gallery' in subset_data:
        gallery_data = subset_data['gallery']
        print(f"\n📚 Gallery Split ({len(gallery_data):,} samples):")
        
        categories = [sample['category'] for sample in gallery_data]
        category_counts = Counter(categories)
        print(f"\n  Categories ({len(category_counts)} unique):")
        for cat, count in category_counts.most_common(10):
            print(f"    • {cat}: {count}")


def display_sample_images(dataset, subset_name, num_samples=4):
    """Display sample images from the dataset"""
    print_section(f"Sample Images from '{subset_name}'")
    
    if subset_name not in dataset:
        print(f"Subset '{subset_name}' not found!")
        return
    
    subset_data = dataset[subset_name]
    if 'query' not in subset_data:
        print(f"Query split not found in '{subset_name}'!")
        return
    
    query_data = subset_data['query']
    num_samples = min(num_samples, len(query_data))
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx in range(num_samples):
        sample = query_data[idx]
        ax = axes[idx]
        
        # Display image
        ax.imshow(sample['image'])
        ax.axis('off')
        
        # Create title with metadata
        title = f"Category: {sample['category']}\n"
        if 'main_attribute' in sample:
            title += f"Attribute: {sample['main_attribute']}\n"
        if 'task' in sample:
            title += f"Task: {sample['task']}"
        
        ax.set_title(title, fontsize=9, pad=10)
    
    plt.tight_layout()
    save_path = f'sample_images_{subset_name}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Sample images saved to: {save_path}")
    plt.close()


def compare_subsets(dataset):
    """Compare statistics across all subsets"""
    print_section("Cross-Subset Comparison")
    
    print("\n📊 Subset Comparison:")
    print(f"{'Subset':<25} {'Query':<10} {'Gallery':<10} {'Total':<10}")
    print("-" * 60)
    
    total_queries = 0
    total_gallery = 0
    
    for subset_name in sorted(dataset.keys()):
        subset_data = dataset[subset_name]
        num_queries = len(subset_data['query']) if 'query' in subset_data else 0
        num_gallery = len(subset_data['gallery']) if 'gallery' in subset_data else 0
        total = num_queries + num_gallery
        
        print(f"{subset_name:<25} {num_queries:<10,} {num_gallery:<10,} {total:<10,}")
        total_queries += num_queries
        total_gallery += num_gallery
    
    print("-" * 60)
    print(f"{'TOTAL':<25} {total_queries:<10,} {total_gallery:<10,} {total_queries + total_gallery:<10,}")


def generate_dataset_summary(dataset):
    """Generate a comprehensive dataset summary"""
    print_section("Dataset Summary Report")
    
    total_images = 0
    total_categories = set()
    
    for subset_name in dataset.keys():
        subset_data = dataset[subset_name]
        for split_name in subset_data.keys():
            split_data = subset_data[split_name]
            total_images += len(split_data)
            
            # Collect unique categories
            for sample in split_data:
                if 'category' in sample:
                    total_categories.add(sample['category'])
    
    print(f"\n📈 Overall Statistics:")
    print(f"  • Total subsets: {len(dataset.keys())}")
    print(f"  • Total images: {total_images:,}")
    print(f"  • Unique categories: {len(total_categories)}")
    
    print(f"\n📋 Available subsets:")
    for subset in sorted(dataset.keys()):
        print(f"  • {subset}")


def main():
    print("="*70)
    print(" LookBench Dataset Exploration")
    print("="*70)
    
    # 1. Download dataset
    print("\n[1/6] Downloading LookBench dataset from Hugging Face...")
    print("This may take a few minutes on first run...")
    
    # Available configs
    configs = ['aigen_streetlook', 'aigen_studio', 'real_streetlook', 'real_studio_flat', 'noise']
    
    try:
        # Load all configs
        dataset = {}
        for config_name in configs:
            print(f"  Loading {config_name}...")
            dataset[config_name] = load_dataset("srpone/look-bench", config_name)
        print("✅ Dataset downloaded successfully!")
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        print("Please check your internet connection and try again.")
        return
    
    # 2. Explore structure
    print("\n[2/6] Exploring dataset structure...")
    explore_dataset_structure(dataset)
    
    # 3. Generate summary
    print("\n[3/6] Generating dataset summary...")
    generate_dataset_summary(dataset)
    
    # 4. Compare subsets
    print("\n[4/6] Comparing subsets...")
    compare_subsets(dataset)
    
    # 5. Analyze each subset
    print("\n[5/6] Analyzing individual subsets...")
    # Analyze main subsets (skip noise for detailed analysis)
    main_subsets = ['real_studio_flat', 'aigen_studio', 'real_streetlook', 'aigen_streetlook']
    for subset_name in main_subsets:
        if subset_name in dataset:
            analyze_subset_statistics(dataset, subset_name)
    
    # 6. Display sample images
    print("\n[6/6] Displaying sample images...")
    for subset_name in ['real_studio_flat', 'real_streetlook']:
        if subset_name in dataset:
            try:
                display_sample_images(dataset, subset_name, num_samples=4)
            except Exception as e:
                print(f"⚠️  Could not display images for {subset_name}: {e}")
    
    # Final summary
    print_section("Exploration Complete")
    print("\n✅ Dataset exploration completed successfully!")
    print("\n📚 Next steps:")
    print("  1. Review the generated statistics above")
    print("  2. Check the saved sample image files")
    print("  3. Run '01_quickstart.py' to test model inference")
    print("  4. Run '02_model_evaluation.py' for full benchmark evaluation")
    print("\n💡 Paper: https://arxiv.org/abs/2601.14706")
    print("💡 Dataset: https://huggingface.co/datasets/srpone/look-bench")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

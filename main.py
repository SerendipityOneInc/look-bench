#!/usr/bin/env python3
"""
LookBench - Fashion Image Retrieval Benchmark
Main Entry Point
BEIR-style benchmark for evaluating fashion image retrieval models
"""

import argparse
import warnings
import logging

from runner.runner import Runner
from utils.logging import setup_logging

# Suppress warnings
warnings.filterwarnings("ignore")

# Setup professional logging
setup_logging(
    level=logging.INFO,
    console_output=True,
    suppress_external=True
)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="LookBench - Fashion Image Retrieval Benchmark (BEIR-style)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--pipeline",
        type=str,
        default=None,
        help="Pipeline name to run (overrides config)"
    )

    args = parser.parse_args()

    # Build runner from config
    runner = Runner.build_from_config(args.config, pipeline_name=args.pipeline)

    # Run pipeline
    runner.run()


if __name__ == "__main__":
    main()

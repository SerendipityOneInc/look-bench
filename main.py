#!/usr/bin/env python3
"""Run LookBench from a source checkout: `python main.py --config ...`.

Installed users run the `look-bench` command (same entry point: look_bench.main:main).
"""
from look_bench.main import main

if __name__ == "__main__":
    main()

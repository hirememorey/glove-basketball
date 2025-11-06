#!/usr/bin/env python3
"""
Resumable Batch Processing Script for PADIM

This script provides resumable processing for large-scale NBA data collection.
It can handle interruptions gracefully and resume from where it left off.

Usage:
    python resumable_batch_process.py 2022-23 --max-games 100
    python resumable_batch_process.py 2022-23 --status
    python resumable_batch_process.py 2022-23 --reset
"""

import sys
import os

# Add the src directory to the path so we can import PADIM modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from padim.resumable_processor import main

if __name__ == "__main__":
    main()

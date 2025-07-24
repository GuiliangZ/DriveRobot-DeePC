#!/usr/bin/env python3
"""
Main entry point for PRBS data collection.
Run this script to collect data for building Hankel matrices.
"""

import sys
import os

# Add subdirectories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'core'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'data_collection'))

# Import PRBS data collector
from PRBSDataCollector import main

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Main entry point for DeePC-Runtime controller.
Run this script to start the DeePC controller with enhanced PID fallback.
"""

import sys
import os

# Add subdirectories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'core'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'data_collection'))

# Import main controller
from PIDDeePCControllerFixed import main

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Main entry point for Enhanced PID controller standalone mode.
Run this script to use only the enhanced PID controller without DeePC.
"""

import sys
import os
import subprocess

# Get the path to the enhanced PID controller
script_dir = os.path.dirname(os.path.abspath(__file__))
enhanced_pid_path = os.path.join(script_dir, 'core', 'EnhancedPID-Controller-IWR.py')

# Run the enhanced PID controller as a subprocess
# This handles the module with hyphens in the name
try:
    subprocess.run([sys.executable, enhanced_pid_path], check=True)
except KeyboardInterrupt:
    print("\n[Main] Enhanced PID controller stopped by user.")
except Exception as e:
    print(f"[ERROR] Failed to run enhanced PID controller: {e}")
    sys.exit(1)
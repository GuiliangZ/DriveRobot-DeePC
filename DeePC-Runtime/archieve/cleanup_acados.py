#!/usr/bin/env python3
"""
cleanup_acados.py
Utility to clean up old Acados compilation files when parameters change
"""

import os
import shutil

def cleanup_acados_files():
    """Remove old Acados compilation files"""
    print("Cleaning up Acados compilation files...")
    
    # Directories to remove
    dirs_to_remove = [
        "c_generated_code",
        "__pycache__",
        ".acados",
    ]
    
    # Files to remove
    files_to_remove = [
        "DeePC_acados_ocp.json",
        "acados_sim_solver_sfunction.c",
        "acados_ocp_solver_sfunction.c",
    ]
    
    cleaned = False
    
    for dir_name in dirs_to_remove:
        if os.path.exists(dir_name):
            try:
                shutil.rmtree(dir_name)
                print(f"✓ Removed directory: {dir_name}")
                cleaned = True
            except Exception as e:
                print(f"✗ Could not remove {dir_name}: {e}")
    
    for file_name in files_to_remove:
        if os.path.exists(file_name):
            try:
                os.remove(file_name)
                print(f"✓ Removed file: {file_name}")
                cleaned = True
            except Exception as e:
                print(f"✗ Could not remove {file_name}: {e}")
    
    if not cleaned:
        print("No Acados files found to clean.")
    else:
        print("✓ Cleanup completed. You can now run the controller with updated parameters.")

if __name__ == "__main__":
    cleanup_acados_files()
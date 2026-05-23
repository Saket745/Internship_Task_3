#!/usr/bin/env python3
import os
import shutil
import subprocess
import sys
import json

# Configuration
KERNEL_SLUG = "sigce-group/handwritten-char-training"  # Replace with actual slug if different
WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_TEMP_DIR = os.path.join(WORKSPACE_DIR, "output_temp")
MODELS_DIR = os.path.join(WORKSPACE_DIR, "models")
NOTEBOOKS_DIR = os.path.join(WORKSPACE_DIR, "notebooks")

def check_kaggle_cli():
    """Verify that Kaggle CLI is installed and configured."""
    print("Checking Kaggle CLI installation...")
    if shutil.which("kaggle") is None:
        print("Error: Kaggle CLI not found. Please install it using 'pip install kaggle'.")
        sys.exit(1)
        
    # Verify kaggle.json exists
    home_dir = os.path.expanduser("~")
    kaggle_json_path = os.path.join(home_dir, ".kaggle", "kaggle.json")
    if not os.path.exists(kaggle_json_path):
        # Fallback to alternate profile path if redirected
        alt_kaggle_json = r"C:\Users\mssak\maury\.kaggle\kaggle.json"
        if os.path.exists(alt_kaggle_json):
            # Ensure it is in the active user home folder
            os.makedirs(os.path.dirname(kaggle_json_path), exist_ok=True)
            shutil.copy(alt_kaggle_json, kaggle_json_path)
            print(f"Copied credentials from {alt_kaggle_json} to active profile.")
        else:
            print(f"Error: Kaggle API token not found. Please place kaggle.json in {kaggle_json_path}")
            sys.exit(1)
    print("Kaggle CLI is ready.")

def run_command(command, cwd=None):
    """Utility to run a shell command and return output/status."""
    print(f"Executing: {' '.join(command)}")
    result = subprocess.run(command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        print(f"Error executing command: {result.stderr}")
        return False, result.stderr
    return True, result.stdout

def sync_from_kaggle():
    """Sync the latest kernel outputs and notebook file from Kaggle."""
    # Ensure directories exist
    os.makedirs(OUTPUT_TEMP_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(NOTEBOOKS_DIR, exist_ok=True)

    print(f"\n--- Fetching Latest Files for Kernel: {KERNEL_SLUG} ---")
    
    # 1. Pull the notebook file (.ipynb)
    print("Pulling notebook file...")
    success, stdout = run_command(["kaggle", "kernels", "pull", "-p", OUTPUT_TEMP_DIR, KERNEL_SLUG, "--metadata"])
    if not success:
        # Prompt user to input their custom slug if the default doesn't match
        print("\nCould not find kernel. Make sure the KERNEL_SLUG at the top of this script matches your Kaggle URL slug.")
        sys.exit(1)
        
    # 2. Pull the trained model files (outputs)
    print("Downloading model outputs...")
    success, stdout = run_command(["kaggle", "kernels", "output", KERNEL_SLUG, "-p", OUTPUT_TEMP_DIR])
    if not success:
        sys.exit(1)

    print("\n--- Organizing Downloaded Files ---")
    files_processed = []

    # Look through downloaded files and move to correct locations
    for filename in os.listdir(OUTPUT_TEMP_DIR):
        src_path = os.path.join(OUTPUT_TEMP_DIR, filename)
        
        # Move weights and ONNX models to models/
        if filename.endswith(".pth") or filename.endswith(".onnx") or filename.endswith(".onnx.data"):
            dest_path = os.path.join(MODELS_DIR, filename)
            shutil.move(src_path, dest_path)
            print(f"Moved Model weight: {filename} -> models/")
            files_processed.append(dest_path)
            
        # Move notebook file to notebooks/
        elif filename.endswith(".ipynb"):
            dest_path = os.path.join(NOTEBOOKS_DIR, filename)
            shutil.move(src_path, dest_path)
            print(f"Moved Notebook: {filename} -> notebooks/")
            files_processed.append(dest_path)
            
    # Clean up temp folder
    shutil.rmtree(OUTPUT_TEMP_DIR)
    print("Cleaned up temporary download directory.")
    
    return files_processed

def stage_in_git(files):
    """Stage the files in Git to prepare for commit."""
    if not files:
        print("No new files found to stage.")
        return
        
    print("\n--- Staging Files in Git ---")
    
    # Add files to git
    for file_path in files:
        success, stdout = run_command(["git", "add", file_path], cwd=WORKSPACE_DIR)
        if success:
            print(f"Staged in Git: {os.path.basename(file_path)}")
            
    # Commit changes locally
    commit_msg = "Sync trained model and notebook from Kaggle cloud build"
    success, stdout = run_command(["git", "commit", "-m", commit_msg], cwd=WORKSPACE_DIR)
    if success:
        print(f"\nSuccess! Locally committed updates with message: '{commit_msg}'")
        print("\n>>> The codebase is now ready. Run 'git push' when you want to upload to GitHub.")
    else:
        print("\nStaged files, but skipped commit (already up to date or no changes detected).")

if __name__ == "__main__":
    check_kaggle_cli()
    downloaded_files = sync_from_kaggle()
    stage_in_git(downloaded_files)

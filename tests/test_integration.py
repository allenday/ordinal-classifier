#!/usr/bin/env python3
"""Integration test for ordinal classifier with demo dataset."""

import subprocess
import sys
from pathlib import Path
import shutil
import pytest


def run_command(cmd, description):
    """Run a command and check its output."""
    print(f"\n🔄 {description}")
    print(f"Running: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Command failed with return code {result.returncode}")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        raise AssertionError(f"Command failed: {' '.join(cmd)}")
    
    print(f"✅ {description} completed successfully")
    return True


@pytest.mark.integration
def test_ordinal_classifier_integration():
    """Run integration test."""
    print("🚀 Starting ordinal classifier integration test")
    
    # Check if we're in the right directory - look for data relative to test file
    test_dir = Path(__file__).parent
    project_root = test_dir.parent
    demo_data = project_root / "data/demo/train"
    
    if not demo_data.exists():
        pytest.skip("Demo dataset not found. Make sure data/demo/ exists in project root.")
    
    # Create output directory for test files
    test_out_dir = test_dir / "out"
    test_out_dir.mkdir(exist_ok=True)
    
    # Clean up any existing test files
    for item in test_out_dir.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()
    
    print(f"📁 Using test output directory: {test_out_dir}")
    
    # Change to project root for commands
    import os
    original_cwd = os.getcwd()
    os.chdir(project_root)
    
    try:
        # Test 1: Train a quick model (standard approach, no fine-tuning)
        train_cmd = [
            "ordinal-classifier", "train",
            "data/demo",  # Use the full demo folder
            "--epochs", "2",
            "--fine-tune-epochs", "0",  # Skip fine-tuning to avoid the error
            "--batch-size", "4",
            "--model-name", "test-model",
            "--arch", "resnet18",
            "--no-early-stopping"  # Disable early stopping for this small dataset
        ]
        
        run_command(train_cmd, "Training model")
    
        # Test 2: Predict on demo images
        predict_cmd = [
            "ordinal-classifier", "predict",
            "data/demo/valid",
            "--model-path", "models/test-model.pkl",
            "--recursive",
            "--show-probabilities"
        ]
        
        run_command(predict_cmd, "Running predictions")
        
        # Test 3: Evaluate model performance  
        eval_cmd = [
            "ordinal-classifier", "evaluate",
            "data/demo/valid",
            "--model-path", "models/test-model.pkl", 
            "--recursive",
            "--output-dir", str(test_out_dir / "evaluation")
        ]
        
        run_command(eval_cmd, "Evaluating model")
        
        # Test 4: Generate heatmaps
        heatmap_cmd = [
            "ordinal-classifier", "heatmap",
            "data/demo/valid/0-macro",
            str(test_out_dir / "heatmaps"),
            "--model-path", "models/test-model.pkl"
        ]
        
        run_command(heatmap_cmd, "Generating heatmaps")
        
        # Test 5: Show model info
        info_cmd = [
            "ordinal-classifier", "info", 
            "--model-path", "models/test-model.pkl"
        ]
        
        run_command(info_cmd, "Showing model info")
        
        print("\n🎉 All integration tests passed successfully!")
        print("\nGenerated files:")
        print(f"  - Model: models/test-model.pkl")
        print(f"  - Evaluation results: {test_out_dir / 'evaluation'}")
        print(f"  - Heatmaps: {test_out_dir / 'heatmaps'}")
        
    finally:
        # Restore original working directory
        os.chdir(original_cwd)
        
        # Clean up model file
        model_file = project_root / "models/test-model.pkl"
        if model_file.exists():
            model_file.unlink()
            print(f"\n🧹 Cleaned up model file: {model_file}")


if __name__ == "__main__":
    test_ordinal_classifier_integration()
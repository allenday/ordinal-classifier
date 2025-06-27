# Ordinal Classifier

FastAI-based tools for classifying images into ordered categories with ordinal-aware training and evaluation.

## Overview

This package provides a complete toolkit for ordinal image classification, and includes a demo dataset for shot type classification in filmmaking. It supports both standard classification and ordinal-aware training that reduces penalties for adjacent classification errors.

## Features

- **Ordinal-aware training**: Reduces penalties for adjacent misclassifications
- **Multiple architectures**: Support for ResNet and EfficientNet models
- **Comprehensive evaluation**: Confusion matrices, metrics, and visualizations
- **Heatmap generation**: Activation heatmaps for model interpretability
- **Flexible CLI**: Easy-to-use command-line interface

## Installation

```bash
# Clone the repository
git clone https://github.com/allenday/ordinal-classifier.git
cd ordinal-classifier

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install the package
pip install -e .
```

## Quick Start

### Dataset Structure

Organize your data with numeric prefixes indicating ordinal order:

```
data/
├── train/
│   ├── 0-macro/
│   ├── 1-close-up/
│   ├── 2-medium-close-up/
│   ├── 3-medium/
│   ├── 4-long/
│   └── 5-wide/
└── valid/
    ├── 0-macro/
    ├── 1-close-up/
    ├── 2-medium-close-up/
    ├── 3-medium/
    ├── 4-long/
    └── 5-wide/
```

### Basic Usage

```bash
# Train a model with ordinal-aware loss
ordinal-classifier train data/ --ordinal --epochs 10

# Make predictions on new images
ordinal-classifier predict images/ --recursive --show-probabilities

# Evaluate model performance
ordinal-classifier evaluate test_data/ --recursive

# Generate activation heatmaps
ordinal-classifier heatmap images/ heatmaps/
```

## Commands

### Training

```bash
# Standard training
ordinal-classifier train data/ --epochs 10 --arch resnet50

# Ordinal training with label smoothing
ordinal-classifier train data/ --ordinal --smoothing 0.1 --epochs 10

# Custom image size and batch size
ordinal-classifier train data/ --image-size 224,224 --batch-size 32
```

### Prediction

```bash
# Single image
ordinal-classifier predict image.jpg

# Directory of images
ordinal-classifier predict images/ --recursive

# Show prediction probabilities
ordinal-classifier predict images/ --show-probabilities
```

### Evaluation

```bash
# Evaluate on test set
ordinal-classifier evaluate test_data/ --recursive

# Save evaluation results
ordinal-classifier evaluate test_data/ --output-dir results/
```

### Heatmaps

```bash
# Generate heatmaps for interpretability
ordinal-classifier heatmap images/ heatmaps/

# Adjust blending alpha
ordinal-classifier heatmap images/ heatmaps/ --alpha 0.7
```

## Demo Dataset

A small demo dataset is included in `data/demo/` with shot type examples. Test the installation:

```bash
# Run integration test
python test_integration.py

# Quick training test
ordinal-classifier train data/demo/ --epochs 2 --arch resnet18 --no-early-stopping
```

## Model Architectures

Supported architectures:
- `resnet18`, `resnet34`, `resnet50`, `resnet101`, `resnet152`
- `efficientnet_b0`, `efficientnet_b3`, `efficientnet_b5`

## Ordinal Training

The ordinal training approach:
1. Uses numeric prefixes (0-, 1-, 2-, etc.) to define order
2. Applies label smoothing that considers adjacent categories
3. Reduces penalty for "near miss" predictions
4. Particularly effective for naturally ordered categories

## License

MIT License - see LICENSE file for details.

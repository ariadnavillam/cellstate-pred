# Introduction

**cellstate-pred** is a Python package designed for segmenting cells from drug assays and performing inference with pretrained deep learning models.

## Overview

This package provides tools for:

- **Cell Segmentation**: Automated segmentation of cells from microscopy images using Otsu thresholding and watershed segmentation
- **Feature Extraction**: Extract features from segmented cells using state-of-the-art vision models
- **Inference Pipeline**: Run inference on multiple images efficiently with batch processing

## Key Features

### OperaImage Class
The core class for handling microscopy images from Opera microscopes, providing methods for:
- Loading images from files
- Automatic cell segmentation
- Visualization of segmentation results
- Individual cell extraction

### Deep Learning Integration
Built-in integration with Hugging Face transformers for:
- Feature extraction using pretrained vision models
- Efficient batch processing
- GPU acceleration support

### Flexible Pipeline
- Support for custom image transforms
- Configurable batch sizes and processing parameters
- Automatic handling of edge cases (empty images, no cells found)

## Use Cases

This package is particularly useful for:
- High-throughput screening of drug compounds
- Cell morphology analysis
- Phenotypic profiling
- Computer vision applications in biology
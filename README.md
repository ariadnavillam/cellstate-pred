# cellstate-pred

A Python package for cell segmentation from drug assays and inference with pretrained deep learning models.

## Overview

`cellstate-pred` is designed to analyze microscopy images from drug screening assays by automatically segmenting individual cells and extracting morphological features using pretrained deep learning models. The package provides tools for:

- **Cell Segmentation**: Automated segmentation of cells from microscopy images using advanced image processing techniques
- **Feature Extraction**: Extraction of morphological features from segmented cells using pretrained deep learning models
- **Drug Assay Analysis**: Specialized workflows for analyzing cellular responses in drug screening experiments

## Key Features

- Support for Opera microscope image formats
- Otsu thresholding and watershed segmentation for robust cell detection
- Integration with Hugging Face transformers for feature extraction
- Batch processing capabilities for high-throughput analysis
- Flexible data loading with PyTorch DataLoader integration

## Installation

### Requirements

- Python 3.10+
- Dependencies as specified in `pyproject.toml`

### Install from source

```bash
git clone https://github.com/rendeirolab/cellstate-pred.git
cd cellstate-pred
pip install -e .
```

## Quick Start

```python
from cellstate_pred.image_utils import OperaImage
from cellstate_pred._inference import inference
from pathlib import Path

# Load and segment cells from a microscopy image
image = OperaImage()
image.load_image("path/to/your/image.tiff")
image.segment_cells(min_distance=7)

# Extract features using a pretrained model
inference(
    batch_size=32,
    image=image,
    model_name="microsoft/resnet-50",
    output_dir=Path("./features"),
    num_workers=4
)
```

## Documentation

For detailed usage examples and API documentation, see the notebooks in the `scripts/` directory.

## Contributing

This repository was created with a [cookiecutter template](https://github.com/rendeirolab/_project_template), version 0.4.1dev.

## License

[Add your license information here]

# Installation

## Requirements

- Python 3.8+
- PyTorch
- scikit-image
- Hugging Face transformers
- PIL/Pillow
- NumPy
- Pandas

## Install from Source

To install cellstate-pred from source:

```bash
git clone https://github.com/rendeirolab/cellstate-pred.git
cd cellstate-pred
uv sync
```

## Development Installation

For development, install with the development dependencies:

```bash
git clone https://github.com/rendeirolab/cellstate-pred.git
cd cellstate-pred
uv sync --all-extras
```

## Dependencies

The package uses modern Python dependency management with `uv` and includes:

### Core Dependencies
- `scikit-image`: Image processing and segmentation
- `transformers`: Deep learning model integration  
- `torch`: PyTorch for GPU acceleration
- `pillow`: Image handling
- `numpy`: Numerical computations
- `pandas`: Data manipulation
- `scipy`: Scientific computing
- `matplotlib`: Plotting and visualization

### Optional Dependencies
- `torchvision`: Computer vision transforms
- `tqdm`: Progress bars for batch processing
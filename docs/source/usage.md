# Usage Guide

## Basic Usage

### Loading and Processing Images

```python
from cellstate_pred.image_utils import OperaImage
from cellstate_pred.feature_extraction import extract_features

# Load an image
img = OperaImage()
img.load_image("path/to/your/image.tiff")

# Or create with image data directly
from skimage.data import human_mitosis
img = OperaImage(image=human_mitosis())

# Segment cells automatically
img.segment_cells()

# Visualize results
img.plot_segmentation()
img.plot_cells(n=10)  # Show first 10 cells
```

### Feature Extraction

```python
# Extract features using a pretrained model
features_df = extract_features(
    img, 
    model_name="facebook/dinov2-vits16", 
    batch_size=32
)

print(features_df.shape)  # (n_cells, n_features)
print(features_df.head())
```

### Advanced Usage

#### Custom Transforms

```python
from torchvision import transforms

# Define custom image transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

# Use with feature extraction
features_df = extract_features(
    img,
    model_name="microsoft/resnet-50",
    transform=transform,
    batch_size=16
)
```

#### Processing Multiple Images

```python
import os
from pathlib import Path

# Process all images in a directory
image_dir = Path("data/images/")
results = []

for image_path in image_dir.glob("*.tiff"):
    img = OperaImage()
    img.load_image(image_path)
    
    features_df = extract_features(img, model_name="vinid/plip")
    features_df['image_name'] = image_path.stem
    
    results.append(features_df)

# Combine all results
import pandas as pd
all_features = pd.concat(results, ignore_index=True)
```

#### Cell Segmentation Parameters

```python
# Custom segmentation parameters
img.segment_cells(min_distance=10)  # Adjust minimum distance between cell centers

# Check segmentation results
print(f"Found {len(img.cell_images)} cells")
print(f"Binary mask shape: {img.cells.shape}")
print(f"Labeled regions: {img.segmented_cells.max()} unique cells")
```

## Working with Different Models

### Vision Transformers

```python
# DINOv2 models
features_df = extract_features(img, model_name="facebook/dinov2-vits16")
features_df = extract_features(img, model_name="facebook/dinov2-vitb14")

# CLIP-based models  
features_df = extract_features(img, model_name="vinid/plip")
```

### ResNet Models

```python
# ResNet architectures
features_df = extract_features(img, model_name="microsoft/resnet-50")
features_df = extract_features(img, model_name="microsoft/resnet-101")
```

## Output and Saving

The `extract_features` function automatically saves results to disk and returns a pandas DataFrame:

```python
features_df = extract_features(
    img,
    model_name="facebook/dinov2-vits16",
    output_dir="./results",
    batch_size=32
)

# Features are saved as .npz files in the output directory
# DataFrame has cell IDs as index and features as columns
print(features_df.index.name)  # Cell IDs
print(features_df.columns)     # Feature dimensions
```

## Error Handling

The package gracefully handles edge cases:

```python
# Empty or problematic images
empty_img = OperaImage(image=np.zeros((100, 100)))
features_df = extract_features(empty_img)
print(features_df.empty)  # True - no cells found

# Images with no segmentable cells return empty DataFrames
# No crashes or errors, just empty results
```
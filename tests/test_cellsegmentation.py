import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
from skimage.data import human_mitosis

from cellstate_pred.image_utils import OperaImage


@pytest.fixture(scope="session")
def reference_mask():
    """Load the reference mask once for all tests."""
    reference_mask_path = Path(__file__).parent / "cells_mask.txt"
    return np.loadtxt(reference_mask_path)


@pytest.fixture(scope="module")
def sample_image():
    """Load the sample image once per test module."""
    return human_mitosis()


@pytest.fixture
def opera_image(sample_image):
    """Create a fresh OperaImage for each test."""
    return OperaImage(image=sample_image)


@pytest.fixture
def segmented_opera_image(opera_image):
    """Create a segmented OperaImage for tests that need it."""
    opera_image.segment_cells()
    return opera_image


def test_image_importer(opera_image, sample_image):
    """Test that the image importer works correctly with the human_mitosis sample image."""

    # Verify that the image was loaded correctly
    assert opera_image.image is not None
    assert np.array_equal(opera_image.image, sample_image)
    assert opera_image.image.shape == sample_image.shape
    assert opera_image.image.dtype == sample_image.dtype


@patch("cellstate_pred.image_utils.ski.io.imread")
def test_image_load_from_file(mock_imread, sample_image):
    """Test loading an image from a file path."""

    # Set up mock to return the sample image
    mock_imread.return_value = sample_image

    # Create OperaImage and load from file
    img = OperaImage()
    img.load_image("/test/path.tiff")

    # Verify the file loading worked
    assert img.filepath == "/test/path.tiff"
    assert img.image is not None
    assert np.array_equal(img.image, sample_image)
    mock_imread.assert_called_once_with("/test/path.tiff", plugin="tifffile")


def test_cell_segmentation_accuracy(segmented_opera_image, reference_mask):
    """Test that cell segmentation produces results similar to the reference mask."""

    img = segmented_opera_image

    # Check that segmentation was performed
    assert img.cells is not None
    assert img.segmented_cells is not None
    assert img.cell_images is not None

    # Check that the mask has the right shape
    assert img.cells.shape == reference_mask.shape

    # Convert boolean mask to float for comparison
    segmented_mask = img.cells.astype(float)

    # Calculate similarity metrics
    # Jaccard index (Intersection over Union)
    intersection = np.logical_and(segmented_mask, reference_mask).sum()
    union = np.logical_or(segmented_mask, reference_mask).sum()
    jaccard_index = intersection / union if union > 0 else 0

    # Pixel accuracy
    pixel_accuracy = np.mean(segmented_mask == reference_mask)

    # Set reasonable thresholds for similarity
    JACCARD_THRESHOLD = 0.7  # 70% overlap
    PIXEL_ACCURACY_THRESHOLD = 0.85  # 85% pixel accuracy

    # Assert that the segmentation is sufficiently similar to the reference
    assert jaccard_index >= JACCARD_THRESHOLD, (
        f"Jaccard index {jaccard_index:.3f} below threshold {JACCARD_THRESHOLD}"
    )
    assert pixel_accuracy >= PIXEL_ACCURACY_THRESHOLD, (
        f"Pixel accuracy {pixel_accuracy:.3f} below threshold {PIXEL_ACCURACY_THRESHOLD}"
    )

    # Additional checks
    assert len(img.cell_images) > 0, "No cells were segmented"
    assert img.segmented_cells.max() > 0, "No labeled regions found"


def test_segmentation_consistency(sample_image):
    """Test that segmentation produces consistent results across multiple runs."""

    # Run segmentation multiple times
    results = []
    for _ in range(3):
        img = OperaImage(image=sample_image)
        img.segment_cells()
        assert img.cells is not None  # Ensure segmentation worked
        results.append(img.cells.copy())

    # Check that results are identical (should be deterministic)
    for i in range(1, len(results)):
        assert np.array_equal(results[0], results[i]), (
            "Segmentation is not deterministic"
        )


def test_segmentation_attributes(segmented_opera_image):
    """Test that all expected attributes are set after segmentation."""

    img = segmented_opera_image

    # Check that all attributes are properly set
    assert hasattr(img, "cells")
    assert hasattr(img, "segmented_cells")
    assert hasattr(img, "cell_images")

    assert img.cells is not None
    assert img.segmented_cells is not None
    assert img.cell_images is not None

    # Check data types
    assert isinstance(img.cells, np.ndarray)
    assert isinstance(img.segmented_cells, np.ndarray)
    assert isinstance(img.cell_images, list)

    # Check that cell_images contains numpy arrays
    if len(img.cell_images) > 0:
        assert all(isinstance(cell, np.ndarray) for cell in img.cell_images)

import pytest
import numpy as np
import pandas as pd
from PIL import Image
from unittest.mock import patch, MagicMock
from skimage.data import human_mitosis

from cellstate_pred.image_utils import OperaImage
from cellstate_pred._inference import CellsDataset
from cellstate_pred.feature_extraction import extract_features


def test_dataset():
    """Test CellsDataset functionality."""
    
    # Create OperaImage with sample data
    opera_image = OperaImage(image=human_mitosis())
    
    # Create dataset
    dataset = CellsDataset(opera_image)
    
    opera_image.segment_cells()  # Ensure cells are segmented for the test

    # Verify length matches number of segmented cells
    assert opera_image.cell_images is not None
    assert len(dataset) == len(opera_image.cell_images)
    
    # Test __getitem__ for first cell
    item = dataset[0]
    assert isinstance(item, dict)
    assert 'image' in item and 'cell_id' in item
    assert isinstance(item['image'], Image.Image)
    assert item['cell_id'] == 0
    
    # Test __getitem__ for last cell
    last_index = len(dataset) - 1
    item = dataset[last_index]
    assert item['cell_id'] == last_index
    
    # Test out-of-bounds access raises IndexError
    with pytest.raises(IndexError):
        _ = dataset[len(dataset)]

def test_dataset_no_cells():
    """Test CellsDataset with an image that has no cells."""
    
    # Create OperaImage with empty/no cells
    opera_image = OperaImage(image=np.zeros((50, 50)))  # Small empty image
    
    # Create dataset
    from cellstate_pred._inference import CellsDataset
    dataset = CellsDataset(opera_image)
    
    # Should have zero length
    assert len(dataset) == 0
    
    # Accessing any index should raise IndexError
    with pytest.raises(IndexError):
        _ = dataset[0]

def test_extract_features_returns_dataframe():
    """Test that extract_features returns a pandas DataFrame."""
    
    # Create OperaImage with sample data
    opera_image = OperaImage(image=human_mitosis())
    
    # Mock the inference function to avoid needing actual models
    mock_features = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])  # 3 cells, 3 features each
    mock_cell_ids = np.array([0, 1, 2])
    
    with patch('cellstate_pred.feature_extraction.inference') as mock_inference:
        # Configure mock to return our test features and cell IDs
        mock_inference.return_value = (mock_features, mock_cell_ids)
        
        # Call extract_features
        result = extract_features(opera_image, model_name="test-model", batch_size=2)
        
        # Verify it returns a pandas DataFrame
        import pandas as pd
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (3, 3)  # 3 cells, 3 features each
        
        # Verify the index matches cell IDs
        np.testing.assert_array_equal(result.index.values, mock_cell_ids)
        
        # Verify the content matches our mock data
        np.testing.assert_array_equal(result.values, mock_features)


def test_extract_features_empty_image():
    """Test that extract_features handles images with no cells."""
    
    # Create OperaImage with empty/no cells
    opera_image = OperaImage(image=np.zeros((50, 50)))  # Small empty image
    
    # Mock the inference function to return empty arrays
    with patch('cellstate_pred.feature_extraction.inference') as mock_inference:
        mock_inference.return_value = (np.array([]), np.array([]))
        
        # Call extract_features
        result = extract_features(opera_image)
        
        # Should return empty DataFrame when no cells are found
        assert isinstance(result, pd.DataFrame)
        assert result.empty

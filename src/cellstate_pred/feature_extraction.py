import numpy as np

from cellstate_pred.image_utils import OperaImage
from cellstate_pred._inference import inference
import pandas as pd
from pathlib import Path


def extract_features(
    opera_image: OperaImage, 
    model_name: str = "vinid/plip",
    batch_size: int = 32,  
    output_dir: Path | str = "./features",
    transform=None
) -> pd.DataFrame:
    """Extract features from cells in an OperaImage object.
    
    Parameters
    ----------
    opera_image : OperaImage
        An OperaImage object. If not already segmented, cells will be segmented automatically.
    model_name : str, optional
        Name of the transformers model to use for feature extraction.
        Default is "vinid/plip".
    batch_size : int, optional
        Batch size for processing. Default is 32.
    output_dir : Path | str, optional
        Directory to save the output features. Default is "./features".
    transform : callable, optional
        Optional transform to apply to cell images.
        
    Returns
    -------
    pd.DataFrame
        A DataFrame where rows are cell IDs (index) and columns are features.
        Shape: (n_cells, n_features)
    """
    # Ensure output_dir is a Path object and create directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    features_matrix, cell_ids = inference(batch_size=batch_size, 
                        image=opera_image, 
                        model_name=model_name, 
                        output_dir= output_dir,
                        transform=transform)

    # Mean pooling
    if features_matrix.ndim == 3:
        features_matrix = np.mean(features_matrix, axis=1)
    else:
        features_matrix = features_matrix

    df = pd.DataFrame(features_matrix, index=cell_ids)
    return df

if __name__ == "__main__":
    pass 
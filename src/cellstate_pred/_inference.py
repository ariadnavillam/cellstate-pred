from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import torch
from .image_utils import OperaImage, normalize_cell
from torch.utils.data import Dataset, DataLoader
from transformers import pipeline
from datasets import Dataset as HFDataset

from torchvision.transforms import v2



def make_transform(resize_size: int = 256):
    to_tensor = v2.ToImage()
    resize = v2.Resize((resize_size, resize_size), antialias=True)
    to_float = v2.ToDtype(torch.float32, scale=True)
    normalize = v2.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    return v2.Compose([to_tensor, resize, to_float, normalize])

class CellsDataset(Dataset):
    def __init__(self, image: OperaImage, transform=None):
        """Initialize the CellsDataset.

        Parameters
        ----------
        image : OperaImage
            An OperaImage object. If not already segmented, segment_cells() 
            will be called automatically.
        transform : callable, optional
            Optional transform to be applied to the cell images.
        """
        self.image =image
        self.transform = transform

        # Ensure cells have been segmented
        if self.image.cell_images is None:
            self.image.segment_cells()

        # Store cell images and create cell IDs
        self.cell_images = self.image.cell_images
        self.cell_ids = list(range(len(self.cell_images)))

    def __len__(self) -> int:
        """Return the number of cells in the dataset."""
        return len(self.cell_images)

    def __getitem__(self, idx) -> dict:
        """Get a cell image and its ID.

        Parameters
        ----------
        idx : int
            Index of the cell to retrieve.

        Returns
        -------
        dict
            A dictionary containing:
            - 'image': The normalized cell image as a PIL Image
            - 'cell_id': The cell ID (integer)
        """
        if idx >= len(self.cell_images):
            raise IndexError(
                f"Index {idx} out of range for dataset with {len(self.cell_images)} cells"
            )

        # Get the cell image and normalize it
        cell_image = self.cell_images[idx]
        normalized_cell = normalize_cell(cell_image)
        # Ensure the image is in the right format (0-255 range for PIL)
        pil_image = Image.fromarray((normalized_cell * 255).astype(np.uint8))

        # Convert to PIL Image if transforms are provided (common for vision models)
        if self.transform:
            # Convert to PIL Image for transformations

            normalized_cell = self.transform(pil_image)
        else:
            # If no transform is provided, ensure the image is a numpy array
            normalized_cell = pil_image

        return {"image": normalized_cell, "cell_id": self.cell_ids[idx]}


def inference(
    batch_size: int, image: OperaImage, model_name: str, output_dir: Path | str, num_workers=10, transform=None
) -> tuple[np.ndarray, np.ndarray]:
    """Run inference on segmented cells using a transformers model.

    Parameters
    ----------
    batch_size : int
        Batch size for processing.
    image : OperaImage
        An OperaImage object. If not already segmented, cells will be 
        segmented automatically.
    model_name : str
        Name of the transformers model to use for feature extraction.
    output_dir : Path | str
        Directory to save the output features.
    num_workers : int, optional
        Number of workers for the dataloader. Default is 10.
    transform : callable, optional
        Optional transform to apply to cell images.
        
    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple containing:
        - features_array: Feature matrix of shape (n_cells, n_features)
        - cell_ids_array: Array of cell IDs corresponding to each row
    """
    
    cells_dataset = CellsDataset(image, transform=transform)

    records = [cells_dataset[i] for i in range(len(cells_dataset))]

    dataset = HFDataset.from_list(records)

    if len(dataset) == 0:
        return np.array([]), np.array([])

    device = 0 if torch.cuda.is_available() else -1

    feature_extractor = pipeline(
                task="image-feature-extraction",
                model=model_name,
                device=device,       # GPU if available
                batch_size=batch_size,  # adjust batch size
            )
    
    def process_batch(batch):
        # batch["image"] is a list of PIL Images
        features = feature_extractor(batch["image"])  # list of (seq_len, hidden_dim) arrays

        batch["features"] = features  # assign the pooled features
        return batch
    
    output = dataset.map(process_batch, batched=True, batch_size=batch_size)

    features_list = output["features"]    # list of embeddings
    cell_ids_list = output["cell_id"]

    features_array = np.vstack(features_list)  # (N, seq_len,D)
    cell_ids_array = np.array(cell_ids_list)

    filename = Path(image.filepath).stem if image.filepath else "unknown"
    # Ensure output_dir is a Path object
    output_dir = Path(output_dir)
    # The output filename should reflect the new model and parameters
    output_filename = f"{filename}_{model_name.replace('/', '_')}_nucleus_fts.npz"

    np.savez_compressed(
        output_dir / output_filename,
        features=features_array,
        cell_id=cell_ids_array,
    )
    return features_array, cell_ids_array

# dataloader = DataLoader(
#         dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
#         collate_fn=custom_collate_fn
#     )
#     device = 0 if torch.cuda.is_available() else -1

#     feature_extractor = pipeline(
#         model=model_name,
#         task="image-feature-extraction",
        
#         device=device,
#         truncation=True
#     )

#     features_list = []
#     cell_ids_list = []

#     for batch in tqdm(dataloader, total=len(dataloader)):
#         images = batch["image"]      # list of PIL Images or tensors
#         cell_ids = batch["cell_id"]

#         # Run the whole batch at once
#         batch_features = feature_extractor(images)  # list of arrays

#         features_list.extend(batch_features)
#         cell_ids_list.extend(cell_ids)

#     features_array = np.vstack(features_list)  # shape: (N, D)
#     cell_ids_array = np.array(cell_ids_list)

    # features_list = []
    # cell_ids_list = []
    # start_time = pd.Timestamp.now()
    # print(f"start time: {start_time}")
    
    # for batch in tqdm(dataloader):
    #     images = batch["image"]
    #     cell_ids = batch["cell_id"]

    #     # Convert tensor cell_ids to list if needed
    #     if hasattr(cell_ids, 'tolist'):
    #         cell_ids = cell_ids.tolist()
        
    #     # Process each image in the batch individually (like in your example)
    #     for i in range(len(images)):
    #         # Extract features for single image
    #         single_image = images[i]
    #         single_cell_id = cell_ids[i]
            
    #         # Pipeline expects single image or list with one image
    #         features = feature_extractor(single_image)
            
    #         # Extract the pooled features (first element)
    #         if isinstance(features, list) and len(features) > 0:
    #             feature_vector = features[0]
    #         else:
    #             feature_vector = features
            
    #         features_list.append(feature_vector)
    #         cell_ids_list.append(single_cell_id)

    # end_time = pd.Timestamp.now()
    # print(f"time to process {len(cell_ids_list)} cells: {end_time - start_time}")

    # features_array = np.array(features_list)
    # cell_ids_array = np.array(cell_ids_list)

if __name__ == "__main__":
    pass
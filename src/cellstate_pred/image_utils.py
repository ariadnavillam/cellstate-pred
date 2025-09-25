from pathlib import Path
import skimage as ski
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from skimage import color, feature, measure, segmentation


def otsu_threshold(image: np.ndarray, bins: int=256) -> int | None:
    """Compute Otsu's threshold for an image.

    Parameters
    ----------
    image : ndarray
        Input image (grayscale).
    bins : int, optional
        Number of bins to use for the histogram. Default is 256.

    Returns
    -------
    threshold : int or None
        Computed Otsu's threshold value as bin index, or None if no valid 
        threshold can be determined.
    """

    hist, bin_edges = np.histogram(image, range=(0, bins), bins=bins - 1)
    hist = hist / hist.sum()
    bin_mids = bin_edges[:-1] + bin_edges[1:] / 2
    w_zero = np.cumsum(hist)
    w_one = np.cumsum(hist[::-1])[::-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        mu_zero = np.cumsum(hist * bin_mids) / w_zero
        mu_one = (np.cumsum((hist * bin_mids)[::-1]) / w_one[::-1])[::-1]
        inter_class_var = w_zero[:-1] * w_one[1:] * (mu_zero[:-1] - mu_one[1:]) ** 2

    if np.all(np.isnan(inter_class_var)):
        return
    elif len(np.unique(inter_class_var))==1:
        return
    else:
        return int(np.nanargmax(inter_class_var))
    


def normalize_cell(cell: np.ndarray) -> np.ndarray:
    """Normalize a single cell image to the range [0, 1].

    Parameters
    ----------
    cell : ndarray
        Input cell image.

    Returns
    -------
    cell_norm : ndarray
        Normalized cell image.
    """
    mask = cell > 0

    cell_norm = np.zeros_like(cell, dtype=float)
    if np.any(mask):
        min_val = cell[mask].min()
        max_val = cell[mask].max()
        cell_norm[mask] = (
            (cell[mask] - min_val) / (max_val - min_val) if max_val > min_val else 0
        )

    return cell_norm


class OperaImage:
    """A class to represent and process an image from an Opera microscope.

    Attributes
    ----------
    filepath : str
        Path to the image file.
    image : ndarray
        The image data.
    attrs : dict
        A dictionary of image attributes.
    cells : ndarray
        A boolean mask of the cells in the image.
    segmented_cells : ndarray
        A labeled array of the segmented cells.
    cell_images : list
        A list of ndarrays, where each array is the image of a single cell.
    """

    def __init__(self, filepath: str | Path | None =None, image=None, attrs=None):
        """
        Parameters
        ----------
        filepath : str, optional
            Path to the image file.
        image : ndarray, optional
            The image data.
        attrs : dict, optional
            A dictionary of image attributes.
        """
        self.filepath = filepath
        self.image = image
        self.attrs = attrs if attrs else {}
        self.cells = None
        self.segmented_cells = None
        self.cell_images = None

    def load_image(self, filepath):
        """Load an image from a file.

        Parameters
        ----------
        filepath : str
            Path to the image file.
        """
        self.filepath = filepath
        self.image = ski.io.imread(filepath, plugin="tifffile")

    def show_image(self):
        """Display the image."""
        if self.image is None:
            raise ValueError("No image loaded. Please load an image first.")
        plt.imshow(self.image, cmap="gray")
        plt.axis("off")
        plt.show()

    def segment_cells(self, min_distance=7):
        """Segment the cells in the image.

        This method uses Otsu's thresholding to create a binary mask of the
        cells, followed by a watershed segmentation to separate overlapping
        nuclei. If no cells can be segmented, all attributes are set to 
        empty/zero arrays.

        Parameters
        ----------
        min_distance : int, optional
            The minimum distance between peaks in the distance transform.
            Default is 7.
        
        Notes
        -----
        After calling this method, the following attributes are set:
        - cells : boolean mask of detected cells
        - segmented_cells : labeled array of individual cells
        - cell_images : list of individual cell image arrays
        """
        threshold = otsu_threshold(self.image)
        
        # Handle case where no valid threshold is found
        if threshold is None:
            # Set empty results when no cells can be segmented
            self.cells = np.zeros_like(self.image, dtype=bool)
            self.segmented_cells = np.zeros_like(self.image, dtype=int)
            self.cell_images = []
            return
        
        self.cells = self.image > threshold
        
        # If no cells were found after thresholding
        if not np.any(self.cells):
            self.segmented_cells = np.zeros_like(self.image, dtype=int)
            self.cell_images = []
            return
        
        distance = ndi.distance_transform_edt(self.cells)
        local_max_coords = feature.peak_local_max(distance, min_distance=min_distance)
        
        # If no local maxima are found
        if len(local_max_coords) == 0:
            self.segmented_cells = np.zeros_like(self.image, dtype=int)
            self.cell_images = []
            return
        
        local_max_mask = np.zeros(distance.shape, dtype=bool)
        local_max_mask[tuple(local_max_coords.T)] = True
        markers = measure.label(local_max_mask)
        self.segmented_cells = segmentation.watershed(
            -distance, markers, mask=self.cells
        )
        props = measure.regionprops(self.segmented_cells, intensity_image=self.image)
        self.cell_images = [props[i].image_intensity for i in range(len(props))]

    def plot_segmentation(self):
        """Plot the segmentation results.

        This method displays the binary mask of the cells and the final
        segmented image side by side.
        
        Raises
        ------
        ValueError
            If cells have not been segmented yet.
        """
        if self.segmented_cells is None:
            raise ValueError(
                "Cells have not been segmented yet. Call segment_cells() first."
            )

        fig, ax = plt.subplots(ncols=2, figsize=(10, 5))
        ax[0].imshow(self.cells, cmap="gray")
        ax[0].set_title("Overlapping nuclei")
        ax[0].axis("off")
        ax[1].imshow(color.label2rgb(self.segmented_cells, bg_label=0))
        ax[1].set_title("Segmented nuclei")
        ax[1].axis("off")
        plt.show()

    def plot_cells(self, n=5):
        """Plot a sample of the segmented cells.

        Parameters
        ----------
        n : int, optional
            The number of cells to plot. Default is 5.
            
        Raises
        ------
        ValueError
            If cells have not been segmented yet.
        """
        if self.cell_images is None:
            raise ValueError(
                "Cells have not been segmented yet. Call segment_cells() first."
            )

        fig, ax = plt.subplots(1, n, figsize=(15, 5))
        for i in range(n):
            cell = self.cell_images[i]
            ax[i].imshow(normalize_cell(cell), cmap="gray")
            ax[i].axis("off")
        plt.show()

if __name__ == "__main__":
    pass
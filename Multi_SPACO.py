from SpaCoObject import SPACO
from M_Z_csv import SpectrumData
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
from stardist.models import StarDist2D
from csbdeep.utils import normalize
import tensorflow as tf
import tifffile
import scipy.sparse as sp
import scanpy as sc
from scipy import ndimage
from skimage.measure import label as sk_label

class MultiModalRegistration(SpectrumData):
    def __init__(self, *args, use_gpu=True, visium=False, maldi=False, **kwargs):
        super().__init__(*args, **kwargs)
        # Additional initialization for multimodal registration can go here
        # Configure GPU settings
        self.use_gpu = use_gpu
        if self.use_gpu:
            self._configure_gpu()

        # precompute variables
        self.level_data = None

    def _configure_gpu(self):
        """
        Configure TensorFlow to use GPU if available.
        """
        if self.use_gpu:
            # List available GPUs
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                try:
                    # Enable memory growth to avoid allocating all GPU memory at once
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    print(f"GPU(s) found and configured: {len(gpus)} GPU(s)")
                    print(f"GPU details: {gpus}")
                except RuntimeError as e:
                    print(f"GPU configuration error: {e}")
            else:
                print("No GPU found. Running on CPU.")
        else:
            # Disable GPU and run on CPU only
            tf.config.set_visible_devices([], 'GPU')
            print("GPU disabled. Running on CPU.")

    def load_sparse_data(self) -> None:
        """
        Loads Visium spatial transcriptomics data from the specified path.
        """

        if self.maldi: 
            # Load the imzML file using pyimzML
            spectra = [self.parser.getspectrum(idx) for idx in range(len(self.parser.coordinates))]
            
            # grab the m/z and intensity values for each spectrum
            mzs = [s[0] for s in spectra]
            intensities = [s[1] for s in spectra]

            # Get coordinates
            coords = np.array(self.parser.coordinates)
            self.coords = coords[:, :2]  # Keep only x and y
            
            # Concatenate all m/z and intensity arrays to build the COO matrix
            all_mzs = np.concatenate(mzs) # makes all m/z values in one array, same / intsensity
            all_intensities = np.concatenate(intensities)

            # Build row indices: repeat each sample index by its spectrum length
            row_indices = np.repeat(np.arange(len(mzs)), [len(m) for m in mzs])

            # Get unique m/z values and mapping
            unique_mzs, col_indices = np.unique(all_mzs, return_inverse=True)
            
            # Build COO matrix
            coo = sp.coo_matrix((all_intensities, (row_indices, col_indices)),
                                shape=(len(mzs), len(unique_mzs)))
            
            sparse_SF = coo.tocsr()  # Convert to CSR format for efficient row slicing
            return sparse_SF
        
        elif self.visium:
            
            # read the 10x Genomics Visium data using Scanpy
            adata = sc.read_10x_h5(self.data_path)

            # make this a sparse matrix
            sparse_SF = sp.csr_matrix(adata.X)

            return sparse_SF
        
        else: 
            raise ValueError("Please specify either MALDI or Visium data to load.")

        
    def load_visium_data(self, full_data=True, sample_features=True):
        """
        Loads Visium spatial transcriptomics data from the specified path.
        """
        
    def load_normalize_HE(self, he_path: str):
        """
        Loads H&E stained image from the specified path and then normalize.

        Parameters
        ----------
        he_path : str or Path
            Path to the H&E stained image file.

        Returns
        -------
        he_image_norm : ndarray
            Loaded H&E stained image that has been normalized as a NumPy array.
        """

        # reading the H&E image using OpenCV
        he_image = cv2.imread(str(he_path))

        # processing and normalizing the image if needed
        he_image_norm = normalize(he_image, 1, 99.8)

        print("Final shape:", he_image.shape)
        print("Final ndim:", he_image_norm.ndim)
        return he_image_norm
    
    def run_stardist(self, he_image, prob_thresh=0.3, nms_thresh=0.8, n_tiles=(4, 4, 1), show_tile_progress=True):
        """
        Runs StarDist model on the H&E stained image for nuclei segmentation.

        Parameters
        ----------
        he_image : ndarray
            H&E stained image as a NumPy array.

        Returns
        -------
        nuclei_masks : ndarray
            Segmented nuclei masks.
        """
        # Placeholder for StarDist implementation
        # creates a pretrained model
        model = StarDist2D.from_pretrained('2D_versatile_he')
        
        # return the predicted labels and details
        labels, details = model.predict_instances(
            he_image,
            axes='YXC',
            prob_thresh=prob_thresh,        # Much lower to catch more nuclei
            nms_thresh=nms_thresh,
            n_tiles=n_tiles,      # Process in tiles to handle large image (4x4 grid)
            show_tile_progress=show_tile_progress # Show progress bar
        )

        return labels, details
        
    def segmentation_pipeline(self, he_path: str, scale=0.25, save_overlay=False):
        """
        Pipeline for multimodal registration using SPACO and SpectrumData.

        Parameters
        ----------
        scale : float
            Display scale (0.25 = 25% size)
        overlay_alpha : float
            Opacity of mask overlay (0.0 = transparent, 1.0 = opaque)
        """
        # loads the H&E stained image
        print("Loading and normalizing H&E stained image...")
        he_image_norm = self.load_normalize_HE(he_path)

        # runs StarDist model on the H&E stained image for nuclei segmentation
        print("Running StarDist for nuclei segmentation...")
        nuclei_masks, _ = self.run_stardist(he_image=he_image_norm, n_tiles=(8, 8, 1))

        if save_overlay:
            
            print("Creating full-resolution overlay...")
            
            # Convert to uint8
            if he_image_norm.max() <= 1.0:
                he_uint8 = (he_image_norm * 255).astype(np.uint8)
            else:
                he_uint8 = he_image_norm.astype(np.uint8)
            
            # Create boundaries
            print("Generating nuclei boundaries...")
            boundaries = np.zeros(nuclei_masks.shape, dtype=bool)
            #kernel = np.ones((3, 3), np.uint8)

            # Using morphological gradient to find boundaries
            print("finding boundaries of nuclei using morphological gradient...")
            boundaries = ndimage.morphological_gradient(nuclei_masks, size=3) > 0
            
            # Create overlay with red boundaries
            overlay = he_uint8.copy()
            overlay[boundaries] = [255, 0, 0]
            
            # Save with tifffile (preserves quality better)
            output_path = he_path.replace('.tif', '_nuclei_overlay.tif')
            tifffile.imwrite(output_path, overlay, compression='lzma', photometric='rgb')
            print(f"✓ Saved overlay: {output_path}")
            
            # Save masks as 16-bit TIFF
            mask_path = he_path.replace('.tif', '_nuclei_masks.tif')
            tifffile.imwrite(mask_path, nuclei_masks.astype(np.uint16), compression='lzma')
            print(f"✓ Saved masks: {mask_path}")
        print("✓ Segmentation pipeline completed.")
        return nuclei_masks
    
    def data_level_creation(self, coords):
        """
        Parameters
        ----------
        coords : DataFrame with columns containing 'x' and 'y' (case-insensitive),
                or list of (x, y, ...) tuples

        Returns
        -------
        M0 : ndarray
            Binary mask of shape (H0, W0) where M0[i, j] = 1 if (x, y) coordinate exists in the input, else 0.
        min_x : int
            Minimum x-coordinate (used for alignment).
        min_y : int
            Minimum y-coordinate (used for alignment).

        Notes
        -----
        This function takes in a DataFrame or list of coordinates and creates a binary mask of the same shape.
        The mask is created by setting the value of each coordinate in the mask to 1 if the corresponding (x, y) coordinate exists in the input, else 0.
        The minimum x and y coordinates are also returned, which are used for alignment.
        """

        # coerce list of tuples → DataFrame
        if isinstance(coords, list):
            # only for maldi....this needs updated for visium 
            coords = pd.DataFrame(coords, columns=['x', 'y', 'z'][:len(coords[0])])[['x', 'y']]

        elif isinstance(coords, pd.DataFrame):
            lower_cols = {col: col.lower() for col in coords.columns}

            x_col = next((col for col, low in lower_cols.items() if 'x' in low), None)
            y_col = next((col for col, low in lower_cols.items() if 'y' in low), None)

            if x_col is None or y_col is None:
                raise ValueError(
                    f"Could not find 'x' and 'y' columns (case-insensitive). "
                    f"Got: {coords.columns.tolist()}"
                )
            # standardise to 'x', 'y'
            coords = coords.rename(columns={x_col: 'x', y_col: 'y'})

        else:
            raise TypeError(f"coords must be a list of tuples or a DataFrame, got {type(coords)}")

        # get maximum and minimum x and y coordinates
        max_x, max_y = coords[['x', 'y']].max()
        min_x, min_y = coords[['x', 'y']].min()

        # calculate the height and width of the mask
        H0 = int(max_x - min_x + 1)
        W0 = int(max_y - min_y + 1)

        # create the mask
        # deal with potential non-integer coordinates by flooring to the nearest integer (assuming coordinates are pixel centers)
        rows = (coords['x'].to_numpy() - min_x).astype(np.int32)
        cols = (coords['y'].to_numpy() - min_y).astype(np.int32)

        # x_range, y_range = np.arange(min_x, max_x + 1), np.arange(min_y, max_y + 1)
        # rows = np.floor(coords['x'].to_numpy() - min_x).astype(np.int32)
        # cols = np.floor(coords['y'].to_numpy() - min_y).astype(np.int32)
        # M0 = np.zeros((rows, cols), dtype=np.uint8)
        M0 = np.zeros((H0, W0), dtype=np.uint8)
        M0[rows, cols] = 1

        # check that the number of data points matches the sum of the mask
        assert M0.sum() == len(coords), "Data points do not match the mask sum."

        return M0, min_x, min_y

    @staticmethod
    def coarsen_mask_child_logic(M_prev):
        """
        Coarsens a binary mask M_prev by a factor of 2 using the >=2 rule and enforcing 8-connectivity.
        
        Parameters
        ----------
        M_prev : ndarray of shape (H_prev, W_prev)
            The binary mask to be coarsened.
        
        Returns
        -------
        M_coarse : ndarray of shape (H, W)
            The coarsened binary mask.
        """
        
        # Get the shape of the input mask
        H_prev, W_prev = M_prev.shape
        
        # Calculate the shape of the output mask (which is half the size of the input mask)
        H = H_prev // 2
        W = W_prev // 2
        
        # Crop the input mask to even dimensions
        M = M_prev[:2*H, :2*W]
        
        # Reshape the mask into 2x2 blocks
        blocks_mask = M.reshape(H, 2, W, 2)
        
        # Apply the >=2 rule to each block
        child_counts = blocks_mask.sum(axis=(1, 3))
        M_coarse = (child_counts >= 2)
        
        # Enforce 8-connectivity by keeping only the largest connected component
        label_connectivity = sk_label(M_coarse, connectivity=2) # connectivity=2 for 8-connectivity
        
        # Count the number of pixels in each connected component
        bincount = np.bincount(label_connectivity.flat)
        
        # Find the label of the largest connected component
        largest_connected_component = np.argmax(bincount[1:]) + 1
        
        # Create a mask to keep only the largest connected component
        keep_mask = (label_connectivity == largest_connected_component)
        
        # Convert the mask to uint8 and return
        M_coarse = keep_mask.astype(np.uint8)
        
        return M_coarse
    
    def coarsen_levels(self, M0, n_levels=4, plot=True):
        """
        Creates a pyramid of coarsened masks from the finest level mask M0.

        Parameters
        ----------
        H0, W0 : int
            The height and width of the finest level mask M0.
        M0 : ndarray
            The finest level mask (occupancy).
        n_levels : int
            The number of coarsened levels to create.
        plot : bool
            If True, plot the coarsened masks at different levels.

        Returns
        -------
        pyramid_masks : dict
            A dictionary containing the coarsened masks at different levels.
            Keys are the level numbers (0 = finest, 1 = coarsest, ...).
        """
        # Initialize the dictionary of masks at different levels
        pyramid_masks = {}

        # Add the finest level mask to the dictionary
        pyramid_masks[0] = M0  # finest level

        # Iterate over the number of levels
        for level in range(1, n_levels):
            # Get the current mask
            current_mask = pyramid_masks[level - 1]

            # Coarsen the mask using the child logic
            coarsened_mask = self.coarsen_mask_child_logic(current_mask)

            # Add the coarsened mask to the dictionary
            pyramid_masks[level] = coarsened_mask

        print("✓ Pyramid Masks created.")

        # If plotting is enabled
        if plot:
            # Set the figure size
            figsize_per_panel = (7, 7)
            cols = len(pyramid_masks)
            fig_w = figsize_per_panel[0] * cols
            fig_h = figsize_per_panel[1]

            # Create a figure with subplots
            fig, axes = plt.subplots(1, cols, figsize=(fig_w, fig_h))
            if cols == 1:
                axes = [axes]

            # Iterate over the masks and plot them
            for ax, (lvl, img) in zip(axes, pyramid_masks.items()):
                # Plot the image
                ax.imshow(img, cmap='Spectral', interpolation='nearest')
                # Set the title
                ax.set_title(f'Level {lvl}: Dim {img.shape}')
                # Remove axis labels and ticks
                ax.axis('off')

            # Set the figure title
            fig.suptitle("Coarsened Masks at Different Levels", fontsize=16)
            # Adjust the layout
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            # Show the plot
            plt.show()

        return pyramid_masks

    def aggregate_data_on_pyramid(
        self,
        sample_features: pd.DataFrame,
        pyramid_masks: dict,
        min_x: int,
        min_y: int,
    ) -> dict[int, np.ndarray]:
        """
        Aggregate data on a pyramid with the given masks.

        Parameters
        ----------
        sample_features : pd.DataFrame
            Sample features matrix (n_samples, n_features).
        pyramid_masks : dict[int, np.ndarray]
            Pyramid masks at different levels {level: (H, W)}.
        min_x : int
            Minimum x-coordinate of the coordinates.
        min_y : int
            Minimum y-coordinate of the coordinates.

        Returns
        -------
        dict[int, np.ndarray]
            Aggregated data on the pyramid at different levels {level: (H, W, n_features)}.
        """
        # Get coordinates of sample features
        coords   = np.array(sample_features.index.tolist())

        # Compute fine row and column indices (at finest resolution)
        try: 
            fine_row = (coords[:, 0] - min_x).astype(np.int32)
            fine_col = (coords[:, 1] - min_y).astype(np.int32)
        except:
            print("Error computing fine row and column indices. Check the coordinates and min_x, min_y values.")
            raise TypeError("Coordinates should be in the format (x, y). Most likely the data is not in sample x feature format or the coordinates are not properly extracted.")

        # Get sample feature values
        values   = sample_features.to_numpy(dtype=np.float64)    # (n_samples, n_features)

        # Initialize aggregated data on the pyramid
        pyramid_data = {}

        # Iterate over each level in the pyramid
        for level, mask in pyramid_masks.items():
            # Get dimensions of the current level mask
            H, W  = mask.shape

            # Compute the scaling factor for the current level (2^level)
            # This is used to compute the coarse row and column indices
            scale = 2 ** level

            # Compute coarse row and column indices (at current resolution)
            row_c = fine_row // scale
            col_c = fine_col // scale

            # Find valid pixels in the current level
            valid  = (row_c >= 0) & (row_c < H) & (col_c >= 0) & (col_c < W)
            
            # flatten represenetation of the 2D mask for efficient indexing
            flat_idx = row_c[valid] * W + col_c[valid]  # (n_valid,)

            # Get sample feature values for valid pixels
            vals = values[valid]                     # (n_valid, n_features)

            # Initialize sum and count arrays for current level
            n_pixels = H * W
            n_feats  = vals.shape[1]
            sums   = np.zeros((n_pixels, n_feats), dtype=np.float64)
            counts = np.zeros((n_pixels, 1),       dtype=np.float64)  # feature-independent

            # Scatter-sum and count valid children per pixel
            np.add.at(sums,   flat_idx, vals)
            np.add.at(counts, flat_idx, 1)

            # Average over valid children only
            with np.errstate(invalid="ignore", divide="ignore"):
                means = np.where(counts > 0, sums / counts, np.nan)  # (n_pixels, n_features)

            # Log-transform (log1p to handle zeros gracefully)
            with np.errstate(invalid="ignore"):
                log_means = np.log1p(means)

            # Reshape and mask non-tissue pixels
            log_means = log_means.reshape(H, W, n_feats)
            log_means[~mask.astype(bool)] = np.nan

            pyramid_data[level] = log_means            # (H, W, n_features)

        return pyramid_data
    
    def plot_pyramid_qc(
        self,
        pyramid_data: dict,
        feature_idx: int = 0,
        cmap: str = "Spectral",
    ):
        """
        This function generates a quality control heatmap for each pyramid level.
        It shows the summed intensity for a single feature across all levels.

        Parameters
        ----------
        pyramid_data : output of aggregate_data_on_pyramid {level: (H, W, n_features)}
            - contains data aggregated on each pyramid level
        
        feature_idx  : which feature (column index) to visualise
        
        cmap         : matplotlib colormap
        """
        # get the number of levels in the pyramid
        n_levels = len(pyramid_data)

        # create figure with subplots
        fig, axes = plt.subplots(1, n_levels, figsize=(5 * n_levels, 5),
                                gridspec_kw={"wspace": 0.3})
        if n_levels == 1:
            axes = [axes]

        # loop over each level, plotting the heatmap for that level
        for ax, (level, data) in zip(axes, sorted(pyramid_data.items())):
            # get the dimensions of the data for this level
            H, W = data.shape[:2]

            # plot the heatmap for this level
            im = ax.imshow(data[:, :, feature_idx], cmap=cmap,
                        interpolation="nearest", aspect="equal")
            ax.set_title(f"Level {level}\n{H}×{W}", fontsize=11, fontweight="bold")
            ax.axis("off")

            # add colorbar
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # add a title for the entire figure
        fig.suptitle(
            f"Pyramid QC  —  feature index {feature_idx}  (aggregated log-intensity)",
            fontsize=13, fontweight="bold"
        )
        #plt.tight_layout()
        plt.show()
    # Additional methods for multimodal registration can go here

if __name__ == "__main__":
    print("This is a module for multimodal registration using SPACO and SpectrumData classes.")

    # loading data and pathways 
    imzml_path = Path("./MSI_data_grant/Mass_Spec_data/20251012_old_liver_area.imzML")

    # Example usage:
    multimodal_registration = MultiModalRegistration(
        imzml_path=imzml_path, 
        min_intensity=1, 
        min_count=100, 
        mz_tol=0.0042, 
        use_gpu=True
        )
    
    # Run the pipeline for cell segmentation and overlay creation.    
    #he_image_path = "./high_res_MSI/MSI data to share/20250923_young_liver_9AA_I90S25/Histo_2025-10-01_11.40.25_9211.ome.tif"
    
    # generating the nuclei masks to segment the pixels of interest 
    #multimodal_registration.segmentation_pipeline(he_image_path, save_overlay=False)

    # run the pipeline for data level creation and aggregation
    # create the data levels for the pyramid
    sample_features = multimodal_registration._sample_feature_genration(agg_func="sum")
    M0, min_x, min_y = multimodal_registration.data_level_creation(sample_features.index.tolist())

    # create the coarsened masks for the pyramid
    pyramid_masks = multimodal_registration.coarsen_levels(M0, n_levels=5, plot=True)

    # aggregate the data on the pyramid
    pyramid_data = multimodal_registration.aggregate_data_on_pyramid(
        sample_features=sample_features,
        pyramid_masks=pyramid_masks,
        min_x = min_x,
        min_y = min_y)
    
    # plot the pyramid QC for the first feature (index 0)
    multimodal_registration.plot_pyramid_qc(pyramid_data, feature_idx=1, cmap="Spectral")




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
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import tifffile
from scipy import ndimage

class MultiModalRegistration(SpectrumData):
    def __init__(self, *args, use_gpu=True, **kwargs):
        super().__init__(*args, **kwargs)
        # Additional initialization for multimodal registration can go here
        # Configure GPU settings
        self.use_gpu = use_gpu
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

    def load_massi_spectrometry_data(self, full_data=False, sample_features=True):
        """
        Loads mass spectrometry data from imzML file into SpectrumData object.

        Parameters
        ----------
        full_data : bool, optional
            If True, returns the full SpectrumData object dataframe and the coordinates of the samples.
        sample_features : bool, optional
            If True, returns the sample features matrix generated using the _sample_feature_genration method and the coordinates of the samples.

        Returns
        -------
        If full_data is True, returns a tuple containing the SpectrumData object dataframe and a list of the coordinates of the samples.
        If sample_features is True, returns a tuple containing the sample features matrix and a list of the coordinates of the samples.
        """
        
        # transform to full dataframe
        if full_data: 
            return self.df, [coords[:2] for coords in self.df['coordinates']]
        
        # transform to sample features matrix 
        if sample_features: 
            return self._sample_feature_genration(agg_func="sum"), [coords[:2] for coords in self.df['coordinates']]
        
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
    
    def data_level_creation(self, mz_df: pd.DataFrame, cmap: str = "Spectral"): 
        """
        Create data levels for multimodal registration.
        """

        # get size of data (need to make coordinates as x, y)then take max of each column  
        mz_df[['x', 'y', 'z']] = pd.DataFrame(mz_df['coordinates'].tolist(), index=mz_df.index)
        max_x, max_y = mz_df[['x', 'y']].max()
        min_x, min_y = mz_df[['x', 'y']].min()
        
        # grabbing dimensions of RASTER grid 
        H0 = max_x - min_x + 1
        W0 = max_y - min_y + 1

        # initialize mask from the data dimensions
        M0 = np.zeros((H0, W0), dtype=np.uint8) # initialize mask
        for coord in mz_df['coordinates']:
            x, y = coord[:2]
            row = x - min_x
            col = y - min_y
            M0[row, col] = 1  # mark presence of data point

        assert M0.sum() == len(mz_df['coordinates']), "Data points do not match the mask sum."

        plt.imshow(M0, cmap=cmap)
        plt.title("Bottom-level mask M0 (occupancy)")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()

        # return the dictionary of level data
        #self.level_data = {i :(round(max_x / 2**i), round(max_y / 2**i)) for i in range(n_levels)}
        #
        # print the level data 

        return H0, W0, M0
    
    def coarsen_levels(self, H0, W0, M0, n_levels=4, plot=True):
        level_data = {i :(round(H0 / 2**i), round(W0 / 2**i)) for i in range(1,n_levels)}

        # create coarsened masks for each level
        pyramid_masks = {}
        for level, (H, W) in level_data.items():
            M_coarse = cv2.resize(M0, (W, H), interpolation=cv2.INTER_NEAREST)
            pyramid_masks[level] = M_coarse

        # include original mask as level 0
        pyramid_masks[0] = M0 
        print("✓ Pyramid Masks created.")
        if plot:
                figsize_per_panel = (7,7)
                cols = len(pyramid_masks.keys())
                fig_w = figsize_per_panel[0] * cols
                fig_h = figsize_per_panel[1]
                fig, axes = plt.subplots(1, cols, figsize=(fig_w, fig_h))

                im = None
                for ax, (title, img) in zip(axes, pyramid_masks.items()):
                    im = ax.imshow(img, cmap='Spectral', interpolation='nearest')
                    ax.set_title(f'level {title}: Dim {pyramid_masks[title].shape}')
                    ax.axis('off')

                fig.suptitle(f"Coarsened Masks at Different Levels", fontsize=16)

                    # remove extra white space
                plt.tight_layout(rect=[0, 0, 1, 0.95])
                plt.show()
        return 

    # Additional methods for multimodal registration can go here
if __name__ == "__main__":
    print("This is a module for multimodal registration using SPACO and SpectrumData classes.")

    # loading data and pathways 
    imzml_path = Path("./MSI_data_grant/Mass_Spec_data/20251012_old_liver_area.imzML")

    # Example usage:
    multimodal_registration = MultiModalRegistration(
        imzml_path=imzml_path, 
        min_intensity=100, 
        min_count=100, 
        mz_tol=0.0042, 
        use_gpu=True
        )
    
    # Run the pipeline for cell segmentation and overlay creation. 
    he_image_path = "./high_res_MSI/MSI data to share/20250923_young_liver_9AA_I90S25/Histo_2025-10-01_11.40.25_9211.ome.tif"
    
    # generating the nuclei masks to segment the pixels of interest 
    multimodal_registration.segmentation_pipeline(he_image_path, save_overlay=False)

    print("seems achim is happy with the segmentation part as it is working now")
    
    # Want the implementation of the multi spaco next?

    # step 1 break down data into levels 




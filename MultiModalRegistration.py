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
from pyimzml.ImzMLParser import ImzMLParser

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

    def load_maldi(self, data_path: str) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
        """
        Load MALDI-MSI data from an imzML file.
        Returns count matrix (n_spots, n_mz), integer coords (n_spots, 2), m/z values.
        """
        parser     = ImzMLParser(Path(data_path))
        spectra    = [parser.getspectrum(i) for i in range(len(parser.coordinates))]
        mzs        = [s[0] for s in spectra]
        intensities= [s[1] for s in spectra]

        # Build COO triplets: row = spot index, col = unique m/z bin
        all_mzs        = np.concatenate(mzs)
        unique_mzs, col_idx = np.unique(all_mzs, return_inverse=True)
        row_idx        = np.repeat(np.arange(len(mzs)), [len(m) for m in mzs])
        coords         = np.array(parser.coordinates)[:, :2]  # integer (x, y)

        X = sp.coo_matrix((np.concatenate(intensities), (row_idx, col_idx)),
                        shape=(len(mzs), len(unique_mzs))).tocsr()
        print(f"Loaded MALDI | spots: {X.shape[0]:,} | m/z bins: {X.shape[1]:,}")
        return X, coords, unique_mzs


    def load_visium(self,data_path: str) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
        """
        Load Visium HD data from a CellRanger output directory.
        Returns count matrix (n_spots, n_genes), float pixel coords, gene names.
        """
        path  = Path(data_path)
        adata = sc.read_10x_h5(path / "filtered_feature_bc_matrix.h5")
        adata.var_names_make_unique()

        # Align spatial coordinates to count matrix barcodes
        pos = (pd.read_parquet(path / "spatial" / "tissue_positions.parquet")
                .set_index("barcode").loc[adata.obs_names])
        coords = pos[["pxl_col_in_fullres", "pxl_row_in_fullres"]].to_numpy()

        X = sp.csr_matrix(adata.X)
        print(f"Loaded Visium | spots: {X.shape[0]:,} | genes: {X.shape[1]:,}")
        return X, coords, adata.var_names.to_numpy()

        
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
    he_image_path = "./high_res_MSI/MSI data to share/20250923_young_liver_9AA_I90S25/Histo_2025-10-01_11.40.25_9211.ome.tif"
    
    # generating the nuclei masks to segment the pixels of interest 
    nuclei_masks = multimodal_registration.segmentation_pipeline(he_image_path, save_overlay=False)

    # run the pipeline for data level creation and aggregation
    # create the data levels for the pyramid
    print('breakpoint')




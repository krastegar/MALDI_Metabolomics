
from __future__ import annotations

from pathlib import Path
from sklearn.neighbors import KDTree
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scanpy as sc
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap
from pyimzml.ImzMLParser import ImzMLParser
from skimage.measure import label as sk_label

class DataLoader:
    def __init__(self, data_path: str, maldi: bool = True, visium: bool = False) -> None:
        self.maldi = maldi
        self.visium = visium
        
        # make sure only one of maldi or visium is True
        if self.maldi: 
            self.data_path = Path(data_path)
            self.parser = ImzMLParser(self.data_path)
        elif self.visium:
            self.data_path = Path(data_path)
        else:
            raise ValueError("Please specify either MALDI or Visium data to load.")
        
        self.sparse_SF = None  # DataFrame to hold the loaded data
        self.maldi_coords = None  # To store coordinates for MALDI data
        self.visium_coords = None  # To store coordinates for Visium data
        self.barcodes = None  # To store barcodes for Visium data
        self.features = None  # To store features for both features 
        self.barcode_mapping = None  # To store mapping from grid indices to barcodes for Visium data
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
            self.maldi_coords = coords[:, :2]  # Keep only x and y
            
            # Concatenate all m/z and intensity arrays to build the COO matrix
            all_mzs = np.concatenate(mzs) # makes all m/z values in one array, same / intsensity
            all_intensities = np.concatenate(intensities)

            # Build row indices: repeat each sample index by its spectrum length
            row_indices = np.repeat(np.arange(len(mzs)), [len(m) for m in mzs])

            # Get unique m/z values and mapping
            unique_mzs, col_indices = np.unique(all_mzs, return_inverse=True)
            
            # store the features (m/z values) for later use
            self.features = unique_mzs

            # Build COO matrix
            coo = sp.coo_matrix((all_intensities, (row_indices, col_indices)),
                                shape=(len(mzs), len(unique_mzs)))
            
            self.sparse_SF = coo.tocsr()  # Convert to CSR format for efficient row slicing
            return self.sparse_SF
        
        elif self.visium:
            """
            data_path should point to the directory containing the Visium data, which typically includes:
            - `filtered_feature_bc_matrix.h5` (or similar): The main data file containing the count matrix.
            - `spatial` directory: Contains spatial information and images.
            """
            # ------------------------------------------------------------------ #
            #  1. Count matrix                                                     #
            # ------------------------------------------------------------------ #
            adata = sc.read_10x_h5(
                self.data_path / "filtered_feature_bc_matrix.h5"
            )
            adata.var_names_make_unique()

            # ------------------------------------------------------------------ #
            #  2. Spatial coordinates from parquet                                 #
            # ------------------------------------------------------------------ #
            tissue_positions = pd.read_parquet(
                self.data_path / "spatial" / "tissue_positions.parquet"
            )

            # Align barcodes between count matrix and tissue positions
            tissue_positions = tissue_positions.set_index("barcode")
            tissue_positions = tissue_positions.loc[adata.obs_names]

            # ------------------------------------------------------------------ #
            #  3. Store everything                                                 #
            # ------------------------------------------------------------------ #
            self.coords   = tissue_positions[
                ["pxl_col_in_fullres", "pxl_row_in_fullres"]
            ].to_numpy()                                  # shape (n_spots, 2)
            self.barcodes = adata.obs_names.to_numpy()    # shape (n_spots,)
            self.features    = adata.var_names.to_numpy()    # shape (n_genes,)
            self.sparse_SF = sp.csr_matrix(adata.X)

            print(f"Loaded Visium | spots: {self.sparse_SF.shape[0]:,} | "
                f"genes: {self.sparse_SF.shape[1]:,} | ")
                #f"non-zero entries: {self.sparse_SF.nnz:,}")

            return self.sparse_SF
        
        else: 
            raise ValueError("Please specify either MALDI or Visium data to load.")
    
    def data_level_creation(self) -> np.ndarray:
        """
        Creates the bottom level grid for the spatial pyramid. For MALDI, the coordinates are integers.  
        For Visium, they are float values. We want to create a mask of valid positions in the grid for both 
        types of data.

        Returns
        -------
        M0 : np.ndarray
            A binary mask of shape (H, W) where H and W are the dimensions of the grid. Valid positions are 
            marked with 1, and invalid positions are marked with 0.
        """
        if self.maldi:

            # determine the grid size based on the max coordinates
            n_rows: int = self.maldi_coords[:, 0].max() + 1
            n_cols: int = self.maldi_coords[:, 1].max() + 1

            # create the mask of valid positions using COO format for efficiency
            rows : np.ndarray = self.maldi_coords[:, 0]              
            cols : np.ndarray = self.maldi_coords[:, 1]
            values = np.ones(len(rows), dtype=np.uint8) # binary: 1 = valid position

            # create the mask in COO format
            M0_coo = sp.coo_matrix(
                (values, (rows, cols)),
                shape=(n_rows, n_cols),
                dtype=np.uint8
            )
            # Convert to CSR format for efficient slicing later
            M0 = M0_coo.tocsr()
            
            # plotting the mask
            cmap = ListedColormap(['black', 'red'])
            plt.imshow(M0.toarray(), cmap=cmap, vmin=0, vmax=1)
            plt.title("MALDI Valid Position Mask (M0)")
            plt.axis('off')

            # Create legend handles
            legend_handles = [
                Patch(facecolor='black', edgecolor='black', label='0 (invalid)'),
                Patch(facecolor='red', edgecolor='black', label='1 (valid)')
            ]
            plt.legend(handles=legend_handles, loc='lower right', frameon=True)
            plt.show()
            return M0
        
        elif self.visium: 
            # we need to determine the grid size but the coordinates are float values.
            # we should discritize the coordinates and figure out which cells are valid 
            # to do this we look at the median distance of the nearest neighbors and use that as the 
            # grid resolution.
            tree = KDTree(self.coords)
            distances, _ = tree.query(self.coords, k=2)  # k=2 since k=1 is self
            resolution = np.median(distances[:, 1])
            print(f"Inferred resolution: {resolution:.4f}px")

            # shifting coordinates to start from (0, 0) # this is important so that we 
            shifted_coords = self.coords - self.coords.min(axis=0)

            # Discretize coordinates to create grid indices
            grid_indices = np.floor(shifted_coords / resolution).astype(int)

            # save the discretized coordinates for later use
            self.visium_coords = grid_indices

            # Create a mask of valid positions using COO format for efficiency
            rows : np.ndarray = grid_indices[:, 0]
            cols : np.ndarray = grid_indices[:, 1]
            values = np.ones(len(rows), dtype=np.uint8) # binary: 1 = valid position

            # create the mask in COO format and then convert to CSR for efficient slicing later
            M0_coo = sp.coo_matrix(
                (values, (rows, cols)),
                shape=(rows.max() + 1, cols.max() + 1),
                dtype=np.uint8
            )
            M0 = M0_coo.tocsr()

            # now we want the barcode mapping so that we can map from the grid to the barcodes 
            # we can do this by creating flat indices for the grid and then mapping those to the barcodes
            flat_indices = grid_indices[:, 1] * M0.shape[1] + grid_indices[:, 0] # popular flat encoding method provided by claude
            self.barcode_mapping = dict(zip(flat_indices, self.barcodes))

            # plotting the mask
            cmap = ListedColormap(['black', 'red'])
            plt.imshow(M0.toarray(), cmap=cmap, vmin=0, vmax=1)
            plt.title("Visium Valid Position Mask (M0)")
            plt.axis('off')

            # Create legend handles
            legend_handles = [
                Patch(facecolor='black', edgecolor='black', label='0 (invalid)'),
                Patch(facecolor='red', edgecolor='black', label='1 (valid)')
            ]
            plt.legend(handles=legend_handles, loc='lower right', frameon=True)
            plt.show()

            return M0
        else: 
            raise ValueError("Place holder for when we return both visium and maldi.")
    
    def coarsen_mask_child_logic(self, M_prev):
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
        
        # check if input mask grid is sparse 
        if sp.issparse(M_prev):
            M_prev = M_prev.toarray()  # Convert to dense array for processing
        
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

        # Convert to sparse matrix for memory efficiency
        M_coarse = sp.coo_matrix(M_coarse, dtype=np.uint8).tocsr()
        
        return M_coarse
    def coarsen_levels(self, M0: sp.csr_matrix, target_max: int, plot: bool = True) -> dict[int, dict]:
        """
        Creates a pyramid of coarsened masks from the finest level mask M0.
        Each level stores the mask, tissue_idx, and grid_shape.

        Parameters
        ----------
        M0         : sp.csr_matrix  Binary mask at finest level, shape (H, W)
        target_max : int            Coarsen until n_valid < target_max
        plot       : bool           If True, plot masks at each level

        Returns
        -------
        pyramid_levels : dict[int, dict]
            {
                level: {
                    'mask'       : sp.csr_matrix  shape (H, W)
                    'tissue_idx' : np.ndarray     flat indices of tissue pixels
                    'grid_shape' : (H, W)
                }
            }
        """

        def make_level(mask: sp.csr_matrix) -> dict:
            """ Package a mask into a pyramid level dict. """
            H, W    = mask.shape
            coo     = mask.tocoo()
            tissue_idx = coo.row * W + coo.col
            return {
                'mask'       : mask,
                'tissue_idx' : tissue_idx,
                'grid_shape' : (H, W)
            }

        # ------------------------------------------------------------------ #
        #  Build pyramid                                                        #
        # ------------------------------------------------------------------ #
        pyramid_levels        = {}
        pyramid_levels[0]     = make_level(M0)
        M_current             = M0

        # Coarsen until n_valid < target_max (dynamically determine number of levels based on target_max)
        level = 0
        while M_current.nnz >= target_max:
            print(f"Level {level}: n = {M_current.nnz:,}")
            M_current             = self.coarsen_mask_child_logic(M_current)
            level                += 1
            pyramid_levels[level] = make_level(M_current)

        print(f"✓ Pyramid built — {level + 1} levels | "
            f"final n = {M_current.nnz:,}")

        # ------------------------------------------------------------------ #
        #  Plot                                                              #
        # ------------------------------------------------------------------ #
        if plot:
            n_panels = len(pyramid_levels)
            n_cols   = min(4, n_panels)
            n_rows   = int(np.ceil(n_panels / n_cols))

            fig_w = 7 * n_cols * 1.5
            fig_h = 9 * n_rows * 1.5

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), squeeze=False)
            axes_flat = axes.flatten()

            for ax, (lvl, lvl_data) in zip(axes_flat, pyramid_levels.items()):
                ax.imshow(lvl_data['mask'].toarray(), cmap='Spectral',
                        interpolation='nearest', vmin=0, vmax=1)
                ax.set_title(f'Level {lvl}: {lvl_data["grid_shape"]}')
                ax.axis('off')

            for ax in axes_flat[n_panels:]:
                ax.axis('off')

            fig.suptitle("Coarsened Masks at Different Levels", fontsize=16)
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.subplots_adjust(hspace=0.5, wspace=0.3)
            plt.show()

        return pyramid_levels

    def aggregate_data_on_pyramid(
            self,
            sample_features: sp.csr_matrix,
            pyramid_levels: dict[int, dict]
        ) -> dict[int, dict]:
        """
        Aggregate count data at each pyramid level using tissue indices.

        Parameters
        ----------
        sample_features : sp.csr_matrix
            Count matrix, shape (n_spots, n_feats)
        pyramid_levels : dict[int, dict]
            Output of coarsen_levels(), keyed by level with 'mask', 'tissue_idx', 'grid_shape'

        Returns
        -------
        pyramid_levels : dict[int, dict]
            Same structure with 'data' added at each level:
            level: {
                'mask'       : sp.csr_matrix  shape (H, W)
                'tissue_idx' : np.ndarray     flat indices of tissue pixels
                'grid_shape' : (H, W)
                'data'       : sp.csr_matrix  shape (H*W, n_feats)
            }
        """
        # ------------------------------------------------------------------ #
        #  1. Get coordinates                                                  #
        # ------------------------------------------------------------------ #
        if self.visium:
            coords = self.visium_coords
        elif self.maldi:
            coords = self.maldi_coords
        else:
            raise ValueError("Please specify either MALDI or Visium data.")

        fine_row = (coords[:, 0] - coords[:, 0].min()).astype(np.int32)
        fine_col = (coords[:, 1] - coords[:, 1].min()).astype(np.int32)

        # ------------------------------------------------------------------ #
        #  2. Ensure sample_features is CSR                                    #
        # ------------------------------------------------------------------ #
        if not sp.issparse(sample_features):
            sample_features = sp.csr_matrix(sample_features)
        else:
            sample_features = sample_features.tocsr()

        n_feats = sample_features.shape[1]

        # ------------------------------------------------------------------ #
        #  3. Aggregate at each level                                          #
        # ------------------------------------------------------------------ #
        for level, lvl_data in pyramid_levels.items():

            H, W     = lvl_data['grid_shape']
            n_pixels = H * W
            scale    = 2 ** level

            # Coarse indices for every spot at this level
            row_c     = fine_row // scale
            col_c     = fine_col // scale
            in_bounds = (row_c >= 0) & (row_c < H) & (col_c >= 0) & (col_c < W)
            flat_idx  = (row_c[in_bounds] * W + col_c[in_bounds]).astype(np.int32)
            spot_idx  = np.where(in_bounds)[0]
            n_valid   = len(spot_idx)

            # ------------------------------------------------------------------ #
            #  4. Aggregation matrix A (n_pixels, n_valid)                        #
            # ------------------------------------------------------------------ #
            A = sp.coo_matrix(
                (np.ones(n_valid, dtype=np.float64),
                (flat_idx, np.arange(n_valid))),
                shape=(n_pixels, n_valid)
            ).tocsr()

            # ------------------------------------------------------------------ #
            #  5. Scatter sum → mean → log1p                                       #
            # ------------------------------------------------------------------ #
            valid_features  = sample_features[spot_idx]         # (n_valid, n_feats)
            sums            = A @ valid_features                 # (n_pixels, n_feats)
            counts          = np.asarray(A.sum(axis=1))         # (n_pixels, 1)

            occupied_idx    = np.where(counts.flatten() > 0)[0]
            counts_occupied = counts[occupied_idx]

            means      = sums[occupied_idx].multiply(1.0 / counts_occupied).tocsr()
            means.data = np.log1p(means.data)

            # ------------------------------------------------------------------ #
            #  6. Reconstruct full pixel space                                     #
            # ------------------------------------------------------------------ #
            rows_out, cols_out = means.nonzero()
            vals_out           = np.asarray(means[rows_out, cols_out]).flatten()
            actual_rows        = occupied_idx[rows_out]

            data = sp.coo_matrix(
                (vals_out, (actual_rows, cols_out)),
                shape=(n_pixels, n_feats)
            ).tocsr()

            print(f"Level {level} | grid: ({H} x {W}) | "
                f"tissue pixels: {len(occupied_idx):,} | "
                f"nnz: {data.nnz:,}")

            lvl_data['data'] = data

        return pyramid_levels
    def plot_pyramid_data(self, pyramid_levels: dict[int, dict], feature_name: float = 0) -> None:
        """
        Plot aggregated feature data at each pyramid level.

        Parameters
        ----------
        pyramid_levels : dict[int, dict]
            Output of aggregate_data_on_pyramid(), keyed by level.
        feature_idx : int
            Index of the feature (gene) to visualise across levels.
        """
        n_panels = len(pyramid_levels)
        n_cols   = min(4, n_panels)
        n_rows   = int(np.ceil(n_panels / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(10.5 * n_cols, 13.5 * n_rows), squeeze=False)
        axes_flat = axes.flatten()

        for ax, (level, lvl_data) in zip(axes_flat, pyramid_levels.items()):

            H, W       = lvl_data['grid_shape']
            tissue_idx = lvl_data['tissue_idx']
            data       = lvl_data['data']

            # Extract single feature column — avoids reconstructing full (H, W, n_feats)
            feature_col = np.asarray(data[:, feature_idx].todense()).flatten()

            # Scatter into (H, W) grid — non-tissue pixels remain NaN
            grid              = np.full((H, W), np.nan)
            rows, cols        = tissue_idx // W, tissue_idx % W
            grid[rows, cols]  = feature_col[tissue_idx]

            im = ax.imshow(grid, cmap='viridis', interpolation='nearest')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(f'Level {level}: {lvl_data["grid_shape"]} | '
                        f'tissue pixels: {len(tissue_idx):,}')
            ax.axis('off')

        for ax in axes_flat[n_panels:]:
            ax.axis('off')

        fig.suptitle(f'Aggregated data across pyramid levels | feature index: {feature_idx}',
                    fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.subplots_adjust(hspace=0.4, wspace=0.3)
        plt.show()


if __name__ == "__main__":

    # Example usage:
    data_loader = DataLoader(data_path="MSI_data_grant/Mass_Spec_data/20251012_old_liver.imzML", maldi=True, visium=False)
    sparse_matrix = data_loader.load_sparse_data()
    print(sparse_matrix.shape)

    # sparse matrix for visium data
    data_loader_visium = DataLoader(data_path="MSI_data_grant/cellranger/329537/outs/binned_outputs/square_002um/", visium=True, maldi=False)
    sparse_matrix_visium = data_loader_visium.load_sparse_data()
    print(sparse_matrix_visium.shape)



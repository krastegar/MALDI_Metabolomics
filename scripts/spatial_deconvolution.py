"""
MALDI-MSI Spatial Deconvolution — True Cellular Resolution
===========================================================

PROBLEM STATEMENT
-----------------
Each MALDI pixel measurement I(p) is a physical mixture of metabolite
signal from every cell within the instrument beam footprint:

    I(p) = sum_k  w_pk * c_k  +  epsilon

where:
    c_k    = true metabolite concentration in cell k  (UNKNOWN — what we want)
    w_pk   = PSF weight: contribution of cell k to pixel p  (KNOWN from geometry)
    epsilon = noise

This is a linear unmixing problem. Given the measurement vector I and the
weight matrix W (built from PSF + cell positions), we solve for c:

    c_hat = argmin ||Wc - I||^2   subject to  c >= 0

Output: a (n_cells x n_channels) matrix of per-cell metabolite concentrations.
This is TRUE cellular resolution — one metabolic profile per nucleus,
not a blurred PSF-weighted estimate.

MATHEMATICAL BASIS
------------------
W is sparse: each MALDI pixel only receives contributions from cells within
~3 PSF radii (~75 um at 25 um pixel size). With 270k pixels and 100k+ cells,
W has shape (n_pixels, n_cells) but >99% zeros. scipy.sparse handles this.

The per-channel NNLS solve is independent — can be parallelised trivially.

COMPARISON TO PSF CENTROID METHOD (CellRegPipeline.py)
-------------------------------------------------------
PSF centroid (old):  assigns one H&E coordinate to each MALDI pixel
                     (blurring: the coordinate is a weighted average of cells)
                     Output: pixels x coords   (no per-cell metabolite data)

Spatial deconvolution (this):  assigns one metabolite profile to each cell
                     (unblurring: inverts the PSF mixing)
                     Output: cells x metabolites  (true cellular data)

INPUTS REQUIRED
---------------
1. coordinate_mapping.csv     — from COARSE_REG_pt2.py
                                 columns: maldi_x, maldi_y, he_x, he_y
2. nuclei_masks.tif           — from StarDist segmentation (instance labels)
3. imzML file                 — raw MALDI spectra
4. H&E image (optional)       — for visualisation overlay

OUTPUTS
-------
cell_metabolite_profiles.csv   — (n_cells x n_channels) per-cell concentrations
cell_metadata.csv              — cell centroids, area, nearest MALDI pixel
deconvolution_qc.png           — QC plots: residuals, condition number, coverage
metabolite_overlay_*.png       — per-channel maps coloured by deconvolved value
"""

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Circle
from pathlib import Path
from scipy import ndimage as ndi
from scipy.spatial import KDTree
from scipy.optimize import nnls
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import lsqr
import warnings
warnings.filterwarnings('ignore')

try:
    import tifffile
    TIFFFILE_AVAILABLE = True
except ImportError:
    TIFFFILE_AVAILABLE = False

try:
    from pyimzml.ImzMLParser import ImzMLParser
    IMZML_AVAILABLE = True
except ImportError:
    IMZML_AVAILABLE = False
    print("WARNING: pyimzml not installed. Install with: pip install pyimzml")

try:
    from joblib import Parallel, delayed
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False
    print("WARNING: joblib not installed. Parallel solving disabled.")
    print("Install with: pip install joblib")


# ===========================================================================
#  UTILITY FUNCTIONS
# ===========================================================================

def load_nuclei_mask(mask_path):
    """
    Load nuclei segmentation mask.
    Accepts .tif (tifffile), .npy, or anything OpenCV can read.
    Returns integer-labelled array (each nucleus = unique int > 0).
    """
    path = str(mask_path)
    if path.endswith('.npy'):
        return np.load(path)
    if TIFFFILE_AVAILABLE and path.endswith(('.tif', '.tiff')):
        mask = tifffile.imread(path)
        print(f"  Loaded nuclei mask via tifffile: shape={mask.shape}, "
              f"dtype={mask.dtype}, max_label={mask.max()}")
        return mask
    mask = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(f"Cannot load mask: {path}")
    print(f"  Loaded nuclei mask via OpenCV: shape={mask.shape}")
    return mask


def load_he_image(he_path):
    """Load H&E image (BigTIFF safe)."""
    path = str(he_path)
    if TIFFFILE_AVAILABLE:
        try:
            img = tifffile.imread(path)
            if img.ndim == 4: img = img[0]
            if img.ndim == 2: img = np.stack([img]*3, axis=-1)
            if img.shape[2] == 4: img = img[:, :, :3]
            if img.dtype != np.uint8:
                img = img.astype(np.float32)
                img = ((img-img.min())/(img.max()-img.min()+1e-8)*255).astype(np.uint8)
            return img
        except Exception:
            pass
    img = cv2.imread(path)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ===========================================================================
#  CORE CLASS
# ===========================================================================

class SpatialDeconvolution:
    """
    PSF-based spatial deconvolution of MALDI-MSI to per-cell metabolite profiles.

    Usage
    -----
    dec = SpatialDeconvolution(
        coord_mapping_csv='registration_outputs/coordinate_mapping.csv',
        nuclei_mask_path='registration_outputs/nuclei_masks.tif',
        imzml_path='MSI_data/20251012_old_liver.imzML',
        maldi_pixel_size_um=25.0,
        he_pixel_size_um=4.0,
        output_dir='deconvolution_outputs',
    )
    dec.run()
    """

    def __init__(self,
                 coord_mapping_csv,
                 nuclei_mask_path,
                 imzml_path,
                 maldi_pixel_size_um=25.0,
                 he_pixel_size_um=4.0,
                 psf_sigma_um=None,
                 psf_truncate_sigma=3.0,
                 output_dir='deconvolution_outputs',
                 he_image_path=None,
                 n_jobs=-1,
                 min_cells_per_pixel=1,
                 regularisation=0.01):
        """
        Parameters
        ----------
        coord_mapping_csv : str
            Path to coordinate_mapping.csv from COARSE_REG_pt2.py.
            Columns: maldi_x, maldi_y, he_x, he_y
        nuclei_mask_path : str
            Path to StarDist nuclei segmentation (instance-labelled .tif).
        imzml_path : str
            Path to .imzML file (raw MALDI spectra).
        maldi_pixel_size_um : float
            MALDI raster step in µm (default 25).
        he_pixel_size_um : float
            H&E scanner pixel size in µm (default 4).
        psf_sigma_um : float or None
            PSF sigma in µm. None = FWHM=maldi_pixel_size → sigma=pixel/2.355.
        psf_truncate_sigma : float
            Search radius in multiples of sigma (default 3 = 99.7% of Gaussian).
        output_dir : str
            Directory for all outputs.
        he_image_path : str or None
            Optional H&E image for visualisation overlays.
        n_jobs : int
            Parallel workers for NNLS solve. -1 = all cores.
        min_cells_per_pixel : int
            Minimum cells in PSF footprint for a pixel to contribute to the solve.
            Pixels with fewer cells are excluded (unreliable mixing).
        regularisation : float
            Tikhonov regularisation added to diagonal of W^T W.
            Stabilises solve when cells are heavily overlapping in PSF space.
            0.0 = pure NNLS, 0.01 = mild, 0.1 = strong.
        """
        self.coord_mapping_csv   = coord_mapping_csv
        self.nuclei_mask_path    = nuclei_mask_path
        self.imzml_path          = imzml_path
        self.maldi_pixel_size_um = maldi_pixel_size_um
        self.he_pixel_size_um    = he_pixel_size_um
        self.psf_truncate_sigma  = psf_truncate_sigma
        self.output_dir          = Path(output_dir)
        self.he_image_path       = he_image_path
        self.n_jobs              = n_jobs
        self.min_cells_per_pixel = min_cells_per_pixel
        self.regularisation      = regularisation

        # Derived
        self.scale_factor = maldi_pixel_size_um / he_pixel_size_um

        # PSF sigma in H&E pixels
        if psf_sigma_um is None:
            self.psf_sigma_um = maldi_pixel_size_um / 2.355  # FWHM = pixel size
        else:
            self.psf_sigma_um = psf_sigma_um
        self.psf_sigma_he = self.psf_sigma_um / he_pixel_size_um

        # Search radius in H&E pixels
        self.search_radius_he = self.psf_truncate_sigma * self.psf_sigma_he

        # Will be populated during run()
        self.coord_mapping       = None
        self.nuclei_mask         = None
        self.cell_centroids_xy   = None   # (n_cells, 2) in H&E pixel coords
        self.cell_areas          = None   # (n_cells,) in H&E pixels^2
        self.cell_labels         = None   # (n_cells,) original StarDist labels
        self.W                   = None   # sparse PSF weight matrix (n_pixels, n_cells)
        self.pixel_to_maldi      = None   # maps row of W to (maldi_x, maldi_y)
        self.mz_axis             = None   # common m/z axis
        self.spectra_matrix      = None   # (n_pixels, n_channels) intensity matrix
        self.cell_profiles       = None   # (n_cells, n_channels) DECONVOLVED
        self.residuals           = None   # (n_pixels, n_channels) W*c_hat - I
        self.he_image            = None

        self.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*65}")
        print(f"MALDI-MSI SPATIAL DECONVOLUTION")
        print(f"{'='*65}")
        print(f"  MALDI pixel size:   {maldi_pixel_size_um} µm")
        print(f"  H&E pixel size:     {he_pixel_size_um} µm")
        print(f"  Scale factor:       {self.scale_factor:.2f} H&E px / MALDI px")
        print(f"  PSF sigma:          {self.psf_sigma_um:.1f} µm  "
              f"= {self.psf_sigma_he:.1f} H&E px")
        print(f"  Search radius:      {psf_truncate_sigma}σ "
              f"= {self.search_radius_he:.1f} H&E px  "
              f"= {self.search_radius_he * he_pixel_size_um:.1f} µm")
        print(f"  Regularisation:     {regularisation}")
        print(f"  Output dir:         {self.output_dir.resolve()}")

    # ------------------------------------------------------------------
    def load_inputs(self):
        """Load coordinate mapping, nuclei mask, and H&E image."""
        print(f"\nStep 1/6: Loading inputs...")

        # Coordinate mapping from coarse registration
        self.coord_mapping = pd.read_csv(self.coord_mapping_csv)
        print(f"  Coordinate mapping: {len(self.coord_mapping):,} MALDI pixels")
        print(f"  H&E x range: [{self.coord_mapping['he_x'].min():.0f}, "
              f"{self.coord_mapping['he_x'].max():.0f}]")
        print(f"  H&E y range: [{self.coord_mapping['he_y'].min():.0f}, "
              f"{self.coord_mapping['he_y'].max():.0f}]")

        # Nuclei mask
        self.nuclei_mask = load_nuclei_mask(self.nuclei_mask_path)

        # H&E image (optional, for visualisation)
        if self.he_image_path:
            self.he_image = load_he_image(self.he_image_path)
            if self.he_image is not None:
                print(f"  H&E image: {self.he_image.shape}")

    # ------------------------------------------------------------------
    def extract_cell_centroids(self):
        """
        Compute cell centroids, areas, and labels from the nuclei mask.
        Handles non-contiguous StarDist label sequences (no NaN crash).
        """
        print(f"\nStep 2/6: Extracting cell centroids...")

        mask = self.nuclei_mask

        # Handle binary vs instance-labelled masks
        if mask.max() <= 1:
            print("  Detected binary mask — labelling connected components...")
            labeled, _ = ndi.label(mask)
        else:
            labeled = mask.astype(int)

        # Use only labels that actually exist (StarDist NMS leaves gaps)
        present_labels = np.unique(labeled)
        present_labels = present_labels[present_labels > 0]
        print(f"  Cell labels found: {len(present_labels):,}")

        # Centroids (row, col) = (y, x)
        centroids_rc = np.array(
            ndi.center_of_mass(mask > 0, labeled, present_labels)
        )

        # Drop NaN/inf (safety net)
        valid = np.isfinite(centroids_rc).all(axis=1)
        n_invalid = (~valid).sum()
        if n_invalid > 0:
            print(f"  WARNING: {n_invalid} centroids with NaN/inf dropped")
        centroids_rc = centroids_rc[valid]
        present_labels = present_labels[valid]

        # Convert (row, col) -> (x, y)
        self.cell_centroids_xy = centroids_rc[:, [1, 0]]
        self.cell_labels       = present_labels

        # Cell areas (in H&E pixels)
        areas = ndi.sum(mask > 0, labeled, present_labels)
        self.cell_areas = np.array(areas)[valid]

        n_cells = len(self.cell_centroids_xy)
        print(f"  Cells extracted:  {n_cells:,}")
        print(f"  Centroid X range: [{self.cell_centroids_xy[:,0].min():.0f}, "
              f"{self.cell_centroids_xy[:,0].max():.0f}]")
        print(f"  Centroid Y range: [{self.cell_centroids_xy[:,1].min():.0f}, "
              f"{self.cell_centroids_xy[:,1].max():.0f}]")
        print(f"  Mean cell area:   {self.cell_areas.mean():.0f} H&E px² "
              f"({self.cell_areas.mean() * self.he_pixel_size_um**2:.0f} µm²)")

        # Coordinate space sanity check
        he_xmin = self.coord_mapping['he_x'].min()
        he_xmax = self.coord_mapping['he_x'].max()
        cx_min  = self.cell_centroids_xy[:,0].min()
        cx_max  = self.cell_centroids_xy[:,0].max()
        if he_xmax < cx_min or cx_max < he_xmin:
            raise ValueError(
                f"\nCoordinate space MISMATCH:\n"
                f"  H&E mapping x range: [{he_xmin:.0f}, {he_xmax:.0f}]\n"
                f"  Cell centroid x range: [{cx_min:.0f}, {cx_max:.0f}]\n"
                f"  These don't overlap — check registration and mask alignment."
            )
        print(f"  Coordinate spaces overlap — OK")

    # ------------------------------------------------------------------
    def build_psf_weight_matrix(self):
        """
        Build the sparse PSF weight matrix W of shape (n_pixels, n_cells).

        W[p, k] = exp(-0.5 * ||r_p - r_k||^2 / sigma^2)

        where r_p is the H&E coordinate of MALDI pixel p and
        r_k is the centroid of cell k (both in H&E pixel space).

        W is highly sparse: each pixel only has cells within search_radius_he.
        Uses scipy.sparse.lil_matrix for efficient construction then converts
        to CSR for fast arithmetic.

        Also normalises each row so weights sum to 1 (mixing proportions).
        """
        print(f"\nStep 3/6: Building PSF weight matrix W...")

        he_coords = self.coord_mapping[['he_x', 'he_y']].values
        n_pixels  = len(he_coords)
        n_cells   = len(self.cell_centroids_xy)
        print(f"  Matrix shape: ({n_pixels:,} pixels, {n_cells:,} cells)")
        print(f"  Building KDTree over cell centroids...")

        tree = KDTree(self.cell_centroids_xy)

        # Batch radius query — find all cells within search radius for each pixel
        neighbour_lists = tree.query_ball_point(
            he_coords, r=self.search_radius_he, workers=-1)

        print(f"  Filling sparse weight matrix...")
        W = lil_matrix((n_pixels, n_cells), dtype=np.float32)

        n_active_pixels  = 0
        n_empty_pixels   = 0
        cells_per_pixel  = []

        for p, (he_xy, nb_idx) in enumerate(zip(he_coords, neighbour_lists)):
            if len(nb_idx) < self.min_cells_per_pixel:
                n_empty_pixels += 1
                continue
            nearby = self.cell_centroids_xy[nb_idx]
            d      = np.linalg.norm(nearby - he_xy, axis=1)
            w      = np.exp(-0.5 * (d / self.psf_sigma_he)**2).astype(np.float32)
            w_sum  = w.sum()
            if w_sum < 1e-10:
                continue
            w /= w_sum   # normalise: mixing proportions sum to 1
            for j, cell_idx in enumerate(nb_idx):
                W[p, cell_idx] = w[j]
            n_active_pixels += 1
            cells_per_pixel.append(len(nb_idx))

        self.W                = W.tocsr()
        self.active_pixel_mask = np.array([len(nb) >= self.min_cells_per_pixel
                                           for nb in neighbour_lists])
        cells_per_pixel = np.array(cells_per_pixel)

        sparsity = 1.0 - (self.W.nnz / (n_pixels * n_cells))
        print(f"  Active pixels:   {n_active_pixels:,} / {n_pixels:,} "
              f"({100*n_active_pixels/n_pixels:.1f}%)")
        print(f"  Empty pixels:    {n_empty_pixels:,} (no cells in PSF footprint)")
        print(f"  Non-zeros in W:  {self.W.nnz:,}")
        print(f"  W sparsity:      {sparsity*100:.2f}%")
        if len(cells_per_pixel):
            print(f"  Cells/pixel:     mean={cells_per_pixel.mean():.1f}  "
                  f"median={np.median(cells_per_pixel):.0f}  "
                  f"max={cells_per_pixel.max()}")

    # ------------------------------------------------------------------
    def load_spectra(self, mz_min=None, mz_max=None, bin_resolution=0.1):
        """
        Load MALDI spectra from imzML and align to a common m/z axis.

        For processed imzML (already binned): reads directly.
        For continuous imzML: bins spectra to a common axis.

        Parameters
        ----------
        mz_min, mz_max : float or None
            m/z range to load. None = full range.
        bin_resolution : float
            m/z bin width in Da for continuous data (default 0.1 Da).
        """
        print(f"\nStep 4/6: Loading MALDI spectra from imzML...")
        if not IMZML_AVAILABLE:
            raise ImportError("pyimzml required. Install with: pip install pyimzml")

        parser = ImzMLParser(self.imzml_path)

        # Read all spectra to find common m/z axis
        print(f"  Reading spectra ({len(parser.coordinates)} pixels)...")
        all_mzs     = []
        all_intens  = []
        coord_list  = []

        for idx, coord in enumerate(parser.coordinates):
            mzs, intens = parser.getspectrum(idx)
            all_mzs.append(mzs)
            all_intens.append(intens)
            coord_list.append(coord[:2])  # (x, y) — 1-indexed

        coord_array = np.array(coord_list) - 1  # convert to 0-based

        # Determine m/z range
        global_mz_min = min(m.min() for m in all_mzs if len(m))
        global_mz_max = max(m.max() for m in all_mzs if len(m))
        if mz_min is None: mz_min = global_mz_min
        if mz_max is None: mz_max = global_mz_max
        print(f"  m/z range: [{mz_min:.2f}, {mz_max:.2f}] Da")

        # Check if data is already binned (all spectra same length + same m/z)
        lengths = [len(m) for m in all_mzs]
        is_binned = (len(set(lengths)) == 1 and
                     np.allclose(all_mzs[0], all_mzs[1]) if len(all_mzs) > 1 else True)

        if is_binned:
            print(f"  Data appears pre-binned ({lengths[0]} channels)")
            common_mz = all_mzs[0]
            # Filter to requested range
            mz_mask    = (common_mz >= mz_min) & (common_mz <= mz_max)
            self.mz_axis = common_mz[mz_mask]
            raw_matrix   = np.array([i[mz_mask] for i in all_intens],
                                     dtype=np.float32)
        else:
            print(f"  Continuous data — binning to {bin_resolution} Da resolution...")
            self.mz_axis = np.arange(mz_min, mz_max + bin_resolution, bin_resolution)
            n_bins = len(self.mz_axis)
            raw_matrix = np.zeros((len(all_mzs), n_bins), dtype=np.float32)
            for i, (mzs, intens) in enumerate(zip(all_mzs, all_intens)):
                # Find bin indices
                idx_arr = np.searchsorted(self.mz_axis, mzs)
                idx_arr = np.clip(idx_arr, 0, n_bins - 1)
                np.add.at(raw_matrix[i], idx_arr, intens)

        print(f"  Spectral matrix: {raw_matrix.shape[0]} pixels x "
              f"{raw_matrix.shape[1]} channels")

        # Match spectra to coordinate mapping
        # coord_mapping has (maldi_x, maldi_y) in 0-based coords
        mapping_xy = self.coord_mapping[['maldi_x', 'maldi_y']].values.astype(int)

        # Build lookup: (x, y) -> row index in raw_matrix
        coord_to_idx = {tuple(c): i for i, c in enumerate(coord_array)}

        # Align: for each row in coord_mapping, find corresponding spectrum
        aligned_matrix = np.zeros((len(mapping_xy), raw_matrix.shape[1]),
                                   dtype=np.float32)
        matched = 0
        for row, xy in enumerate(mapping_xy):
            key = tuple(xy)
            if key in coord_to_idx:
                aligned_matrix[row] = raw_matrix[coord_to_idx[key]]
                matched += 1

        print(f"  Spectra matched to mapping: {matched:,} / {len(mapping_xy):,}")
        if matched < len(mapping_xy) * 0.9:
            print(f"  WARNING: <90% of mapping pixels have matching spectra.")
            print(f"  Check that coordinate_mapping.csv and imzML are from the same scan.")

        self.spectra_matrix = aligned_matrix
        print(f"  Spectral range: [{self.mz_axis[0]:.2f}, {self.mz_axis[-1]:.2f}] Da  "
              f"({len(self.mz_axis)} channels)")

    # ------------------------------------------------------------------
    def solve_nnls(self, channel_batch_size=50):
        """
        Solve the non-negative least squares problem for each m/z channel.

        For each channel c:
            I_c = W * cell_c  +  epsilon_c
            cell_c_hat = argmin ||W * cell_c - I_c||^2  s.t. cell_c >= 0

        With Tikhonov regularisation (if self.regularisation > 0):
            Augmented system:  [W ; sqrt(lambda)*I] * c = [I ; 0]

        Parallelised across channels using joblib.

        Parameters
        ----------
        channel_batch_size : int
            Channels processed per batch in progress reporting.
        """
        print(f"\nStep 5/6: Solving NNLS deconvolution...")

        # Only use active pixels (those with cells in PSF footprint)
        active_mask = self.active_pixel_mask
        W_active    = self.W[active_mask]
        I_active    = self.spectra_matrix[active_mask]

        n_active, n_cells    = W_active.shape
        n_channels           = self.spectra_matrix.shape[1]

        print(f"  Active pixels:  {n_active:,}")
        print(f"  Cells:          {n_cells:,}")
        print(f"  Channels:       {n_channels:,}")
        print(f"  Regularisation: {self.regularisation}")

        # Convert to dense for NNLS (sparse NNLS is not directly available)
        # For very large problems, use lsqr on the sparse system instead
        use_sparse_solver = (n_active * n_cells > 5e8)

        if use_sparse_solver:
            print(f"  Problem too large for dense NNLS — using sparse LSQR solver")
            print(f"  (LSQR does not enforce non-negativity; clip negatives post-solve)")
            solver_fn = self._solve_channel_lsqr
        else:
            print(f"  Using dense NNLS solver (scipy.optimize.nnls)")
            W_dense = W_active.toarray()
            if self.regularisation > 0:
                # Augment with Tikhonov regularisation
                reg_rows = np.sqrt(self.regularisation) * np.eye(n_cells, dtype=np.float32)
                W_aug    = np.vstack([W_dense, reg_rows])
                print(f"  Augmented W: {W_aug.shape}")
            else:
                W_aug = W_dense
            solver_fn = lambda ch_idx: self._solve_channel_nnls(
                W_aug, I_active[:, ch_idx], n_cells)

        print(f"  Solving {n_channels} channels "
              f"({'parallel' if JOBLIB_AVAILABLE else 'sequential'})...")

        if JOBLIB_AVAILABLE and not use_sparse_solver:
            results = Parallel(n_jobs=self.n_jobs, verbose=0)(
                delayed(solver_fn)(ch) for ch in range(n_channels)
            )
            cell_profiles = np.array(results).T  # (n_cells, n_channels)
        else:
            cell_profiles = np.zeros((n_cells, n_channels), dtype=np.float32)
            for ch in range(n_channels):
                if ch % channel_batch_size == 0:
                    print(f"    Channel {ch}/{n_channels}...")
                if use_sparse_solver:
                    cell_profiles[:, ch] = self._solve_channel_lsqr(
                        W_active, I_active[:, ch], ch)
                else:
                    cell_profiles[:, ch] = solver_fn(ch)

        self.cell_profiles = cell_profiles

        # Compute residuals: W * c_hat - I  (for active pixels only)
        print(f"  Computing residuals...")
        I_reconstructed = W_active @ cell_profiles   # (n_active, n_channels)
        residuals_active = I_reconstructed - I_active
        self.residuals = residuals_active

        # Residual statistics
        rel_error = (np.abs(residuals_active).mean() /
                     (np.abs(I_active).mean() + 1e-10))
        print(f"\n  Deconvolution complete.")
        print(f"  Mean absolute residual: {np.abs(residuals_active).mean():.4f}")
        print(f"  Relative error:         {rel_error*100:.2f}%")
        print(f"  Cell profile matrix:    {cell_profiles.shape} "
              f"(cells x channels)")

    def _solve_channel_nnls(self, W_aug, I_col, n_cells):
        """Solve one channel with dense NNLS."""
        if self.regularisation > 0:
            # Augment observation vector with zeros for regularisation rows
            I_aug = np.concatenate([I_col, np.zeros(n_cells, dtype=np.float32)])
        else:
            I_aug = I_col
        try:
            c_hat, _ = nnls(W_aug, I_aug)
            return c_hat.astype(np.float32)
        except Exception:
            return np.zeros(n_cells, dtype=np.float32)

    def _solve_channel_lsqr(self, W_sparse, I_col, ch_idx=None):
        """Solve one channel with sparse LSQR (no NNLS constraint)."""
        result = lsqr(W_sparse, I_col, damp=np.sqrt(self.regularisation),
                      iter_lim=500, atol=1e-6, btol=1e-6)
        c_hat = np.clip(result[0], 0, None)  # clip negatives post-solve
        return c_hat.astype(np.float32)

    # ------------------------------------------------------------------
    def save_outputs(self):
        """
        Save per-cell metabolite profiles and metadata to CSV.
        """
        print(f"\nStep 6/6: Saving outputs...")

        n_cells, n_channels = self.cell_profiles.shape

        # ── Cell metabolite profiles ──
        channel_names = [f"mz_{mz:.4f}" for mz in self.mz_axis]
        profiles_df   = pd.DataFrame(
            self.cell_profiles,
            columns=channel_names
        )
        profiles_df.insert(0, 'cell_label', self.cell_labels)
        profiles_df.insert(1, 'centroid_x_he', self.cell_centroids_xy[:, 0])
        profiles_df.insert(2, 'centroid_y_he', self.cell_centroids_xy[:, 1])
        profiles_path = self.output_dir / 'cell_metabolite_profiles.csv'
        profiles_df.to_csv(profiles_path, index=False)
        print(f"  Saved cell profiles -> '{profiles_path}'")
        print(f"  Shape: {profiles_df.shape}  (cells x channels+metadata)")

        # ── Cell metadata ──
        # Find nearest MALDI pixel for each cell
        he_coords_mapping = self.coord_mapping[['he_x', 'he_y']].values
        tree_pixels = KDTree(he_coords_mapping)
        dist, idx   = tree_pixels.query(self.cell_centroids_xy)
        nearest_maldi_x = self.coord_mapping['maldi_x'].values[idx]
        nearest_maldi_y = self.coord_mapping['maldi_y'].values[idx]

        metadata_df = pd.DataFrame({
            'cell_label':       self.cell_labels,
            'centroid_x_he':    self.cell_centroids_xy[:, 0],
            'centroid_y_he':    self.cell_centroids_xy[:, 1],
            'cell_area_px':     self.cell_areas,
            'cell_area_um2':    self.cell_areas * self.he_pixel_size_um**2,
            'nearest_maldi_x':  nearest_maldi_x,
            'nearest_maldi_y':  nearest_maldi_y,
            'dist_to_nearest_maldi_px': dist,
            'dist_to_nearest_maldi_um': dist * self.he_pixel_size_um,
            'total_signal':     self.cell_profiles.sum(axis=1),
        })
        meta_path = self.output_dir / 'cell_metadata.csv'
        metadata_df.to_csv(meta_path, index=False)
        print(f"  Saved cell metadata -> '{meta_path}'")

        # ── m/z axis ──
        mz_df = pd.DataFrame({'channel_idx': range(n_channels),
                               'mz': self.mz_axis})
        mz_df.to_csv(self.output_dir / 'mz_axis.csv', index=False)

        return profiles_df, metadata_df

    # ------------------------------------------------------------------
    def plot_qc(self):
        """
        QC figure: residual distribution, coverage map, cells-per-pixel histogram.
        """
        print(f"  Generating QC figure...")
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Panel 1: residual distribution (summed across channels)
        if self.residuals is not None:
            res_sum = np.abs(self.residuals).mean(axis=1)
            axes[0].hist(res_sum, bins=60, color='steelblue',
                         edgecolor='white', linewidth=0.5)
            axes[0].axvline(np.median(res_sum), color='tomato',
                            linestyle='--', linewidth=2,
                            label=f'Median {np.median(res_sum):.3f}')
            axes[0].set_xlabel('Mean absolute residual per pixel')
            axes[0].set_ylabel('Pixel count')
            axes[0].set_title('Deconvolution residuals', fontweight='bold')
            axes[0].legend()

        # Panel 2: total deconvolved signal per cell (spatial map)
        if self.he_image is not None:
            # Downsample H&E for display
            scale = min(1.0, 2048 / max(self.he_image.shape[:2]))
            dh = int(self.he_image.shape[0] * scale)
            dw = int(self.he_image.shape[1] * scale)
            he_d = cv2.resize(self.he_image, (dw, dh), interpolation=cv2.INTER_AREA)
            axes[1].imshow(he_d, alpha=0.5)

        total_signal = self.cell_profiles.sum(axis=1)
        sc = axes[1].scatter(
            self.cell_centroids_xy[:, 0] * (scale if self.he_image is not None else 1),
            self.cell_centroids_xy[:, 1] * (scale if self.he_image is not None else 1),
            c=total_signal, cmap='hot', s=2, alpha=0.6,
            norm=mcolors.PowerNorm(gamma=0.4)
        )
        plt.colorbar(sc, ax=axes[1], label='Total deconvolved signal')
        axes[1].set_title('Total signal per cell', fontweight='bold')
        axes[1].axis('off')

        # Panel 3: cells per pixel histogram
        he_coords = self.coord_mapping[['he_x', 'he_y']].values
        tree = KDTree(self.cell_centroids_xy)
        n_cells_per_pixel = np.array([
            len(tree.query_ball_point(p, r=self.search_radius_he))
            for p in he_coords
        ])
        axes[2].hist(n_cells_per_pixel[n_cells_per_pixel > 0], bins=40,
                     color='mediumpurple', edgecolor='white', linewidth=0.5)
        axes[2].axvline(n_cells_per_pixel.mean(), color='gold',
                        linestyle='--', linewidth=2,
                        label=f'Mean {n_cells_per_pixel.mean():.1f}')
        axes[2].set_xlabel('Cells per MALDI pixel footprint')
        axes[2].set_ylabel('MALDI pixel count')
        axes[2].set_title('PSF footprint coverage', fontweight='bold')
        axes[2].legend()

        plt.tight_layout()
        qc_path = self.output_dir / 'deconvolution_qc.png'
        plt.savefig(str(qc_path), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved QC figure -> '{qc_path}'")

    def plot_metabolite_overlay(self, mz_targets, mz_tolerance=0.1,
                                 cmap='hot', alpha=0.7, dot_scale=3.0):
        """
        Plot per-cell deconvolved metabolite values overlaid on H&E.

        Parameters
        ----------
        mz_targets : list of float
            m/z values to visualise (e.g. [124.0074] for taurine).
        mz_tolerance : float
            m/z window in Da — averages all channels within ±tolerance.
        cmap : str
            Matplotlib colormap.
        alpha : float
            Dot opacity.
        dot_scale : float
            Dot size multiplier.
        """
        print(f"  Generating metabolite overlay plots...")

        # Downsample H&E for display
        scale = 1.0
        if self.he_image is not None:
            scale = min(1.0, 2048 / max(self.he_image.shape[:2]))
            dh = int(self.he_image.shape[0] * scale)
            dw = int(self.he_image.shape[1] * scale)
            he_d = cv2.resize(self.he_image, (dw, dh), interpolation=cv2.INTER_AREA)

        for mz_target in mz_targets:
            # Find channels within tolerance
            ch_mask = np.abs(self.mz_axis - mz_target) <= mz_tolerance
            if not ch_mask.any():
                print(f"  WARNING: no channels found near m/z={mz_target:.4f}")
                continue
            signal = self.cell_profiles[:, ch_mask].sum(axis=1)

            fig, ax = plt.subplots(1, 1, figsize=(12, 10))
            if self.he_image is not None:
                ax.imshow(he_d, alpha=0.5)

            # Dot size proportional to cell area
            cell_area_display = self.cell_areas * scale**2
            dot_sizes = np.clip(cell_area_display * dot_scale, 1, 50)

            sc = ax.scatter(
                self.cell_centroids_xy[:, 0] * scale,
                self.cell_centroids_xy[:, 1] * scale,
                c=signal, cmap=cmap, s=dot_sizes,
                alpha=alpha, linewidths=0,
                norm=mcolors.PowerNorm(gamma=0.5)
            )
            plt.colorbar(sc, ax=ax, label='Deconvolved intensity')
            ax.set_title(f'Deconvolved signal — m/z {mz_target:.4f} '
                         f'(±{mz_tolerance} Da, {ch_mask.sum()} channels)',
                         fontweight='bold')
            ax.axis('off')

            safe_mz = f"{mz_target:.4f}".replace('.', '_')
            out_path = self.output_dir / f'metabolite_overlay_mz{safe_mz}.png'
            plt.savefig(str(out_path), dpi=150, bbox_inches='tight')
            plt.close()
            print(f"    Saved -> '{out_path}'")

    # ------------------------------------------------------------------
    def run(self, mz_min=None, mz_max=None, mz_visualise=None, bin_resolution=0.1):
        """
        Run the full deconvolution pipeline.

        Parameters
        ----------
        mz_min, mz_max : float or None
            m/z range to process. None = full spectrum.
        mz_visualise : list of float or None
            m/z values to generate overlay visualisations for.
            None = skip overlay plots.
        bin_resolution : float
            m/z bin width in Da for continuous imzML data (default 0.1).
            Use a finer value (e.g. 0.0042) to preserve high-resolution peaks.
            Has no effect if the imzML is already binned.
        """
        self.load_inputs()
        self.extract_cell_centroids()
        self.build_psf_weight_matrix()
        self.load_spectra(mz_min=mz_min, mz_max=mz_max, bin_resolution=bin_resolution)
        self.solve_nnls()
        profiles_df, metadata_df = self.save_outputs()
        self.plot_qc()
        if mz_visualise:
            self.plot_metabolite_overlay(mz_visualise)

        SEP = "=" * 65
        print(f"\n{SEP}")
        print(f"DECONVOLUTION COMPLETE")
        print(f"{SEP}")
        print(f"  Output directory: {self.output_dir.resolve()}")
        print(f"  cell_metabolite_profiles.csv  — {self.cell_profiles.shape[0]:,} cells "
              f"x {self.cell_profiles.shape[1]:,} channels")
        print(f"  cell_metadata.csv")
        print(f"  mz_axis.csv")
        print(f"  deconvolution_qc.png")
        if mz_visualise:
            print(f"  metabolite_overlay_mz*.png  — {len(mz_visualise)} channels")
        print(f"{SEP}\n")
        return profiles_df, metadata_df


# ===========================================================================
#  ENTRY POINT
# ===========================================================================

if __name__ == "__main__":

    # ── Paths ──
    COORD_MAPPING_CSV = "registration_outputs/coordinate_mapping.csv"
    NUCLEI_MASK_PATH  = "high_res_MSI/D2_10x_originalExport_nuclei_masks.tif"
    IMZML_PATH        = "MSI_data_grant/Mass_Spec_data/20251012_old_liver.imzML"
    HE_IMAGE_PATH     = "high_res_MSI/D2_10x_originalExport.tif"  # optional
    OUTPUT_DIR        = "deconvolution_outputs"

    # ── Known m/z values to visualise (examples for liver tissue) ──
    # Taurine:   m/z 124.0074  (negative ion mode: 124.0069)
    # Glycine:   m/z  76.0393
    # ATP:       m/z 506.9958
    # Add your ions of interest here
    MZ_VISUALISE = [124.0074]

    # ── Run ──
    dec = SpatialDeconvolution(
        coord_mapping_csv=COORD_MAPPING_CSV,
        nuclei_mask_path=NUCLEI_MASK_PATH,
        imzml_path=IMZML_PATH,
        maldi_pixel_size_um=25.0,
        he_pixel_size_um=4.0,
        psf_sigma_um=None,       # None = derive from pixel size (FWHM = pixel)
        psf_truncate_sigma=3.0,  # search radius = 3 * sigma
        output_dir=OUTPUT_DIR,
        he_image_path=HE_IMAGE_PATH,
        n_jobs=-1,               # use all CPU cores for parallel NNLS
        min_cells_per_pixel=1,   # exclude pixels with no cells in footprint
        regularisation=0.01,     # Tikhonov regularisation (0 = pure NNLS)
    )

    profiles_df, metadata_df = dec.run(
        mz_min=None,             # None = full m/z range
        mz_max=None,
        bin_resolution=0.09,   # m/z bin width for continuous data (ignored if binned)
        mz_visualise=MZ_VISUALISE,
    )

    # ── Quick summary stats ──
    print("Per-cell profile summary (first 5 cells):")
    print(profiles_df.iloc[:5, :8].to_string())
    print(f"\nTotal signal statistics:")
    print(metadata_df['total_signal'].describe())
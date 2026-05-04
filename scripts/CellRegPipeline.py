"""
MALDI-MSI to H&E Registration Pipeline — Coarse + PSF-Aware Fine Alignment
============================================================================

PIPELINE OVERVIEW
-----------------
Stage 1 (Coarse) — Vessel-Landmark Guided Global Alignment:
  - Interactive landmark selection on vessel structures
  - Similarity transform (translation, rotation, scale — no shear)
  - Edge-correlation affine refinement
  - Thin-plate spline (TPS) non-rigid deformation
  - Output: coordinate_mapping.csv

Stage 2 (Fine) — PSF-Aware Nuclei Centroid Refinement:
  - Loads coarse H&E coordinates from Stage 1
  - Extracts nuclei centroids from H&E segmentation mask
  - For each MALDI pixel mapped to H&E space:
      * Finds all nuclei within the PSF footprint
        (Gaussian PSF, FWHM = MALDI pixel size, models instrument beam spread)
      * Computes PSF-weighted centroid as refined H&E coordinate
        (physically: the MALDI signal is a PSF-weighted mixture of all nuclei
         in the beam footprint, so the weighted centroid is the best location estimate)
  - Output: refined_coordinate_mapping.csv, final_overlay.png

RESOLUTION MISMATCH HANDLING
-----------------------------
MALDI pixel ~ 50 µm, H&E pixel ~ 0.5 µm → scale factor ~ 100x
Each MALDI pixel covers ~100x100 H&E pixels → 5–30 nuclei in liver tissue.
The Gaussian PSF weights closer nuclei more heavily, giving a sub-pixel
refined coordinate without requiring one-to-one nucleus matching.

WHY NOT DIFFEOMORPHIC (e.g., ANTs/LDDMM)?
------------------------------------------
Diffeomorphic methods require corresponding *dense* features in both modalities.
Since MALDI PCA produces no nuclear signal, there is no intensity basis for
voxel-level diffeomorphic matching. PSF-aware centroid refinement is the
correct approach here: it uses H&E nuclei as structural anchors and infers
where each MALDI pixel's "center of mass" most likely falls.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
from scipy import ndimage as ndi
from scipy.interpolate import RBFInterpolator
from scipy.optimize import minimize
from scipy.spatial import KDTree
from skimage import transform, filters
from pyimzml.ImzMLParser import ImzMLParser
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd
from stardist.models import StarDist2D
from csbdeep.utils import normalize
import warnings
import tifffile
warnings.filterwarnings('ignore')


# ===========================================================================
#  STAGE 1 — COARSE REGISTRATION  (Affine + TPS, original pipeline)
# ===========================================================================

class MALDIRegistration:
    """
    Vessel-landmark-guided coarse registration.
    Produces coordinate_mapping.csv consumed by Stage 2.
    """

    def __init__(self, he_path, maldi_path,
                 imzml_path="MSI_data_grant/Mass_Spec_data/20251012_old_liver.imzML"):
        self.parser = ImzMLParser(imzml_path)
        self.maldi_df = pd.DataFrame(
            (
                (*self.parser.getspectrum(idx), coord)
                for idx, coord in enumerate(self.parser.coordinates)
            ),
            columns=["mzs", "intensities", "coordinates"]
        )

        self.he_image = cv2.cvtColor(cv2.imread(he_path), cv2.COLOR_BGR2RGB)
        self.maldi_image = cv2.imread(maldi_path, cv2.IMREAD_UNCHANGED)
        if self.maldi_image.shape[2] == 4:
            self.maldi_image = cv2.cvtColor(self.maldi_image, cv2.COLOR_BGRA2RGBA)

        self.he_shape   = self.he_image.shape[:2]
        self.maldi_shape = self.maldi_image.shape[:2]

        maldi_rgb = self.maldi_image[:, :, :3]
        self.maldi_gray = (0.299 * maldi_rgb[:, :, 0] +
                           0.587 * maldi_rgb[:, :, 1] +
                           0.114 * maldi_rgb[:, :, 2]) / 255.0
        self.he_gray    = (0.299 * self.he_image[:, :, 0] +
                           0.587 * self.he_image[:, :, 1] +
                           0.114 * self.he_image[:, :, 2]) / 255.0

        self.he_landmarks       = []
        self.maldi_landmarks    = []
        self.affine_matrix      = None
        self.refined_affine     = None
        self.registered_affine  = None
        self.registered_nonrigid = None
        self.maldi_grid         = None
        self.displacement_field_x = None
        self.displacement_field_y = None
        self.rbf_x = None
        self.rbf_y = None

        print(f"Loaded H&E image: {self.he_shape}")
        print(f"Loaded MALDI image: {self.maldi_shape}")

    # ------------------------------------------------------------------
    def select_landmarks(self, n_points=5):
        """Interactive landmark selection on vessel structures."""
        print(f"\n{'='*60}\nLANDMARK SELECTION — click {n_points} vessel pairs\n{'='*60}")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        ax1.imshow(self.he_image);   ax1.set_title('H&E — Click FIRST',  fontweight='bold'); ax1.axis('off')
        ax2.imshow(self.maldi_image); ax2.set_title('MALDI — Click SECOND', fontweight='bold'); ax2.axis('off')

        current_image = 'he'; he_count = 0; maldi_count = 0

        def onclick(event):
            nonlocal current_image, he_count, maldi_count
            if event.inaxes is None: return
            x, y = event.xdata, event.ydata

            if event.inaxes == ax1 and current_image == 'he' and he_count < n_points:
                self.he_landmarks.append([x, y])
                ax1.plot(x, y, 'ro', markersize=10, markeredgecolor='white', markeredgewidth=2)
                ax1.text(x, y, str(he_count + 1), color='yellow', fontsize=12, fontweight='bold', ha='center', va='center')
                he_count += 1
                print(f"H&E landmark {he_count}/{n_points}: ({x:.1f}, {y:.1f})")
                if he_count == n_points:
                    current_image = 'maldi'
                    ax2.set_title('MALDI — CLICK NOW', fontweight='bold', color='red')

            elif event.inaxes == ax2 and current_image == 'maldi' and maldi_count < n_points:
                self.maldi_landmarks.append([x, y])
                ax2.plot(x, y, 'ro', markersize=10, markeredgecolor='white', markeredgewidth=2)
                ax2.text(x, y, str(maldi_count + 1), color='yellow', fontsize=12, fontweight='bold', ha='center', va='center')
                maldi_count += 1
                print(f"MALDI landmark {maldi_count}/{n_points}: ({x:.1f}, {y:.1f})")
                if maldi_count == n_points:
                    ax2.set_title('MALDI — COMPLETE! Close window.', fontweight='bold', color='green')
            fig.canvas.draw()

        fig.canvas.mpl_connect('button_press_event', onclick)
        plt.tight_layout(); plt.show()

        self.he_landmarks    = np.array(self.he_landmarks)
        self.maldi_landmarks = np.array(self.maldi_landmarks)
        print(f"Collected {len(self.he_landmarks)} landmark pairs")

    # ------------------------------------------------------------------
    def compute_affine_transform(self):
        """Compute similarity transform from landmarks (no shear)."""
        print(f"\nComputing affine transformation from landmarks...")
        if len(self.he_landmarks) < 3:
            raise ValueError(f"Need at least 3 landmark pairs, got {len(self.he_landmarks)}")

        tform = transform.SimilarityTransform()
        if not tform.estimate(self.maldi_landmarks, self.he_landmarks):
            raise RuntimeError("Failed to estimate affine transformation")

        self.affine_matrix = tform.params
        print(f"Affine matrix:\n{self.affine_matrix}")

        self.registered_affine = cv2.warpAffine(
            self.maldi_gray, self.affine_matrix[:2, :],
            (self.he_shape[1], self.he_shape[0]),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )
        print(f"Initial affine transformation applied")

    # ------------------------------------------------------------------
    def extract_tissue_mask(self, image, threshold=0.1):
        """Binary tissue mask with morphological cleanup."""
        mask = image > threshold
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return mask.astype(bool)

    # ------------------------------------------------------------------
    def refine_affine(self, downsample=8):
        """
        Refine affine by maximising edge-map correlation.

        Speed optimisation: the cost function is evaluated on a downsampled
        copy of both images (factor = downsample, default 8x).  For a
        12468x15297 H&E image this reduces each warpAffine call from ~190M
        pixels to ~3M pixels — roughly 60x faster per iteration.
        The translation parameters are rescaled accordingly so the final
        affine is always in full-resolution pixel coordinates.
        The final warp that stores registered_affine uses full resolution.

        Parameters
        ----------
        downsample : int
            Downscale factor for optimisation (default 8).
            Increase to 16 for even faster but slightly less precise results.
            Set to 1 to disable downsampling (original behaviour).
        """
        print(f"\nRefining affine transformation (downsample={downsample}x for speed)...")

        # ---- Downsample both images for the optimisation loop ----
        d = downsample
        he_small    = cv2.resize(self.he_gray,
                                  (self.he_shape[1] // d, self.he_shape[0] // d),
                                  interpolation=cv2.INTER_AREA)
        maldi_small = cv2.resize(self.maldi_gray,
                                  (self.maldi_shape[1] // d, self.maldi_shape[0] // d),
                                  interpolation=cv2.INTER_AREA)
        small_h, small_w = he_small.shape

        he_mask_small    = self.extract_tissue_mask(he_small)
        maldi_mask_small = self.extract_tissue_mask(maldi_small)
        he_edges_small   = filters.sobel(he_small * he_mask_small)

        # ---- Scale the initial affine params into downsampled space ----
        # The rotation/scale components are unchanged; only translation scales.
        a00, a01, a10, a11 = (self.affine_matrix[0, 0], self.affine_matrix[0, 1],
                               self.affine_matrix[1, 0], self.affine_matrix[1, 1])
        tx_small = self.affine_matrix[0, 2] / d
        ty_small = self.affine_matrix[1, 2] / d

        initial_params = np.array([a00, a01, a10, a11, tx_small, ty_small])

        call_count = [0]

        def cost(params):
            call_count[0] += 1
            affine_small = np.array([[params[0], params[1], params[4]],
                                     [params[2], params[3], params[5]]])
            warped = cv2.warpAffine(maldi_mask_small.astype(np.float32),
                                    affine_small, (small_w, small_h),
                                    flags=cv2.INTER_LINEAR)
            edges  = filters.sobel(warped)
            # Use normalised cross-correlation (faster than np.corrcoef)
            he_f = he_edges_small.ravel()
            ed_f = edges.ravel()
            he_std = he_f.std(); ed_std = ed_f.std()
            if he_std < 1e-8 or ed_std < 1e-8:
                return 1.0
            return -float(np.dot(he_f - he_f.mean(), ed_f - ed_f.mean()) /
                          (len(he_f) * he_std * ed_std))

        print(f"  Optimising on {small_w}x{small_h} px image "
              f"(full res: {self.he_shape[1]}x{self.he_shape[0]})...")

        # Powell converges faster than Nelder-Mead for this smooth objective
        result = minimize(cost, initial_params, method='Powell',
                          options={'maxiter': 200, 'ftol': 1e-5, 'disp': False})

        print(f"  Optimiser finished: {call_count[0]} cost evaluations, "
              f"correlation = {-result.fun:.4f}")

        rp = result.x
        # ---- Rescale translation back to full-resolution coordinates ----
        self.refined_affine = np.array([[rp[0], rp[1], rp[4] * d],
                                        [rp[2], rp[3], rp[5] * d],
                                        [0, 0, 1]])

        # ---- Final warp at full resolution ----
        self.registered_affine = cv2.warpAffine(
            self.maldi_gray, self.refined_affine[:2, :],
            (self.he_shape[1], self.he_shape[0]),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )
        print(f"Affine refinement complete — correlation: {-result.fun:.4f}")

    # ------------------------------------------------------------------
    def apply_nonrigid_deformation(self):
        """TPS non-rigid deformation from vessel landmarks."""
        print(f"\nApplying non-rigid TPS deformation...")

        maldi_lm_transformed = cv2.transform(
            self.maldi_landmarks.reshape(-1, 1, 2),
            self.refined_affine[:2, :]
        ).reshape(-1, 2)

        displacements = self.he_landmarks - maldi_lm_transformed

        self.rbf_x = RBFInterpolator(maldi_lm_transformed, displacements[:, 0],
                                      kernel='thin_plate_spline', smoothing=0.0)
        self.rbf_y = RBFInterpolator(maldi_lm_transformed, displacements[:, 1],
                                      kernel='thin_plate_spline', smoothing=0.0)

        y_coords, x_coords = np.mgrid[0:self.he_shape[0], 0:self.he_shape[1]]
        points = np.column_stack([x_coords.ravel(), y_coords.ravel()])

        print(f"Computing displacement field ({len(points)} points)...")
        dx = self.rbf_x(points).reshape(self.he_shape)
        dy = self.rbf_y(points).reshape(self.he_shape)

        self.displacement_field_x = dx
        self.displacement_field_y = dy

        map_x = (x_coords - dx).astype(np.float32)
        map_y = (y_coords - dy).astype(np.float32)

        self.registered_nonrigid = cv2.remap(
            self.registered_affine, map_x, map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )
        print(f"Non-rigid deformation complete")

    # ------------------------------------------------------------------
    def transform_maldi_to_he_coordinates(self, maldi_coords):
        """Apply affine + TPS to map MALDI pixel coords → H&E space."""
        if self.refined_affine is None or self.rbf_x is None:
            raise RuntimeError("Complete registration pipeline first.")

        maldi_coords = np.atleast_2d(maldi_coords).copy()
        maldi_coords[:, 0] = np.clip(maldi_coords[:, 0], 0, self.maldi_shape[1] - 1)
        maldi_coords[:, 1] = np.clip(maldi_coords[:, 1], 0, self.maldi_shape[0] - 1)

        hom = np.column_stack([maldi_coords, np.ones(len(maldi_coords))])
        affine_coords = (self.refined_affine @ hom.T).T[:, :2]

        dx = self.rbf_x(affine_coords)
        dy = self.rbf_y(affine_coords)
        return affine_coords + np.column_stack([dx, dy])

    def transform_he_to_maldi_coordinates(self, he_coords, max_iterations=50, tolerance=0.5):
        """Inverse coordinate transform (iterative Newton-like solver)."""
        if self.refined_affine is None or self.rbf_x is None:
            raise RuntimeError("Complete registration pipeline first.")

        he_coords  = np.atleast_2d(he_coords)
        maldi_coords = np.zeros_like(he_coords)
        affine_inv = np.linalg.inv(self.refined_affine)

        for i, target in enumerate(he_coords):
            guess = (affine_inv @ np.append(target, 1))[:2]
            for _ in range(max_iterations):
                predicted = self.transform_maldi_to_he_coordinates(guess.reshape(1, -1))[0]
                err = np.linalg.norm(predicted - target)
                if err < tolerance: break
                guess -= 0.5 * (predicted - target)
            maldi_coords[i] = guess
        return maldi_coords

    # ------------------------------------------------------------------
    def create_coordinate_mapping_grid(self, grid_spacing=1, tissue_only=True,
                                        intensity_threshold=0.1):
        """Build the MALDI→H&E coordinate mapping (saved to CSV)."""
        print(f"Creating coordinate mapping grid (spacing={grid_spacing})...")

        if tissue_only:
            tissue_coords = np.asarray([c[:2] for c in self.maldi_df['coordinates']])
            maldi_grid = tissue_coords if grid_spacing == 1 else tissue_coords[::grid_spacing]
        else:
            y = np.arange(0, self.maldi_shape[0], grid_spacing)
            x = np.arange(0, self.maldi_shape[1], grid_spacing)
            xv, yv = np.meshgrid(x, y)
            maldi_grid = np.column_stack([xv.ravel(), yv.ravel()])

        print(f"  Transforming {len(maldi_grid)} coordinates...")
        he_grid = self.transform_maldi_to_he_coordinates(maldi_grid)
        self.maldi_grid = maldi_grid

        mapping_df = pd.DataFrame({
            'maldi_x': maldi_grid[:, 0], 'maldi_y': maldi_grid[:, 1],
            'he_x':    he_grid[:, 0],    'he_y':    he_grid[:, 1]
        })
        print(f"  Generated {len(mapping_df)} coordinate mappings")
        return mapping_df

    def save_coordinate_mapping(self, output_path='coordinate_mapping.csv',
                                 grid_spacing=1, tissue_only=True, intensity_threshold=0.1):
        mapping_df = self.create_coordinate_mapping_grid(grid_spacing, tissue_only, intensity_threshold)
        mapping_df.to_csv(output_path, index=False)
        print(f"Saved coordinate mapping to '{output_path}' ({len(mapping_df)} rows)")
        return mapping_df

    # ------------------------------------------------------------------
    def visualize_results(self):
        """Four-panel coarse registration summary figure."""
        print(f"\nGenerating coarse-registration visualisation...")
        fig, axes = plt.subplots(2, 2, figsize=(16, 16))

        def overlay(a, b, alpha=0.5):
            a = (a - a.min()) / (a.max() - a.min() + 1e-8)
            b = (b - b.min()) / (b.max() - b.min() + 1e-8)
            return alpha * a + (1 - alpha) * b

        maldi_r = cv2.resize(self.maldi_gray, (self.he_shape[1], self.he_shape[0]))
        axes[0, 0].imshow(overlay(self.he_gray, maldi_r), cmap='gray')
        axes[0, 0].set_title('Original (No Registration)', fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(self.he_image)
        if self.refined_affine is not None:
            mt = cv2.transform(self.maldi_landmarks.reshape(-1, 1, 2),
                               self.refined_affine[:2, :]).reshape(-1, 2)
            for i in range(len(self.he_landmarks)):
                axes[0, 1].plot([self.he_landmarks[i, 0], mt[i, 0]],
                                [self.he_landmarks[i, 1], mt[i, 1]], 'y-', lw=2, alpha=0.6)
                axes[0, 1].plot(*self.he_landmarks[i], 'ro', markersize=10,
                                markeredgecolor='white', markeredgewidth=2)
                axes[0, 1].plot(*mt[i], 'bo', markersize=10,
                                markeredgecolor='white', markeredgewidth=2)
        axes[0, 1].set_title('Landmark Correspondence', fontsize=14, fontweight='bold')
        axes[0, 1].axis('off')

        if self.registered_affine is not None:
            axes[1, 0].imshow(overlay(self.he_gray, self.registered_affine), cmap='gray')
            axes[1, 0].set_title('After Affine Registration', fontsize=14, fontweight='bold')
            axes[1, 0].axis('off')

        if self.registered_nonrigid is not None:
            axes[1, 1].imshow(overlay(self.he_gray, self.registered_nonrigid), cmap='gray')
            axes[1, 1].set_title('After Non-Rigid (TPS) Deformation', fontsize=14, fontweight='bold')
            axes[1, 1].axis('off')

        plt.tight_layout()
        plt.savefig('registration_results.png', dpi=150, bbox_inches='tight')
        print(f"Saved 'registration_results.png'")
        plt.show()

    def save_registered_image(self, output_path='registered_maldi.tif'):
        if self.registered_nonrigid is not None:
            cv2.imwrite(output_path, (self.registered_nonrigid * 255).astype(np.uint8))
            print(f"Saved registered image to '{output_path}'")

    def visualize_coordinate_mapping(self):
        """Interactive Plotly displacement-vector figure."""
        print(f"\nGenerating coordinate mapping visualisation...")
        tissue_coords = np.asarray([c[:2] for c in self.maldi_df['coordinates']])
        he_grid       = self.transform_maldi_to_he_coordinates(tissue_coords)
        he_grid_x, he_grid_y = he_grid[:, 0], he_grid[:, 1]

        maldi_hom  = np.column_stack([self.maldi_grid, np.ones(len(self.maldi_grid))])
        affine_only = (self.refined_affine @ maldi_hom.T).T[:, :2]
        displacement     = he_grid - affine_only
        displacement_mag = np.linalg.norm(displacement, axis=1)
        max_displacement = displacement_mag.max()

        subsample_indices = np.random.choice(len(he_grid_x), size=len(he_grid_x)//2, replace=False)

        fig = make_subplots(rows=1, cols=2,
            subplot_titles=('MALDI Grid in H&E Space',
                            f'Non-Rigid Displacement Vectors (max {max_displacement:.1f} px)'),
            horizontal_spacing=0.1)

        fig.add_trace(go.Image(z=self.he_image), row=1, col=1)
        fig.add_trace(go.Scatter(x=he_grid_x[subsample_indices], y=he_grid_y[subsample_indices],
            mode='markers', marker=dict(color='blue', size=3, opacity=0.1),
            name='MALDI points'), row=1, col=1)
        fig.add_trace(go.Scatter(x=self.he_landmarks[:, 0], y=self.he_landmarks[:, 1],
            mode='markers', marker=dict(color='green', size=8,
            line=dict(color='white', width=2)), name='H&E landmarks'), row=1, col=1)

        fig.add_trace(go.Image(z=self.he_image), row=1, col=2)
        mask = displacement_mag > 1.0
        if np.any(mask):
            n_vec = min(500, mask.sum())
            sel   = np.random.choice(np.where(mask)[0], n_vec, replace=False)
            vm    = np.zeros(len(mask), dtype=bool); vm[sel] = True
            norm_d = (displacement_mag[vm] - displacement_mag[vm].min()) / \
                     (displacement_mag[vm].max() - displacement_mag[vm].min() + 1e-10)
            cmap   = cm.get_cmap('Spectral')
            colors = [f'rgb({int(r*255)},{int(g*255)},{int(b*255)})' for r,g,b,_ in cmap(norm_d)]
            for idx, i in enumerate(np.where(vm)[0]):
                fig.add_trace(go.Scatter(
                    x=[affine_only[i,0], he_grid[i,0]], y=[affine_only[i,1], he_grid[i,1]],
                    mode='lines', line=dict(color=colors[idx], width=2), showlegend=False), row=1, col=2)
            fig.add_trace(go.Scatter(x=he_grid[vm,0], y=he_grid[vm,1], mode='markers',
                marker=dict(color=displacement_mag[vm], colorscale='Spectral', size=6,
                    symbol='arrow', colorbar=dict(title='Displacement (px)', x=1.15, len=0.5, y=0.5),
                    showscale=True), name='Displacement'), row=1, col=2)

        fig.update_layout(height=700, width=1600, title_text='Coordinate Mapping Visualisation',
                          showlegend=True, hovermode='closest')
        fig.write_html('coordinate_mapping_accuracy.html')
        fig.show()
        print(f"Saved interactive figure to 'coordinate_mapping_accuracy.html'")

    # ====================================================================
    #  STAGE 2 ENTRY POINT — called at the end of the pipeline
    # ====================================================================

    def run_psf_nuclei_refinement(self,
                                   nuclei_mask,
                                   maldi_pixel_size_um=50.0,
                                   he_pixel_size_um=0.5,
                                   assignment_mode='soft',
                                   search_radius_sigma=3.0,
                                   min_nuclei=1,
                                   output_path='refined_coordinate_mapping.csv'):
        """
        Stage 2: PSF-aware nuclei centroid refinement — runs in-place after Stage 1.

        Seamlessly picks up the coordinate_mapping.csv produced by Stage 1 and
        refines each coarse H&E coordinate using nearby nuclei centroids weighted
        by a Gaussian PSF that models the MALDI instrument beam footprint.

        Parameters
        ----------
        nuclei_mask : ndarray (H_he, W_he), int or bool
            Output of your nuclei segmentation pipeline.
            Binary (0/1) or instance-labelled (each nucleus gets a unique int).
        maldi_pixel_size_um : float
            Physical MALDI pixel size in µm (default 50 µm).
        he_pixel_size_um : float
            Physical H&E pixel size in µm (default 0.5 µm for 10× digital slide).
        assignment_mode : {'soft', 'hard', 'mixed'}
            'soft'  — PSF-weighted centroid of all nuclei in footprint (recommended).
            'hard'  — snap to single nearest nucleus centroid.
            'mixed' — confidence-blended: confident → soft centroid, else keep coarse.
        search_radius_sigma : float
            Search radius expressed as multiples of PSF sigma (default 3×).
        min_nuclei : int
            Minimum nuclei in footprint required to update a coordinate.
            If fewer found, the coarse coordinate is retained unchanged.
        output_path : str
            CSV output path for the refined mapping.

        Returns
        -------
        refined_df : pd.DataFrame
            Refined coordinate mapping with columns:
            maldi_x, maldi_y, he_x_coarse, he_y_coarse,
            he_x_refined, he_y_refined,
            refinement_dx, refinement_dy, refinement_magnitude, confidence
        """
        if self.maldi_grid is None:
            raise RuntimeError(
                "Run save_coordinate_mapping() (Stage 1) before PSF refinement."
            )

        # Retrieve the coarse mapping produced in Stage 1
        coarse_df = self.create_coordinate_mapping_grid(grid_spacing=1, tissue_only=True)

        refiner = PSFNucleiRefinement(
            coord_mapping_df=coarse_df,
            he_image=self.he_image,
            maldi_image=self.maldi_image,
            maldi_pixel_size_um=maldi_pixel_size_um,
            he_pixel_size_um=he_pixel_size_um
        )
        refiner.extract_nuclei_centroids(nuclei_mask)
        refined_df = refiner.compute_psf_weighted_centroids(
            search_radius_sigma=search_radius_sigma,
            assignment_mode=assignment_mode,
            min_nuclei=min_nuclei
        )
        refiner.save_refined_mapping(output_path)
        refiner.visualize_refinement()
        refiner.generate_final_overlay()

        self.psf_refiner = refiner   # keep reference for post-hoc inspection
        return refined_df


# ===========================================================================
#  STAGE 2 — PSF-AWARE NUCLEI CENTROID REFINEMENT
# ===========================================================================

class PSFNucleiRefinement:
    """
    PSF-aware refinement of coarse MALDI→H&E coordinate mapping.

    Can be instantiated either:
    (a) inline — from MALDIRegistration.run_psf_nuclei_refinement()
    (b) standalone — by loading coordinate_mapping.csv from disk

    Physical motivation
    -------------------
    MALDI signal at pixel p is the PSF-convolved sum of metabolite
    concentrations from all cells in the beam footprint:

        I(p) = ∫ C(r) · PSF(p - r) dr   (continuous)
             ≈ Σ_k  C(r_k) · G(||p - r_k|| / σ)   (discrete nuclei)

    The PSF-weighted centroid:
        r_refined(p) = Σ_k r_k · G(...) / Σ_k G(...)

    is therefore the best single-point location estimate for pixel p's
    biological origin — without needing nuclear signal in MALDI at all.
    """

    def __init__(self,
                 coord_mapping_path=None,
                 coord_mapping_df=None,
                 he_image=None,
                 maldi_image=None,
                 he_image_path=None,
                 maldi_image_path=None,
                 maldi_pixel_size_um=50.0,
                 he_pixel_size_um=0.5):
        """
        Parameters
        ----------
        coord_mapping_path : str, optional
            Path to coordinate_mapping.csv (Stage 1 output).
        coord_mapping_df : pd.DataFrame, optional
            In-memory coarse mapping (used when called from MALDIRegistration).
        he_image : ndarray, optional
            H&E image array (RGB). Passed directly from Stage 1.
        maldi_image : ndarray, optional
            MALDI image array (RGBA). Passed directly from Stage 1.
        he_image_path : str, optional
            Path to H&E image (standalone use).
        maldi_image_path : str, optional
            Path to MALDI image (standalone use).
        maldi_pixel_size_um : float
            Physical MALDI pixel size in µm.
        he_pixel_size_um : float
            Physical H&E pixel size in µm.
        """
        # Load coordinate mapping
        if coord_mapping_df is not None:
            self.coord_mapping = coord_mapping_df.copy()
        elif coord_mapping_path is not None:
            self.coord_mapping = pd.read_csv(coord_mapping_path)
        else:
            raise ValueError("Provide either coord_mapping_path or coord_mapping_df.")

        self.maldi_pixel_size_um = maldi_pixel_size_um
        self.he_pixel_size_um    = he_pixel_size_um

        # Scale: how many H&E pixels span one MALDI pixel
        self.scale_factor = maldi_pixel_size_um / he_pixel_size_um  # e.g. 100

        # Gaussian PSF sigma in H&E pixels
        # σ = FWHM / (2√(2 ln 2)) ≈ FWHM / 2.355
        # FWHM is taken as one MALDI pixel (beam limited by pixel step)
        self.psf_sigma_he = self.scale_factor / 2.355


        # FIX 1 — after loading the CSV, convert the H&E coordinates from
        # coarse-registration image-pixel space into the same space the
        # nuclei centroids live in.  The coarse pipeline stores he_x / he_y
        # already in H&E image pixels (origin = top-left), so no rescaling
        # is needed — BUT the maldi_x / maldi_y are raw imzML spot indices
        # (1-indexed).  Convert them to 0-based here so array lookups work.
        self.coord_mapping['maldi_x'] -= 1
        self.coord_mapping['maldi_y'] -= 1

        # FIX 2 — do NOT swap x/y here; the coarse pipeline already stores
        # he_x = column (horizontal) and he_y = row (vertical), matching
        # OpenCV / numpy image convention.  Swapping was the cause of the
        # top-left collapse.
        #
        # Sanity-print so you can verify the ranges look like real image coords
        print(f"  H&E coord ranges from CSV:")
        print(f"    he_x: [{self.coord_mapping['he_x'].min():.0f}, "
              f"{self.coord_mapping['he_x'].max():.0f}]")
        print(f"    he_y: [{self.coord_mapping['he_y'].min():.0f}, "
              f"{self.coord_mapping['he_y'].max():.0f}]")
        print(f"  These should match your H&E image dimensions: {he_image.shape[:2] if he_image is not None else 'load image to check'}")
        # Load images (prefer passed arrays over paths)
        if he_image is not None:
            self.he_image = he_image
        elif he_image_path:
            self.he_image = cv2.cvtColor(cv2.imread(he_image_path), cv2.COLOR_BGR2RGB)
        else:
            self.he_image = None

        if maldi_image is not None:
            self.maldi_image = maldi_image
        elif maldi_image_path:
            raw = cv2.imread(maldi_image_path, cv2.IMREAD_UNCHANGED)
            self.maldi_image = cv2.cvtColor(raw, cv2.COLOR_BGRA2RGBA) if raw.shape[2] == 4 else raw
        else:
            self.maldi_image = None

        # Filled later
        self.nuclei_centroids_xy = None
        self.nuclei_tree         = None
        self.refined_mapping     = None

        print(f"\n{'='*60}")
        print(f"STAGE 2 — PSF-Aware Nuclei Centroid Refinement")
        print(f"{'='*60}")
        print(f"  Coarse mapping loaded: {len(self.coord_mapping)} MALDI pixels")
        print(f"  MALDI pixel size:  {maldi_pixel_size_um} µm")
        print(f"  H&E pixel size:    {he_pixel_size_um} µm")
        print(f"  Scale factor:      {self.scale_factor:.1f} H&E px / MALDI px")
        print(f"  PSF sigma:         {self.psf_sigma_he:.1f} H&E px  "
              f"({self.psf_sigma_he * he_pixel_size_um:.1f} µm)")
        print(f"  PSF FWHM:          {self.scale_factor:.1f} H&E px  "
              f"({maldi_pixel_size_um:.1f} µm)")

    # ------------------------------------------------------------------
    def extract_nuclei_centroids(self, nuclei_mask):
        print(f"\nExtracting nuclei centroids...")
        print(f"  Mask shape: {nuclei_mask.shape}")
        print(f"  Mask dtype: {nuclei_mask.dtype}")
        print(f"  Unique labels: {np.unique(nuclei_mask).size} (including background 0)")

        if nuclei_mask.max() <= 1:
            labeled_mask, n_nuclei = ndi.label(nuclei_mask)
        else:
            labeled_mask = nuclei_mask.astype(int)
            n_nuclei     = int(labeled_mask.max())

        if n_nuclei == 0:
            raise ValueError("No nuclei found in mask — check segmentation output.")
        print(f"  Label count (before filtering): {n_nuclei}")

        # Only compute centroids for labels that actually exist in the mask.
        # StarDist can produce non-contiguous label sequences (e.g. 1,2,5,7...)
        # which causes center_of_mass to return NaN for missing labels.
        present_labels = np.unique(labeled_mask)
        present_labels = present_labels[present_labels > 0]   # drop background (0)
        print(f"  Labels present in mask: {len(present_labels)}")

        centroids_rc = np.array(
            ndi.center_of_mass(nuclei_mask > 0, labeled_mask, present_labels)
        )  # shape (N, 2), values are (row, col) = (y, x)

        # Convert (row, col) → (x, y)
        centroids_xy = centroids_rc[:, [1, 0]]

        # Defensive filter: drop any NaN/inf that slipped through
        valid = np.isfinite(centroids_xy).all(axis=1)
        n_invalid = (~valid).sum()
        if n_invalid > 0:
            print(f"  WARNING: dropping {n_invalid} centroids with NaN/inf coordinates")
            centroids_xy = centroids_xy[valid]

        if len(centroids_xy) == 0:
            raise ValueError("All centroids were NaN/inf — something is wrong with the mask.")

        self.nuclei_centroids_xy = centroids_xy
        self.nuclei_tree = KDTree(self.nuclei_centroids_xy)

        print(f"  Final centroid count: {len(self.nuclei_centroids_xy)}")
        print(f"  X range: [{centroids_xy[:,0].min():.1f}, {centroids_xy[:,0].max():.1f}]")
        print(f"  Y range: [{centroids_xy[:,1].min():.1f}, {centroids_xy[:,1].max():.1f}]")
        return self.nuclei_centroids_xy

    # ------------------------------------------------------------------
    def compute_psf_weighted_centroids(self,
                                        search_radius_sigma=3.0,
                                        assignment_mode='soft',
                                        min_nuclei=1):
        """
        Refine each coarse H&E coordinate using PSF-weighted nuclei centroids.

        For each MALDI pixel coarsely mapped to H&E position p:
          1. Find all nuclei centroids r_k within radius = search_radius_sigma × σ
          2. Weight: w_k = exp(−‖r_k − p‖² / 2σ²)
          3. Refined position: p* = Σ w_k r_k / Σ w_k

        Parameters
        ----------
        search_radius_sigma : float
            Search radius in units of PSF sigma (default 3σ captures 99.7% of Gaussian).
        assignment_mode : {'soft', 'hard', 'mixed'}
            See class docstring for details.
        min_nuclei : int
            Minimum nuclei required within footprint to update coordinate.

        Returns
        -------
        refined_mapping : pd.DataFrame
        """
        if self.nuclei_tree is None:
            raise RuntimeError("Call extract_nuclei_centroids() first.")

        coarse_he = self.coord_mapping[['he_x', 'he_y']].values.copy()

        coarse_he = self.coord_mapping[['he_x', 'he_y']].values.copy()

        # FIX 3 — the search radius must be large enough to cover the physical
        # MALDI footprint.  With scale_factor=6.25, psf_sigma≈2.66 H&E px, and
        # 3σ ≈ 8 px.  That is correct for sub-pixel snap — but if the coarse
        # alignment still has systematic offsets larger than 8 px, increase
        # search_radius_sigma.  Recommended starting value for 25µm/4µm: 5.0
        # which gives a 13 px search window (52 µm — one full MALDI pixel width).
        search_r = search_radius_sigma * self.psf_sigma_he
        print(f"  Search radius: {search_r:.1f} H&E px ({search_r * self.he_pixel_size_um:.1f} µm)")

        # FIX 4 — verify nuclei centroids and coarse H&E coords are in the
        # same space before querying the KD-tree.
        he_xmin, he_xmax = coarse_he[:, 0].min(), coarse_he[:, 0].max()
        nc_xmin, nc_xmax = self.nuclei_centroids_xy[:, 0].min(), self.nuclei_centroids_xy[:, 0].max()
        print(f"  Coarse H&E x range:   [{he_xmin:.0f}, {he_xmax:.0f}]")
        print(f"  Nuclei centroid x range: [{nc_xmin:.0f}, {nc_xmax:.0f}]")
        if he_xmax < nc_xmin or nc_xmax < he_xmin:
            raise ValueError(
                "Coarse H&E coordinates and nuclei centroids do not overlap at all.\n"
                "This means they are in different coordinate spaces.\n"
                "Check: (1) are he_x/he_y in H&E image pixels? "
                "(2) are nuclei centroids computed on the same H&E image?"
            )


        print(f"\nRunning PSF-weighted centroid refinement...")
        print(f"  Assignment mode:   {assignment_mode}")
        print(f"  σ (PSF sigma):     {self.psf_sigma_he:.1f} H&E px")
        print(f"  Search radius:     {search_r:.1f} H&E px  "
              f"({search_r * self.he_pixel_size_um:.1f} µm)")
        print(f"  MALDI pixels:      {len(coarse_he)}")

        refined_he   = np.copy(coarse_he)
        confidence   = np.zeros(len(coarse_he))
        n_updated    = 0
        n_no_nuclei  = 0

        # Batch KD-tree query — returns list of neighbour index lists
        print(f"  Querying KD-tree (batch radius search)...")
        neighbour_lists = self.nuclei_tree.query_ball_point(coarse_he, r=search_r,
                                                             workers=-1)

        for i, (p, nb_idx) in enumerate(zip(coarse_he, neighbour_lists)):
            if len(nb_idx) < min_nuclei:
                n_no_nuclei += 1
                continue

            nearby = self.nuclei_centroids_xy[nb_idx]             # (K, 2)
            d      = np.linalg.norm(nearby - p, axis=1)           # (K,)
            w      = np.exp(-0.5 * (d / self.psf_sigma_he) ** 2) # Gaussian weights

            if assignment_mode == 'hard':
                refined_he[i]  = nearby[np.argmin(d)]
                confidence[i]  = 1.0

            elif assignment_mode == 'soft':
                w_sum = w.sum()
                if w_sum > 1e-12:
                    refined_he[i] = (w[:, None] * nearby).sum(axis=0) / w_sum
                    # Effective number of contributing nuclei relative to search area
                    confidence[i] = min(1.0, w_sum / max(len(nb_idx), 1))

            elif assignment_mode == 'mixed':
                w_sum = w.sum()
                if w_sum > 1e-12:
                    centroid    = (w[:, None] * nearby).sum(axis=0) / w_sum
                    conf        = min(1.0, w_sum / max(len(nb_idx), 1))
                    refined_he[i] = conf * centroid + (1 - conf) * p
                    confidence[i] = conf

            n_updated += 1

        disp     = refined_he - coarse_he
        disp_mag = np.linalg.norm(disp, axis=1)

        print(f"\n  Results:")
        print(f"    Updated:     {n_updated}/{len(coarse_he)} pixels "
              f"({100*n_updated/len(coarse_he):.1f}%)")
        print(f"    No nuclei:   {n_no_nuclei} pixels "
              f"({100*n_no_nuclei/len(coarse_he):.1f}%)")
        valid_conf = confidence[confidence > 0]
        if len(valid_conf):
            print(f"    Mean conf:   {valid_conf.mean():.3f}")
        print(f"    Median shift: {np.median(disp_mag[disp_mag>0]):.2f} H&E px")
        print(f"    Max shift:    {disp_mag.max():.2f} H&E px")

        self.refined_mapping = pd.DataFrame({
            'maldi_x':              self.coord_mapping['maldi_x'].values,
            'maldi_y':              self.coord_mapping['maldi_y'].values,
            'he_x_coarse':          coarse_he[:, 0],
            'he_y_coarse':          coarse_he[:, 1],
            'he_x_refined':         refined_he[:, 0],
            'he_y_refined':         refined_he[:, 1],
            'refinement_dx':        disp[:, 0],
            'refinement_dy':        disp[:, 1],
            'refinement_magnitude': disp_mag,
            'confidence':           confidence
        })
        return self.refined_mapping

    # ------------------------------------------------------------------
    def save_refined_mapping(self, output_path='refined_coordinate_mapping.csv'):
        """Persist the refined coordinate mapping."""
        if self.refined_mapping is None:
            raise RuntimeError("Run compute_psf_weighted_centroids() first.")
        self.refined_mapping.to_csv(output_path, index=False)
        print(f"\nSaved refined mapping → '{output_path}'  ({len(self.refined_mapping)} rows)")
        return self.refined_mapping

    # ------------------------------------------------------------------
    def generate_final_overlay(self, output_path='final_overlay.png', alpha=0.5):
        """
        Overlay in H&E space — each H&E pixel coloured by its corresponding
        MALDI Taurine intensity, blended with the H&E image.

        This gives a full-resolution output at H&E pixel size (4 µm) rather
        than the coarse MALDI resolution (25 µm).
        """
        if self.refined_mapping is None:
            raise RuntimeError("Run compute_psf_weighted_centroids() first.")
        if self.he_image is None or self.maldi_image is None:
            raise RuntimeError("Provide he_image and maldi_image in constructor.")

        print(f"\nGenerating final overlay (H&E-space canvas)...")

        he_h, he_w       = self.he_image.shape[:2]
        maldi_h, maldi_w = self.maldi_image.shape[:2]

        df   = self.refined_mapping
        conf = np.maximum(df['confidence'].values, 0.05)

        # imzML coordinates are 1-indexed — convert to 0-based array indices
        mx = np.clip(df['maldi_x'].values.astype(int) - 1, 0, maldi_w - 1)
        my = np.clip(df['maldi_y'].values.astype(int) - 1, 0, maldi_h - 1)

        # Refined H&E coordinates (float → int for canvas placement)
        hx = np.clip(df['he_x_refined'].values, 0, he_w - 1).astype(int)
        hy = np.clip(df['he_y_refined'].values, 0, he_h - 1).astype(int)

        # Sample the MALDI Taurine intensity for each mapped pixel
        maldi_rgb      = self.maldi_image[:, :, :3].astype(np.float32)
        maldi_colors   = maldi_rgb[my, mx]   # shape (N, 3) — MALDI colour at each spot

        # Build a MALDI-signal canvas in H&E space
        # Each H&E pixel gets the MALDI colour of the MALDI pixel it was assigned to
        maldi_canvas  = np.zeros((he_h, he_w, 3), dtype=np.float64)
        weight_canvas = np.zeros((he_h, he_w),    dtype=np.float64)

        # One MALDI pixel covers ~6.25×6.25 H&E pixels — paint a small square
        # so there are no gaps in the H&E-space canvas
        half = max(1, int(np.ceil(self.scale_factor / 2)))

        for dy in range(-half, half + 1):
            for dx in range(-half, half + 1):
                hy_off = np.clip(hy + dy, 0, he_h - 1)
                hx_off = np.clip(hx + dx, 0, he_w - 1)
                # Gaussian weight within the MALDI pixel footprint
                w = conf * np.exp(-0.5 * ((dx**2 + dy**2) / (self.psf_sigma_he**2 + 1e-6)))
                np.add.at(maldi_canvas,  (hy_off, hx_off), maldi_colors * w[:, None])
                np.add.at(weight_canvas, (hy_off, hx_off), w)

        # Normalise accumulated signal
        valid = weight_canvas > 0
        maldi_canvas[valid] /= weight_canvas[valid, None]

        # Inpaint any remaining gaps
        gap_mask   = (~valid).astype(np.uint8)
        maldi_uint8 = np.clip(maldi_canvas, 0, 255).astype(np.uint8)
        maldi_inpainted = cv2.inpaint(maldi_uint8, gap_mask, inpaintRadius=3,
                                    flags=cv2.INPAINT_NS)

        # Alpha-blend H&E + MALDI signal in H&E space
        he_float     = self.he_image.astype(np.float32)
        maldi_float  = maldi_inpainted.astype(np.float32)
        composite    = ((1 - alpha) * he_float + alpha * maldi_float)
        composite    = np.clip(composite, 0, 255).astype(np.uint8)

        cv2.imwrite(output_path, cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))
        print(f"Saved final overlay → '{output_path}'  ({he_w}×{he_h} px, H&E resolution)")

        fig, axes = plt.subplots(1, 3, figsize=(21, 7))
        axes[0].imshow(self.maldi_image[:, :, :3]); axes[0].set_title('MALDI (Taurine)'); axes[0].axis('off')
        axes[1].imshow(self.he_image);               axes[1].set_title('H&E Histology');  axes[1].axis('off')
        axes[2].imshow(composite);                   axes[2].set_title('Overlay (H&E space, MALDI signal)'); axes[2].axis('off')
        plt.tight_layout()
        plt.savefig(output_path.replace('.png', '_preview.png'), dpi=150, bbox_inches='tight')
        plt.show()
        return composite

    # ------------------------------------------------------------------
    def visualize_refinement(self, n_arrows=600, output_path='psf_refinement_diagnostics.png'):
        """
        Diagnostic figure: displacement vectors + confidence + magnitude histogram.
        """
        if self.refined_mapping is None:
            raise RuntimeError("Run compute_psf_weighted_centroids() first.")

        print(f"\nGenerating PSF refinement diagnostics...")
        df    = self.refined_mapping
        valid = df['confidence'] > 0.05

        fig, axes = plt.subplots(1, 3, figsize=(21, 7))

        # --- Panel 1: Displacement quiver on H&E ---
        if self.he_image is not None:
            axes[0].imshow(self.he_image, alpha=0.7)
        dfv = df[valid]
        if len(dfv) > n_arrows:
            dfv = dfv.sample(n_arrows, random_state=42)
        sc = axes[0].quiver(
            dfv['he_x_coarse'], dfv['he_y_coarse'],
            dfv['refinement_dx'], dfv['refinement_dy'],
            dfv['refinement_magnitude'],
            cmap='hot', scale=None, scale_units='xy',
            angles='xy', width=0.001, alpha=0.8
        )
        plt.colorbar(sc, ax=axes[0], label='Shift (H&E px)')
        axes[0].set_title('PSF Refinement Displacement\n(coarse → refined in H&E space)',
                          fontsize=11, fontweight='bold')
        axes[0].axis('off')

        # --- Panel 2: Confidence map (scatter in H&E space) ---
        if self.he_image is not None:
            axes[1].imshow(self.he_image, alpha=0.4)
        sc2 = axes[1].scatter(
            df['he_x_coarse'], df['he_y_coarse'],
            c=df['confidence'], cmap='viridis', s=1, alpha=0.6, vmin=0, vmax=1
        )
        plt.colorbar(sc2, ax=axes[1], label='Confidence')
        axes[1].set_title('PSF Confidence Map\n(0 = no nuclei found, 1 = full footprint coverage)',
                          fontsize=11, fontweight='bold')
        axes[1].axis('off')

        # --- Panel 3: Refinement magnitude histogram ---
        mag = df.loc[valid, 'refinement_magnitude']
        axes[2].hist(mag, bins=60, color='steelblue', edgecolor='white', linewidth=0.5)
        axes[2].axvline(mag.median(), color='tomato', linestyle='--', linewidth=2,
                        label=f"Median {mag.median():.1f} px")
        axes[2].axvline(mag.mean(),   color='gold',   linestyle='--', linewidth=2,
                        label=f"Mean   {mag.mean():.1f} px")
        axes[2].set_xlabel('Refinement Shift (H&E pixels)')
        axes[2].set_ylabel('MALDI Pixel Count')
        axes[2].set_title('Distribution of PSF Refinement Shifts', fontsize=11, fontweight='bold')
        axes[2].legend()

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved diagnostics → '{output_path}'")
        plt.show()


# ===========================================================================
#  COMBINED PIPELINE
# ===========================================================================

def run_registration_pipeline(he_path, maldi_path, n_landmarks=5,
                               save_coords=True, grid_spacing=1, tissue_only=True,
                               # --- Stage 2 parameters ---
                               nuclei_mask=None,
                               maldi_pixel_size_um=50.0,
                               he_pixel_size_um=0.5,
                               assignment_mode='soft',
                               search_radius_sigma=3.0):
    """
    Combined coarse + fine MALDI-to-H&E registration pipeline.

    Stage 1 (always runs): affine + TPS, saves coordinate_mapping.csv
    Stage 2 (runs if nuclei_mask provided): PSF refinement, saves
        refined_coordinate_mapping.csv and final_overlay.png

    Parameters
    ----------
    he_path, maldi_path : str
        Image file paths.
    n_landmarks : int
        Number of vessel landmark pairs for Stage 1.
    save_coords : bool
        Save Stage 1 coordinate_mapping.csv.
    grid_spacing : int
        Pixel subsampling for coordinate grid (1 = every pixel).
    tissue_only : bool
        Restrict coordinate mapping to tissue (MALDI foreground).
    nuclei_mask : ndarray or None
        H&E nuclei segmentation mask for Stage 2.
        If None, Stage 2 is skipped.
    maldi_pixel_size_um : float
        MALDI physical pixel size in µm (Stage 2).
    he_pixel_size_um : float
        H&E physical pixel size in µm (Stage 2).
    assignment_mode : {'soft', 'hard', 'mixed'}
        PSF centroid assignment mode (Stage 2).
    search_radius_sigma : float
        PSF footprint search radius in units of σ (Stage 2).

    Returns
    -------
    reg : MALDIRegistration
        Fully configured registration object.
        Access Stage 2 outputs via reg.psf_refiner (if run).
    """
    print(f"\n{'='*60}")
    print(f"MALDI-MSI TO H&E REGISTRATION PIPELINE")
    print(f"{'='*60}\n")

    # ---- Stage 1 --------------------------------------------------------
    print(f"Step 1/7: Loading and preprocessing images...")
    reg = MALDIRegistration(he_path, maldi_path)

    print(f"\nStep 2/7: Manual landmark selection...")
    reg.select_landmarks(n_points=n_landmarks)

    print(f"\nStep 3/7: Computing initial affine transformation...")
    reg.compute_affine_transform()

    print(f"\nStep 4/7: Refining affine transformation...")
    reg.refine_affine()

    print(f"\nStep 5/7: Applying non-rigid TPS deformation...")
    reg.apply_nonrigid_deformation()

    print(f"\nStep 6/7: Visualising coarse registration...")
    reg.visualize_results()
    reg.save_registered_image()

    if save_coords:
        reg.save_coordinate_mapping(grid_spacing=grid_spacing, tissue_only=tissue_only)
        reg.visualize_coordinate_mapping()

    # ---- Stage 2 (optional) ---------------------------------------------
    if nuclei_mask is not None:
        print(f"\nStep 7/7: PSF-aware nuclei centroid refinement (Stage 2)...")
        reg.run_psf_nuclei_refinement(
            nuclei_mask=nuclei_mask,
            maldi_pixel_size_um=maldi_pixel_size_um,
            he_pixel_size_um=he_pixel_size_um,
            assignment_mode=assignment_mode,
            search_radius_sigma=search_radius_sigma
        )
    else:
        print(f"\nStep 7/7: Skipped — provide nuclei_mask to run Stage 2.")
        print(f"  Hint: run reg.run_psf_nuclei_refinement(nuclei_mask=...) manually.")

    print(f"\n{'='*60}")
    print(f"REGISTRATION COMPLETE")
    print(f"{'='*60}")
    print(f"\nOutputs:")
    print(f"  registration_results.png       — coarse alignment panels")
    print(f"  registered_maldi.tif           — coarse-registered MALDI")
    print(f"  coordinate_mapping.csv         — coarse MALDI→H&E mapping")
    print(f"  coordinate_mapping_accuracy.html")
    if nuclei_mask is not None:
        print(f"  refined_coordinate_mapping.csv — PSF-refined MALDI→H&E mapping")
        print(f"  final_overlay.png              — H&E overlaid on MALDI space")
        print(f"  psf_refinement_diagnostics.png — shifts / confidence maps")
    print(f"{'='*60}\n")
    return reg


# ===========================================================================
#  STANDALONE STAGE 2 RUNNER
#  Use this if Stage 1 has already been run and you only want to refine.
# ===========================================================================

def run_psf_refinement_from_csv(coord_mapping_csv,
                                 he_image_path,
                                 maldi_image_path,
                                 nuclei_mask,
                                 maldi_pixel_size_um=50.0,
                                 he_pixel_size_um=0.5,
                                 assignment_mode='soft',
                                 search_radius_sigma=3.0,
                                 min_nuclei=1,
                                 output_path='refined_coordinate_mapping.csv'):
    """
    Run Stage 2 standalone from a saved coordinate_mapping.csv.

    Use this when you want to iterate on Stage 2 parameters without re-running
    the expensive Stage 1 registration.

    Parameters
    ----------
    coord_mapping_csv : str
        Path to coordinate_mapping.csv from Stage 1.
    he_image_path, maldi_image_path : str
        Image file paths.
    nuclei_mask : ndarray
        H&E nuclei segmentation output (binary or labelled).
    ...
    """
    refiner = PSFNucleiRefinement(
        coord_mapping_path=coord_mapping_csv,
        he_image_path=he_image_path,
        maldi_image_path=maldi_image_path,
        maldi_pixel_size_um=maldi_pixel_size_um,
        he_pixel_size_um=he_pixel_size_um
    )
    refiner.extract_nuclei_centroids(nuclei_mask)
    refined_df = refiner.compute_psf_weighted_centroids(
        search_radius_sigma=search_radius_sigma,
        assignment_mode=assignment_mode,
        min_nuclei=min_nuclei
    )
    refiner.save_refined_mapping(output_path)
    refiner.visualize_refinement()
    refiner.generate_final_overlay()
    return refiner, refined_df

def segmentation_pipeline(he_path: str, save_overlay=True, prob_thresh=0.3, 
                           nms_thresh=0.8, n_tiles=(4, 4, 1), show_tile_progress=True):

    print("Loading and normalizing H&E stained image...")
    he_image_bgr = cv2.imread(str(he_path))

    # FIX 1+2: convert BGR→RGB first, then normalize
    he_image_rgb  = cv2.cvtColor(he_image_bgr, cv2.COLOR_BGR2RGB)
    he_image_norm = normalize(he_image_rgb, 1, 99.8)   # csbdeep normalize expects RGB [0,1]

    print("Running StarDist for nuclei segmentation...")
    model = StarDist2D.from_pretrained('2D_versatile_he')

    # FIX 3: pass he_image_norm (not he_image) to predict_instances
    nuclei_masks, details = model.predict_instances(
        he_image_norm,                  # ← was he_image (raw uint8 BGR) — now normalized RGB
        axes='YXC',
        prob_thresh=prob_thresh,
        nms_thresh=nms_thresh,
        n_tiles=n_tiles,
        show_tile_progress=show_tile_progress
    )

    if save_overlay:
        print("Creating full-resolution overlay...")
        he_uint8 = (he_image_norm * 255).astype(np.uint8)  # already in [0,1] now, safe cast

        print("Finding boundaries via morphological gradient...")
        boundaries = ndi.morphological_gradient(nuclei_masks, size=3) > 0

        overlay = he_uint8.copy()
        overlay[boundaries] = [255, 0, 0]

        output_path = he_path.replace('.tif', '_nuclei_overlay.tif')
        tifffile.imwrite(output_path, overlay, compression='lzma', photometric='rgb')
        print(f"✓ Saved overlay: {output_path}")

        mask_path = he_path.replace('.tif', '_nuclei_masks.tif')
        tifffile.imwrite(mask_path, nuclei_masks.astype(np.uint16), compression='lzma')
        print(f"✓ Saved masks: {mask_path}")

    print("✓ Segmentation complete.")
    return nuclei_masks
# ===========================================================================
#  ENTRY POINT
# ===========================================================================

if __name__ == "__main__":

    HE_PATH    = "high_res_images/D2_10x_originalExport.tif"
    MALDI_PATH = "img_folder/Taurine_img_withoutborders.tif"

    # =========================================================
    # NUCLEI SEGMENTATION PLACEHOLDER
    # ---------------------------------------------------------
    # Paste your nuclei segmentation pipeline here.
    # Replace `nuclei_mask` with the actual output array.
    #
    # Expected:
    #   nuclei_mask : np.ndarray, shape (H_he, W_he), dtype int or bool
    #   Binary  — 1 where a nucleus pixel is present
    #   Labelled — each nucleus assigned a unique integer (e.g. StarDist output)
    #
    # =========================================================

    # run nuclei segmentation pipeline: 
    nuclei_mask = segmentation_pipeline(HE_PATH, n_tiles=(8, 8, 1))

    # ---- Full pipeline (Stage 1 + Stage 2 if mask provided) ----
    registration = run_registration_pipeline(
        he_path=HE_PATH,
        maldi_path=MALDI_PATH,
        n_landmarks=8,
        save_coords=True,
        grid_spacing=1,
        tissue_only=True,
        # Stage 2:
        nuclei_mask=nuclei_mask,        # set to None to skip Stage 2
        maldi_pixel_size_um=25.0,       # your MALDI raster step in µm
        he_pixel_size_um=4.0,           # your slide scanner pixel size in µm
        assignment_mode='soft',         # 'soft' | 'hard' | 'mixed'
        search_radius_sigma=5.0         # PSF footprint radius (multiples of σ)
    )
"""
    # ---- Alternatively: Stage 2 only (Stage 1 already done) ----
    refiner, refined_df = run_psf_refinement_from_csv(
        coord_mapping_csv='coordinate_mapping.csv',
        he_image_path=HE_PATH,
        maldi_image_path=MALDI_PATH,
        nuclei_mask=nuclei_mask,
        maldi_pixel_size_um=25.0,
        he_pixel_size_um=4.0,
        assignment_mode='soft',
        search_radius_sigma=5.0,   # wider than default — covers full MALDI footprint
        min_nuclei=1,
    )
"""
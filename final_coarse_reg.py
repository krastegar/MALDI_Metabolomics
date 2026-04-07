"""
MALDI-MSI to H&E Image Registration Pipeline with Coordinate Mapping
====================================================================
This script performs coarse-to-fine registration and provides coordinate
transformation functions to map MALDI coordinates to H&E space.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import matplotlib.cm as cm          
import cv2
from scipy import ndimage
from scipy.interpolate import RBFInterpolator
from scipy.optimize import minimize
from skimage import transform, filters
from pyimzml.ImzMLParser import ImzMLParser
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

class MALDIRegistration:
    """
    Main registration class for aligning MALDI-MSI data to H&E histology images.
    Includes coordinate transformation utilities.
    """
    
    def __init__(self, he_path, maldi_path, imzml_path="MSI_data_grant/Mass_Spec_data/20251012_old_liver_area.imzML"):
        """
        Initialize the registration object.
        
        Parameters:
        -----------
        he_path : str
            Path to the H&E stained image file
        maldi_path : str
            Path to the MALDI-MSI image file (RGBA format)
        """

        self.parser = ImzMLParser(imzml_path)
        self.maldi_df =pd.DataFrame(
                    (   # neat trick to unpack the mzs and intensities directly into the row
                        # * is a unpacking operator called splat and it unpacks the tuple of mzs and intensities
                        # from getspectrum into individual elements in the new tuple
                        (*self.parser.getspectrum(idx), coord) for idx, coord in enumerate(self.parser.coordinates)
                    ),
                        columns=["mzs", "intensities", "coordinates"]
            )
        # Load the H&E image (typically RGB, shape: height x width x 3)
        self.he_image = cv2.imread(he_path)
        self.he_image = cv2.cvtColor(self.he_image, cv2.COLOR_BGR2RGB)
        
        # Load the MALDI image (RGBA format, shape: height x width x 4)
        self.maldi_image = cv2.imread(maldi_path, cv2.IMREAD_UNCHANGED)
        if self.maldi_image.shape[2] == 4:
            self.maldi_image = cv2.cvtColor(self.maldi_image, cv2.COLOR_BGRA2RGBA)
        
        # Store original dimensions
        self.he_shape = self.he_image.shape[:2]
        self.maldi_shape = self.maldi_image.shape[:2]
        
        # Convert to grayscale
        maldi_rgb = self.maldi_image[:, :, :3]
        self.maldi_gray = (0.299 * maldi_rgb[:, :, 0] + 
                          0.587 * maldi_rgb[:, :, 1] + 
                          0.114 * maldi_rgb[:, :, 2])
        
        self.he_gray = (0.299 * self.he_image[:, :, 0] + 
                       0.587 * self.he_image[:, :, 1] + 
                       0.114 * self.he_image[:, :, 2])
        
        # Normalize
        self.maldi_gray = self.maldi_gray / 255.0
        self.he_gray = self.he_gray / 255.0
        
        # Initialize storage
        self.he_landmarks = []
        self.maldi_landmarks = []
        self.affine_matrix = None
        self.refined_affine = None
        self.registered_affine = None
        self.registered_nonrigid = None
        self.maldi_grid = None
        # NEW: Storage for coordinate transformation
        self.displacement_field_x = None
        self.displacement_field_y = None
        self.rbf_x = None
        self.rbf_y = None
        
        print(f"Loaded H&E image: {self.he_shape}")
        print(f"Loaded MALDI image: {self.maldi_shape}")
        print(f"Images preprocessed and ready for landmark selection")
    
    def select_landmarks(self, n_points=5):
        """Interactive landmark selection."""
        print(f"\n{'='*60}")
        print(f"LANDMARK SELECTION MODE")
        print(f"{'='*60}")
        print(f"Instructions:")
        print(f"1. Click {n_points} corresponding points on the H&E image (LEFT)")
        print(f"2. Then click {n_points} corresponding points on MALDI image (RIGHT)")
        print(f"3. Choose distinctive features: blood vessels, tissue boundaries, etc.")
        print(f"4. Distribute points across the tissue for better alignment")
        print(f"5. Close the window when done")
        print(f"{'='*60}\n")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        ax1.imshow(self.he_image)
        ax1.set_title('H&E Image - Click HERE FIRST', fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        ax2.imshow(self.maldi_image)
        ax2.set_title('MALDI Image - Click HERE SECOND', fontsize=14, fontweight='bold')
        ax2.axis('off')
        
        he_points_plot = []
        maldi_points_plot = []
        current_image = 'he'
        he_count = 0
        maldi_count = 0
        
        def onclick(event):
            nonlocal current_image, he_count, maldi_count
            
            if event.inaxes is None:
                return
            
            x, y = event.xdata, event.ydata
            
            if event.inaxes == ax1 and current_image == 'he' and he_count < n_points:
                self.he_landmarks.append([x, y])
                point, = ax1.plot(x, y, 'ro', markersize=10, markeredgecolor='white', markeredgewidth=2)
                ax1.text(x, y, str(he_count + 1), color='yellow', fontsize=12, 
                        fontweight='bold', ha='center', va='center')
                he_points_plot.append(point)
                he_count += 1
                print(f"H&E landmark {he_count}/{n_points}: ({x:.1f}, {y:.1f})")
                
                if he_count == n_points:
                    current_image = 'maldi'
                    ax2.set_title('MALDI Image - CLICK NOW', fontsize=14, 
                                fontweight='bold', color='red')
                    print(f"\n>>> Now click {n_points} CORRESPONDING points on MALDI image <<<\n")
            
            elif event.inaxes == ax2 and current_image == 'maldi' and maldi_count < n_points:
                self.maldi_landmarks.append([x, y])
                point, = ax2.plot(x, y, 'ro', markersize=10, markeredgecolor='white', markeredgewidth=2)
                ax2.text(x, y, str(maldi_count + 1), color='yellow', fontsize=12, 
                        fontweight='bold', ha='center', va='center')
                maldi_points_plot.append(point)
                maldi_count += 1
                print(f"MALDI landmark {maldi_count}/{n_points}: ({x:.1f}, {y:.1f})")
                
                if maldi_count == n_points:
                    ax2.set_title('MALDI Image - COMPLETE! Close window.', 
                                fontsize=14, fontweight='bold', color='green')
                    print(f"\n{'='*60}")
                    print(f"Landmark selection complete! Close the window to continue.")
                    print(f"{'='*60}\n")
            
            fig.canvas.draw()
        
        fig.canvas.mpl_connect('button_press_event', onclick)
        plt.tight_layout()
        plt.show()
        
        self.he_landmarks = np.array(self.he_landmarks)
        self.maldi_landmarks = np.array(self.maldi_landmarks)
        
        print(f"Collected {len(self.he_landmarks)} landmark pairs")
        
    def compute_affine_transform(self):
        """Compute initial affine transformation."""
        print(f"\nComputing affine transformation from landmarks...")
        
        if len(self.he_landmarks) < 3:
            raise ValueError(f"Need at least 3 landmark pairs, got {len(self.he_landmarks)}")
        
        tform = transform.SimilarityTransform()
        success = tform.estimate(self.maldi_landmarks, self.he_landmarks)
        
        if not success:
            raise RuntimeError("Failed to estimate affine transformation")
        
        self.affine_matrix = tform.params
        
        print(f"Affine matrix computed:")
        print(self.affine_matrix)
        
        self.registered_affine = cv2.warpAffine(
            self.maldi_gray,
            self.affine_matrix[:2, :],
            (self.he_shape[1], self.he_shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        print(f"Initial affine transformation applied")
        
    def extract_tissue_mask(self, image, threshold=0.1):
        """Extract binary mask of tissue region."""
        mask = image > threshold
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel_close)
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
        return mask.astype(bool)
    
    def refine_affine(self):
        """Refine affine transformation using boundary optimization."""
        print(f"\nRefining affine transformation...")
        
        he_mask = self.extract_tissue_mask(self.he_gray)
        maldi_mask = self.extract_tissue_mask(self.maldi_gray)
        he_edges = filters.sobel(self.he_gray * he_mask)
        
        initial_params = np.array([
            self.affine_matrix[0, 0],
            self.affine_matrix[0, 1],
            self.affine_matrix[1, 0],
            self.affine_matrix[1, 1],
            self.affine_matrix[0, 2],
            self.affine_matrix[1, 2]
        ])
        
        def cost_function(params):
            affine = np.array([
                [params[0], params[1], params[4]],
                [params[2], params[3], params[5]],
            ])
            
            transformed_mask = cv2.warpAffine(
                maldi_mask.astype(float),
                affine,
                (self.he_shape[1], self.he_shape[0]),
                flags=cv2.INTER_LINEAR
            )
            
            transformed_edges = filters.sobel(transformed_mask)
            overlap = np.corrcoef(he_edges.flatten(), transformed_edges.flatten())[0, 1]
            return -overlap
        
        print(f"Optimizing affine parameters...")
        result = minimize(
            cost_function,
            initial_params,
            method='Nelder-Mead',
            options={'maxiter': 100, 'disp': False}
        )
        
        refined_params = result.x
        self.refined_affine = np.array([
            [refined_params[0], refined_params[1], refined_params[4]],
            [refined_params[2], refined_params[3], refined_params[5]],
            [0, 0, 1]
        ])
        
        self.registered_affine = cv2.warpAffine(
            self.maldi_gray,
            self.refined_affine[:2, :],
            (self.he_shape[1], self.he_shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        
        print(f"Affine refinement complete")
        print(f"Optimized correlation: {-result.fun:.4f}")
        
    def apply_nonrigid_deformation(self):
        """Apply non-rigid deformation and store transformation fields."""
        print(f"\nApplying non-rigid deformation...")
        
        # Transform MALDI landmarks using refined affine
        maldi_landmarks_transformed = cv2.transform(
            self.maldi_landmarks.reshape(-1, 1, 2),
            self.refined_affine[:2, :]
        ).reshape(-1, 2)
        
        # Calculate displacement vectors
        displacements = self.he_landmarks - maldi_landmarks_transformed
        
        # Create RBF interpolators and STORE them for later coordinate transformation
        self.rbf_x = RBFInterpolator(
            maldi_landmarks_transformed,
            displacements[:, 0],
            kernel='thin_plate_spline',
            smoothing=0.0
        )
        
        self.rbf_y = RBFInterpolator(
            maldi_landmarks_transformed,
            displacements[:, 1],
            kernel='thin_plate_spline',
            smoothing=0.0
        )
        
        # Create dense grid
        y_coords, x_coords = np.mgrid[0:self.he_shape[0], 0:self.he_shape[1]]
        points = np.column_stack([x_coords.ravel(), y_coords.ravel()])
        
        print(f"Computing displacement field ({len(points)} points)...")
        dx = self.rbf_x(points).reshape(self.he_shape)
        dy = self.rbf_y(points).reshape(self.he_shape)
        
        # STORE displacement fields for coordinate transformation
        self.displacement_field_x = dx
        self.displacement_field_y = dy
        
        # Create mapping grid
        map_x = (x_coords - dx).astype(np.float32)
        map_y = (y_coords - dy).astype(np.float32)
        
        # Apply deformation
        self.registered_nonrigid = cv2.remap(
            self.registered_affine,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        
        print(f"Non-rigid deformation complete")
    
    # ========== NEW COORDINATE TRANSFORMATION METHODS ==========
    
    def transform_maldi_to_he_coordinates(self, maldi_coords):
        """
        Transform MALDI spot coordinates to H&E coordinate space.
        
        This applies the SAME transformation used to register the MALDI image to H&E.
        MALDI spots will map to their corresponding locations in the H&E image.
        
        Parameters:
        -----------
        maldi_coords : array-like, shape (N, 2) or (2,)
            MALDI spot coordinates as [[x1, y1], [x2, y2], ...] or [x, y]
            These are pixel coordinates in the original MALDI image
            
        Returns:
        --------
        he_coords : ndarray, shape (N, 2)
            Corresponding H&E coordinates where each MALDI spot maps to
        """
        if self.refined_affine is None or self.rbf_x is None:
            raise RuntimeError("Must complete registration pipeline before transforming coordinates")
        
        # Convert to numpy array and ensure 2D shape
        maldi_coords = np.atleast_2d(maldi_coords)
        print(f"maldi coords shape: {maldi_coords.shape}, \nmaldi coords: \n{maldi_coords}, \ntype: {type(maldi_coords)} ")


        # Ensure coordinates are within image bounds
        maldi_coords[:, 0] = np.clip(maldi_coords[:, 0], 0, self.maldi_shape[1] - 1)
        maldi_coords[:, 1] = np.clip(maldi_coords[:, 1], 0, self.maldi_shape[0] - 1)
        # Step 1: Apply affine transformation
        # This is the same affine that was applied to the MALDI image
        maldi_coords_homogeneous = np.column_stack([maldi_coords, np.ones(len(maldi_coords))])
        affine_transformed = (self.refined_affine @ maldi_coords_homogeneous.T).T
        affine_coords = affine_transformed[:, :2]
        
        # Step 2: Apply non-rigid displacement
        # This is the same non-rigid deformation applied to the MALDI image
        dx = self.rbf_x(affine_coords)
        dy = self.rbf_y(affine_coords)
        
        # Add displacement to get final H&E coordinates
        # This gives us where each MALDI spot appears in the H&E image
        he_coords = affine_coords + np.column_stack([dx, dy])
        
        return he_coords
    
    def transform_he_to_maldi_coordinates(self, he_coords, max_iterations=50, tolerance=0.5):
        """
        Transform H&E coordinates back to MALDI space (inverse transformation).
        Uses iterative optimization since the transformation is non-linear.
        
        Parameters:
        -----------
        he_coords : array-like, shape (N, 2) or (2,)
            H&E coordinates as [[x1, y1], [x2, y2], ...] or [x, y]
        max_iterations : int
            Maximum iterations for inverse optimization
        tolerance : float
            Convergence tolerance in pixels
            
        Returns:
        --------
        maldi_coords : ndarray, shape (N, 2)
            Corresponding MALDI coordinates
        """
        if self.refined_affine is None or self.rbf_x is None:
            raise RuntimeError("Must complete registration pipeline before transforming coordinates")
        
        he_coords = np.atleast_2d(he_coords)
        maldi_coords = np.zeros_like(he_coords)
        
        # Compute inverse affine for initial guess
        affine_inv = np.linalg.inv(self.refined_affine)
        
        for i, target_he in enumerate(he_coords):
            # Initial guess: inverse affine only
            guess_homogeneous = np.append(target_he, 1)
            guess = (affine_inv @ guess_homogeneous)[:2]
            
            # Iterative refinement
            for _ in range(max_iterations):
                # Forward transform current guess
                predicted_he = self.transform_maldi_to_he_coordinates(guess.reshape(1, -1))[0]
                
                # Check convergence
                error = np.linalg.norm(predicted_he - target_he)
                if error < tolerance:
                    break
                
                # Update guess (simple gradient descent)
                guess -= 0.5 * (predicted_he - target_he)
            
            maldi_coords[i] = guess
        
        return maldi_coords
    
    def create_coordinate_mapping_grid(self, grid_spacing=1, tissue_only=True, intensity_threshold=0.1):
        """
        Create a regular grid mapping between MALDI and H&E coordinates.
        
        Parameters:
        -----------
        grid_spacing : int
            Spacing between grid points in pixels (default=1 for every pixel)
            Use grid_spacing=1 to map EVERY MALDI tissue pixel
            Use grid_spacing>1 for a sparser grid (faster, less memory)
        tissue_only : bool
            If True, only include coordinates where there's actual tissue (default=True)
            Uses extract_tissue_mask() method for detection
            If False, include all coordinates including background
        intensity_threshold : float
            Threshold for tissue detection (0-1 range, default=0.1)
            Passed to extract_tissue_mask()
            
        Returns:
        --------
        mapping_df : pandas.DataFrame
            DataFrame with columns: maldi_x, maldi_y, he_x, he_y
            
        Note:
        -----
        For a MALDI image of size (H, W):
        - grid_spacing=1, tissue_only=False: H*W rows (every pixel)
        - grid_spacing=1, tissue_only=True: only tissue pixels (typically 20-50% of image)
        - grid_spacing=10: (H/10)*(W/10) rows (10% of pixels)
        """
        print(f"Creating coordinate mapping with grid_spacing={grid_spacing}...")
        
        if tissue_only:
            print(f"  Extracting tissue mask (threshold={intensity_threshold})...")
            
            # Use the existing extract_tissue_mask method
            tissue_coords = np.asarray([coords[:2] for coords in self.maldi_df['coordinates']])
            maldi_grid = tissue_coords # original behavior

            # Apply grid spacing if requested
            if grid_spacing > 1:
                # Sample every grid_spacing-th point
                maldi_grid = tissue_coords[::grid_spacing]
            
            print(f"  MALDI image shape: {self.maldi_shape}")
            print(f"  Total MALDI pixels: {self.maldi_shape[0] * self.maldi_shape[1]}")
            print(f"  Mapping grid points (with spacing={grid_spacing}): {len(maldi_grid)}")
            
        else:
            # Original behavior - regular grid over entire image
            y_maldi = np.arange(0, self.maldi_shape[0], grid_spacing)
            x_maldi = np.arange(0, self.maldi_shape[1], grid_spacing)
            xv, yv = np.meshgrid(x_maldi, y_maldi)
            
            maldi_grid = np.column_stack([xv.ravel(), yv.ravel()])
            
            print(f"  MALDI image shape: {self.maldi_shape}")
            print(f"  Total MALDI pixels: {self.maldi_shape[0] * self.maldi_shape[1]}")
            print(f"  Mapping grid points: {len(maldi_grid)}")
        
        # Transform to H&E space
        print(f"  Transforming {len(maldi_grid)} coordinates...")
        he_grid = self.transform_maldi_to_he_coordinates(maldi_grid)
        print(f'\nH&E transformed coords shape: {he_grid.shape}, \nH&E coords: \n{he_grid[:5]}, \ntype: {type(he_grid)} ')
        self.maldi_grid = maldi_grid
        # Create DataFrame
        mapping_df = pd.DataFrame({
            'maldi_x': maldi_grid[:, 0],
            'maldi_y': maldi_grid[:, 1],
            'he_x': he_grid[:, 0],
            'he_y': he_grid[:, 1]
        })
        
        print(f"  Complete! Generated {len(mapping_df)} coordinate mappings")
        
        return mapping_df
    
    def save_coordinate_mapping(self, output_path='coordinate_mapping.csv', grid_spacing=1, 
                               tissue_only=True, intensity_threshold=0.1):
        """
        Save coordinate mapping to CSV file.
        
        Parameters:
        -----------
        output_path : str
            Path to save CSV file
        grid_spacing : int
            Grid spacing for mapping (default=1 for every pixel)
            Use grid_spacing=1 to map ALL tissue pixels (recommended)
            Use grid_spacing>1 for faster/smaller file (sparse sampling)
        tissue_only : bool
            If True, only map tissue regions using extract_tissue_mask() (default=True)
            If False, map entire image including background
        intensity_threshold : float
            Threshold for tissue detection (0-1 range, default=0.1)
            Passed to extract_tissue_mask()
            Lower values include more of the image
            
        Warning:
        --------
        With grid_spacing=1 and large MALDI images, this may take time
        and produce large CSV files. For a 500x500 MALDI image:
        - grid_spacing=1, tissue_only=True: ~50,000-150,000 rows (~3-8 MB)
        - grid_spacing=1, tissue_only=False: 250,000 rows (~15 MB CSV)
        - grid_spacing=5: 10,000 rows (~0.6 MB CSV)
        - grid_spacing=10: 2,500 rows (~0.15 MB CSV)
        """
        print(f"\nGenerating coordinate mapping grid (spacing={grid_spacing} pixels)...")
        mapping_df = self.create_coordinate_mapping_grid(grid_spacing, tissue_only, intensity_threshold)
        
        print(f"Saving to '{output_path}'...")
        mapping_df.to_csv(output_path, index=False)
        
        print(f"Saved coordinate mapping to '{output_path}'")
        print(f"  Rows: {len(mapping_df)}")
        
        return mapping_df
    
    def visualize_coordinate_mapping(self, pixel_size_um=2.0):
        """
        Visualize the coordinate transformation as a vector field and grid deformation.
        Interactive Plotly version. Axes are displayed in µm.

        Parameters
        ----------
        pixel_size_um : float
            Physical size of one pixel in micrometres (default=2.0 µm).
            Used to convert pixel coordinates to µm on all axes.
        """
        print(f"\nGenerating coordinate mapping visualization...")
        tissue_coords = np.asarray([coords[:2] for coords in self.maldi_df['coordinates']])
        he_grid = self.transform_maldi_to_he_coordinates(tissue_coords)

        # Affine-only positions (non-rigid displacement starts here)
        maldi_grid_homogeneous = np.column_stack([tissue_coords, np.ones(len(tissue_coords))])
        affine_only = (self.refined_affine @ maldi_grid_homogeneous.T).T[:, :2]

        # Non-rigid displacement in µm
        displacement = he_grid - affine_only
        displacement_mag = np.sqrt(displacement[:, 0]**2 + displacement[:, 1]**2) * pixel_size_um
        max_displacement = np.max(displacement_mag)

        # ------------------------------------------------------------------ #
        # Convert everything to µm before passing to Plotly.                  #
        # px.imshow with explicit x/y ranges renders the image in µm space,   #
        # so all scatter coordinates must also be in µm to overlay correctly.  #
        # ------------------------------------------------------------------ #
        he_h_px, he_w_px = self.he_shape
        x_range_um = [0, he_w_px * pixel_size_um]
        y_range_um = [0, he_h_px * pixel_size_um]

        # All scatter data converted to µm
        he_grid_um     = he_grid     * pixel_size_um   # shape (N, 2)
        affine_only_um = affine_only * pixel_size_um   # shape (N, 2)
        landmarks_um   = self.he_landmarks * pixel_size_um

        # Summary displacement stats (mean & median of non-rigid shift in µm)
        mean_displacement   = float(np.mean(displacement_mag))
        median_displacement = float(np.median(displacement_mag))

        # Subsample scatter points for performance
        subsample_indices = np.random.choice(len(he_grid_um), size=len(he_grid_um)//2, replace=False)

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=(
                'MALDI Grid Deformed to H&E Space<br>(Shows where MALDI spots map to)',
                f'Non-Rigid Displacement Vectors<br>'
                f'Mean: {mean_displacement:.1f} µm  |  Median: {median_displacement:.1f} µm'
            ),
            horizontal_spacing=0.1
        )

        # ---- Background H&E image rendered in µm space ---- #
        # go.Image supports x0/y0/dx/dy which place it in physical coordinates
        # while preserving the correct top-left origin and RGB colour.
        for col in [1, 2]:
            fig.add_trace(
                go.Image(
                    z=self.he_image,
                    x0=0, y0=0,
                    dx=pixel_size_um, dy=pixel_size_um,
                    hovertemplate='X: %{x:.1f} µm<br>Y: %{y:.1f} µm<extra>H&E</extra>'
                ),
                row=1, col=col
            )

        # ---- Left plot: deformed MALDI grid overlaid on H&E ---- #
        fig.add_trace(
            go.Scatter(
                x=he_grid_um[subsample_indices, 0],
                y=he_grid_um[subsample_indices, 1],
                mode='markers',
                marker=dict(color='blue', size=3, opacity=0.15),
                name='MALDI points',
                hovertemplate='MALDI point<br>X: %{x:.1f} µm<br>Y: %{y:.1f} µm<extra></extra>',
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=landmarks_um[:, 0],
                y=landmarks_um[:, 1],
                mode='markers',
                marker=dict(color='lime', size=10, line=dict(color='white', width=2)),
                name='H&E landmarks',
                hovertemplate='Landmark<br>X: %{x:.1f} µm<br>Y: %{y:.1f} µm<extra></extra>',
            ),
            row=1, col=1
        )

        # ---- Right plot: non-rigid displacement vectors in µm ---- #
        mask = displacement_mag > 2.0  # threshold: > 1 pixel

        if np.any(mask):
            vector_subsample = min(500, np.sum(mask))
            if np.sum(mask) > vector_subsample:
                mask_indices = np.where(mask)[0]
                selected_indices = np.random.choice(mask_indices, size=vector_subsample, replace=False)
                vector_mask = np.zeros(len(mask), dtype=bool)
                vector_mask[selected_indices] = True
            else:
                vector_mask = mask

            norm_disp = (displacement_mag[vector_mask] - displacement_mag[vector_mask].min()) / \
                        (displacement_mag[vector_mask].max() - displacement_mag[vector_mask].min() + 1e-10)

            colormap = cm.get_cmap('Spectral')
            colors = [f'rgb({int(r*255)},{int(g*255)},{int(b*255)})'
                      for r, g, b, _ in colormap(norm_disp)]

            for idx, i in enumerate(np.where(vector_mask)[0]):
                x_start, y_start = affine_only_um[i]
                x_end,   y_end   = he_grid_um[i]
                fig.add_trace(
                    go.Scatter(
                        x=[x_start, x_end],
                        y=[y_start, y_end],
                        mode='lines',
                        line=dict(color=colors[idx], width=2),
                        showlegend=False,
                        hovertemplate=(
                            f'Start: ({x_start:.1f}, {y_start:.1f}) µm<br>'
                            f'End:   ({x_end:.1f}, {y_end:.1f}) µm<br>'
                            f'Displacement: {displacement_mag[i]:.2f} µm<extra></extra>'
                        ),
                    ),
                    row=1, col=2
                )

            fig.add_trace(
                go.Scatter(
                    x=he_grid_um[vector_mask, 0],
                    y=he_grid_um[vector_mask, 1],
                    mode='markers',
                    marker=dict(
                        color=displacement_mag[vector_mask],
                        colorscale='Spectral',
                        size=6,
                        symbol='arrow',
                        colorbar=dict(title='Displacement (µm)', x=1.15, len=0.5, y=0.5),
                        showscale=True
                    ),
                    name='Displacement',
                    hovertemplate='X: %{x:.1f} µm<br>Y: %{y:.1f} µm<br>Displacement: %{marker.color:.2f} µm<extra></extra>'
                ),
                row=1, col=2
            )
        else:
            fig.add_annotation(
                text='No significant non-rigid displacement<br>(all < 2.0 µm)',
                xref='x4', yref='y4',
                x=np.mean(x_range_um), y=np.mean(y_range_um),
                xanchor='center', yanchor='middle',
                showarrow=False,
                font=dict(size=14),
                bgcolor='wheat',
                opacity=0.8,
                row=1, col=2
            )

        # ---- Axis formatting — native µm, y-axis reversed to match image convention ---- #
        for col in [1, 2]:
            fig.update_xaxes(title_text='H&E X (µm)', range=x_range_um, row=1, col=col)
            fig.update_yaxes(title_text='H&E Y (µm)', range=y_range_um,
                             autorange='reversed',
                             scaleanchor=f'x{col if col > 1 else ""}',
                             scaleratio=1, row=1, col=col)
        
        fig.update_layout(
            height=700,
            width=1600,
            title_text='Coordinate Mapping Visualization',
            showlegend=True,
            hovermode='closest'
        )
        
        # Save as HTML
        fig.write_html('coordinate_mapping_accuracy.html')
        print(f"Saved interactive coordinate mapping visualization to 'coordinate_mapping_accuracy.html'")
        print(f"\nRegistration Accuracy:")
        print(f"  Mean non-rigid displacement  : {mean_displacement:.1f} µm")
        print(f"  Median non-rigid displacement: {median_displacement:.1f} µm")
        print(f"  Showing {len(subsample_indices)} of {len(he_grid_um)} total MALDI points")
        if np.any(mask):
            print(f"  Showing {np.sum(vector_mask)} displacement vectors")
        
        fig.show()

    def visualize_mi_heatmap(self, patch_size=64, n_bins=32,
                             pixel_size_um=2.0, output_path='mi_heatmap.html'):
        """
        Compute and visualise a spatial map of local NMI between the registered
        MALDI image and the H&E image, sampled at actual tissue spot locations.

        For each MALDI spot (transformed to H&E space), a patch of size
        `patch_size` × `patch_size` pixels is extracted from both images and
        NMI is computed. Only spots where a full patch fits within the image
        bounds are scored. The result is overlaid on the H&E grayscale in the
        same orientation as visualize_results().

        Parameters
        ----------
        patch_size : int
            Side length of the patch in pixels (default=64).
            Smaller → finer per-spot detail but noisier estimates.
        n_bins : int
            Number of intensity bins for the joint histogram (default=32).
        pixel_size_um : float
            Physical pixel size in µm (default=2.0).
        output_path : str
            File path for the saved interactive HTML figure.
        """
        if self.registered_nonrigid is None:
            raise RuntimeError("Run the full pipeline before calling visualize_mi_heatmap().")

        print("\nComputing local NMI heatmap...")

        he    = self.he_gray
        maldi = self.registered_nonrigid
        eps   = 1e-10

        def _patch_nmi(patch_a, patch_b, n_bins):
            """NMI = 2*MI / (H(A)+H(B)), strictly in [0,1]."""
            hist2d, _, _ = np.histogram2d(
                patch_a.ravel(), patch_b.ravel(),
                bins=n_bins, range=[[0, 1], [0, 1]]
            )
            hist2d = hist2d / (hist2d.sum() + eps)  # proper joint probability
            p_a  = hist2d.sum(axis=1)
            p_b  = hist2d.sum(axis=0)
            h_a  = -np.sum(p_a    * np.log(p_a    + eps))
            h_b  = -np.sum(p_b    * np.log(p_b    + eps))
            h_ab = -np.sum(hist2d * np.log(hist2d + eps))
            mi   = h_a + h_b - h_ab
            return 2.0 * mi / (h_a + h_b + eps)

        # ------------------------------------------------------------------ #
        # Use tissue-only MALDI spot coordinates as patch centres.            #
        # These are the actual measured spots — no grid, no background.       #
        # ------------------------------------------------------------------ #
        tissue_coords = np.asarray([coords[:2] for coords in self.maldi_df['coordinates']])
        # Transform to H&E pixel space so patches align with the registered image
        he_coords = self.transform_maldi_to_he_coordinates(tissue_coords)

        he_h_px, he_w_px = self.he_shape
        half = patch_size // 2

        centres_x, centres_y, nmi_scores = [], [], []

        for (cx_f, cy_f) in he_coords:
            cx, cy = int(round(cx_f)), int(round(cy_f))
            # Skip spots too close to the image border for a full patch
            if cx < half or cy < half or cx + half > he_w_px or cy + half > he_h_px:
                continue
            pa = he   [cy-half:cy+half, cx-half:cx+half]
            pb = maldi[cy-half:cy+half, cx-half:cx+half]
            nmi = _patch_nmi(pa, pb, n_bins)
            centres_x.append(cx)
            centres_y.append(cy)
            nmi_scores.append(nmi)

        centres_x  = np.array(centres_x,  dtype=float)
        centres_y  = np.array(centres_y,  dtype=float)
        nmi_scores = np.array(nmi_scores)

        print(f"  Scored {len(nmi_scores)} tissue spots  "
              f"(patch={patch_size}px = {patch_size*pixel_size_um:.0f} µm)")
        print(f"  NMI range: {nmi_scores.min():.4f} – {nmi_scores.max():.4f}")

        # Convert to µm
        centres_x_um = centres_x * pixel_size_um
        centres_y_um = centres_y * pixel_size_um
        x_range_um   = [0, he_w_px * pixel_size_um]
        y_range_um   = [0, he_h_px * pixel_size_um]

        # ------------------------------------------------------------------ #
        # Background: grayscale H&E via go.Heatmap with y descending so that  #
        # row 0 is at the top — matching matplotlib imshow convention used in  #
        # visualize_results().                                                 #
        # ------------------------------------------------------------------ #
        x_coords_um = np.linspace(0, he_w_px * pixel_size_um, he_w_px)
        # Descending y so row 0 maps to the top of the plot
        y_coords_um = np.linspace(0, he_h_px * pixel_size_um, he_h_px)

        fig = go.Figure()

        fig.add_trace(
            go.Heatmap(
                z=he,
                x=x_coords_um,
                y=y_coords_um,
                colorscale='gray',
                showscale=False,
                hovertemplate='X: %{x:.1f} µm<br>Y: %{y:.1f} µm<extra>H&E</extra>',
                name='H&E'
            )
        )

        # NMI scores overlaid on tissue spots
        marker_size = max(4, int(patch_size * pixel_size_um /
                                 (he_w_px * pixel_size_um / 800)))

        fig.add_trace(
            go.Scatter(
                x=centres_x_um,
                y=centres_y_um,
                mode='markers',
                marker=dict(
                    color=nmi_scores,
                    colorscale='Spectral',
                    cmin=0.0,
                    cmax=1.0,
                    size=marker_size,
                    opacity=0.8,
                    symbol='square',
                    colorbar=dict(
                        title='NMI (0–1)',
                        thickness=18,
                        len=0.75,
                        tickformat='.2f',
                    ),
                    showscale=True,
                    line=dict(width=0),
                ),
                name='Local NMI',
                hovertemplate=(
                    'X: %{x:.1f} µm<br>'
                    'Y: %{y:.1f} µm<br>'
                    'NMI: %{marker.color:.4f}<extra></extra>'
                ),
            )
        )

        # y-axis reversed so row 0 is at top, matching imshow orientation
        fig.update_xaxes(title_text='H&E X (µm)', range=x_range_um)
        fig.update_yaxes(title_text='H&E Y (µm)',
                         range=y_range_um,
                         autorange='reversed',
                         scaleanchor='x', scaleratio=1)

        fig.update_layout(
            title=dict(
                text=(f'Local NMI — MALDI vs H&E (tissue spots only)<br>'
                      f'<sup>patch={patch_size}px ({patch_size*pixel_size_um:.0f} µm)</sup>'),
                x=0.5
            ),
            height=750,
            width=900,
            hovermode='closest',
        )

        fig.write_html(output_path)
        print(f"  Saved NMI heatmap to '{output_path}'")
        fig.show()

    def compute_registration_metrics(self, pixel_size_um=2.0, verbose=True):
        """
        Compute quantitative metrics to evaluate registration quality.

        Metrics are computed at three stages (where available):
          - Pre-registration (raw overlap)
          - Post-affine
          - Post-non-rigid

        Parameters
        ----------
        pixel_size_um : float
            Physical size of one pixel in micrometres (default=2.0 µm).
            Used to convert displacement magnitudes to µm.
        verbose : bool
            If True, print a formatted summary table to stdout.

        Returns
        -------
        metrics : dict
            Nested dict with keys 'pre', 'affine', 'nonrigid'.
            Each stage contains:

            nmi : float
                Normalised Mutual Information between H&E and registered MALDI
                (computed only over the tissue overlap mask).
                Higher is better; 1.0 = perfect statistical dependency.

            edge_correlation : float
                Pearson correlation between Sobel edge maps of both images.
                Modality-agnostic structural similarity.
                Higher is better; 1.0 = perfect edge alignment.

            ssim : float
                Structural Similarity Index on the edge maps.
                Higher is better; 1.0 = identical structure.

            displacement_stats : dict  (nonrigid stage only)
                mean_um, std_um, max_um, pct_folding:
                  - mean/std/max: displacement field magnitude statistics in µm.
                  - pct_folding: % of pixels where Jacobian det < 0 (impossible
                    warps); should be 0.0 for a valid deformation field.
        """
        from skimage.metrics import structural_similarity as ssim_fn

        # ------------------------------------------------------------------ #
        # Helper: Normalised Mutual Information                                #
        # ------------------------------------------------------------------ #
        def _nmi(img_a, img_b, mask=None, n_bins=64):
            """NMI = 2 * MI(A,B) / (H(A) + H(B)), computed over masked region.
            Values are strictly in [0, 1]: 0 = independent, 1 = perfectly dependent.
            """
            a = img_a[mask] if mask is not None else img_a.ravel()
            b = img_b[mask] if mask is not None else img_b.ravel()
            eps = 1e-10
            # Joint histogram — density=False then normalise so values sum to 1
            hist_2d, _, _ = np.histogram2d(a, b, bins=n_bins)
            hist_2d = hist_2d / (hist_2d.sum() + eps)   # proper joint probability
            # Marginals
            p_a = hist_2d.sum(axis=1)
            p_b = hist_2d.sum(axis=0)
            # Entropies
            h_a  = -np.sum(p_a     * np.log(p_a     + eps))
            h_b  = -np.sum(p_b     * np.log(p_b     + eps))
            h_ab = -np.sum(hist_2d * np.log(hist_2d + eps))
            mi = h_a + h_b - h_ab
            return 2.0 * mi / (h_a + h_b + eps)

        # ------------------------------------------------------------------ #
        # Helper: Edge-map correlation + SSIM                                  #
        # ------------------------------------------------------------------ #
        def _edge_metrics(img_a, img_b, mask=None):
            edges_a = filters.sobel(img_a)
            edges_b = filters.sobel(img_b)
            if mask is not None:
                ea = edges_a[mask]
                eb = edges_b[mask]
            else:
                ea, eb = edges_a.ravel(), edges_b.ravel()
            corr = float(np.corrcoef(ea, eb)[0, 1])
            # SSIM on full edge maps (requires same shape)
            data_range = max(edges_a.max(), edges_b.max()) - min(edges_a.min(), edges_b.min())
            sim = ssim_fn(edges_a, edges_b, data_range=data_range)
            return corr, float(sim)

        # ------------------------------------------------------------------ #
        # Helper: Displacement field statistics                                #
        # ------------------------------------------------------------------ #
        def _displacement_stats(dx, dy, pixel_size_um):
            mag_px = np.sqrt(dx**2 + dy**2)
            mag_um = mag_px * pixel_size_um

            # Jacobian determinant  det = (1 + dDx/dx)(1 + dDy/dy) - (dDx/dy)(dDy/dx)
            # Use finite differences
            ddx_dx = np.gradient(dx, axis=1)
            ddx_dy = np.gradient(dx, axis=0)
            ddy_dx = np.gradient(dy, axis=1)
            ddy_dy = np.gradient(dy, axis=0)
            jac_det = (1 + ddx_dx) * (1 + ddy_dy) - ddx_dy * ddy_dx
            pct_fold = float(np.mean(jac_det < 0) * 100)

            return {
                "mean_um":    float(np.mean(mag_um)),
                "std_um":     float(np.std(mag_um)),
                "max_um":     float(np.max(mag_um)),
                "pct_folding": pct_fold,
            }

        # ------------------------------------------------------------------ #
        # Compute overlap mask (tissue present in both images)                 #
        # ------------------------------------------------------------------ #
        he_mask = self.extract_tissue_mask(self.he_gray)

        metrics = {}

        # ---- PRE-REGISTRATION -------------------------------------------- #
        maldi_resized = cv2.resize(
            self.maldi_gray,
            (self.he_shape[1], self.he_shape[0])
        )
        pre_mask = he_mask & self.extract_tissue_mask(maldi_resized)
        pre_nmi = _nmi(self.he_gray, maldi_resized, mask=pre_mask)
        pre_ec, pre_ssim = _edge_metrics(self.he_gray, maldi_resized, mask=pre_mask)
        metrics["pre"] = {"nmi": pre_nmi, "edge_correlation": pre_ec, "ssim": pre_ssim}

        # ---- POST-AFFINE -------------------------------------------------- #
        if self.registered_affine is not None:
            aff_mask = he_mask & self.extract_tissue_mask(self.registered_affine)
            aff_nmi = _nmi(self.he_gray, self.registered_affine, mask=aff_mask)
            aff_ec, aff_ssim = _edge_metrics(self.he_gray, self.registered_affine, mask=aff_mask)
            metrics["affine"] = {"nmi": aff_nmi, "edge_correlation": aff_ec, "ssim": aff_ssim}

        # ---- POST-NON-RIGID ----------------------------------------------- #
        if self.registered_nonrigid is not None:
            nr_mask = he_mask & self.extract_tissue_mask(self.registered_nonrigid)
            nr_nmi = _nmi(self.he_gray, self.registered_nonrigid, mask=nr_mask)
            nr_ec, nr_ssim = _edge_metrics(self.he_gray, self.registered_nonrigid, mask=nr_mask)
            disp_stats = _displacement_stats(
                self.displacement_field_x,
                self.displacement_field_y,
                pixel_size_um
            )
            metrics["nonrigid"] = {
                "nmi": nr_nmi,
                "edge_correlation": nr_ec,
                "ssim": nr_ssim,
                "displacement_stats": disp_stats,
            }

        # ------------------------------------------------------------------ #
        # Store and optionally print                                           #
        # ------------------------------------------------------------------ #
        self.registration_metrics = metrics

        if verbose:
            print(f"\n{'='*62}")
            print(f"  REGISTRATION QUALITY METRICS")
            print(f"{'='*62}")
            header = f"  {'Metric':<28} {'Pre':>8} {'Affine':>8} {'Non-rigid':>10}"
            print(header)
            print(f"  {'-'*58}")

            def _v(stage, key):
                return f"{metrics[stage][key]:.4f}" if stage in metrics else "  N/A  "

            print(f"  {'NMI':<28} {_v('pre','nmi'):>8} {_v('affine','nmi'):>8} {_v('nonrigid','nmi'):>10}")
            print(f"  {'Edge Correlation':<28} {_v('pre','edge_correlation'):>8} {_v('affine','edge_correlation'):>8} {_v('nonrigid','edge_correlation'):>10}")
            print(f"  {'Edge SSIM':<28} {_v('pre','ssim'):>8} {_v('affine','ssim'):>8} {_v('nonrigid','ssim'):>10}")

            if "nonrigid" in metrics and "displacement_stats" in metrics["nonrigid"]:
                ds = metrics["nonrigid"]["displacement_stats"]
                print(f"\n  Non-rigid displacement field ({pixel_size_um} µm/pixel):")
                print(f"    Mean displacement : {ds['mean_um']:.2f} µm")
                print(f"    Std  displacement : {ds['std_um']:.2f} µm")
                print(f"    Max  displacement : {ds['max_um']:.2f} µm")
                print(f"    Folding (det<0)   : {ds['pct_folding']:.3f} %  "
                      f"{'✓ OK' if ds['pct_folding'] < 0.1 else '⚠ CHECK'}")

            print(f"{'='*62}\n")

        return metrics

    def visualize_results(self):
        """Visualize registration results."""
        print(f"\nGenerating visualization...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 16))
        
        # 1. Original
        maldi_resized = cv2.resize(self.maldi_gray, 
                                   (self.he_shape[1], self.he_shape[0]))
        overlay_original = self.create_overlay(self.he_gray, maldi_resized)
        axes[0, 0].imshow(overlay_original, cmap='gray')
        axes[0, 0].set_title('Original (No Registration)', fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')
        
        # 2. Landmarks
        axes[0, 1].imshow(self.he_image)
        if self.refined_affine is not None:
            maldi_transformed = cv2.transform(
                self.maldi_landmarks.reshape(-1, 1, 2),
                self.refined_affine[:2, :]
            ).reshape(-1, 2)
            for i in range(len(self.he_landmarks)):
                axes[0, 1].plot([self.he_landmarks[i, 0], maldi_transformed[i, 0]],
                               [self.he_landmarks[i, 1], maldi_transformed[i, 1]],
                               'y-', linewidth=2, alpha=0.6)
                axes[0, 1].plot(self.he_landmarks[i, 0], self.he_landmarks[i, 1],
                               'ro', markersize=10, markeredgecolor='white', markeredgewidth=2)
                axes[0, 1].plot(maldi_transformed[i, 0], maldi_transformed[i, 1],
                               'bo', markersize=10, markeredgecolor='white', markeredgewidth=2)
        axes[0, 1].set_title('Landmark Correspondence', fontsize=14, fontweight='bold')
        axes[0, 1].axis('off')
        
        # 3. Affine
        if self.registered_affine is not None:
            overlay_affine = self.create_overlay(self.he_gray, self.registered_affine)
            axes[1, 0].imshow(overlay_affine, cmap='gray')
            axes[1, 0].set_title('After Affine Registration', fontsize=14, fontweight='bold')
            axes[1, 0].axis('off')
        
        # 4. Non-rigid
        if self.registered_nonrigid is not None:
            overlay_nonrigid = self.create_overlay(self.he_gray, self.registered_nonrigid)
            axes[1, 1].imshow(overlay_nonrigid, cmap='gray')
            axes[1, 1].set_title('After Non-Rigid Deformation', fontsize=14, fontweight='bold')
            axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig('registration_results.png', dpi=150, bbox_inches='tight')
        print(f"Saved visualization to 'registration_results.png'")
        plt.show()
        
    def create_overlay(self, img1, img2, alpha=0.5):
        """Create blended overlay."""
        img1_norm = (img1 - img1.min()) / (img1.max() - img1.min() + 1e-8)
        img2_norm = (img2 - img2.min()) / (img2.max() - img2.min() + 1e-8)
        overlay = alpha * img1_norm + (1 - alpha) * img2_norm
        return overlay
    
    def save_registered_image(self, output_path='registered_maldi.tif'):
        """Save final registered image."""
        if self.registered_nonrigid is not None:
            output = (self.registered_nonrigid * 255).astype(np.uint8)
            cv2.imwrite(output_path, output)
            print(f"\nSaved registered image to '{output_path}'")
        else:
            print(f"No registered image available yet. Run the full pipeline first.")


def run_registration_pipeline(he_path, maldi_path, n_landmarks=5,
                              save_coords=True, grid_spacing=1, tissue_only=True,
                              pixel_size_um=2.0):
    """
    Complete registration pipeline with coordinate mapping output.
    
    Parameters:
    -----------
    he_path : str
        Path to H&E image
    maldi_path : str
        Path to MALDI image
    n_landmarks : int
        Number of landmark pairs
    save_coords : bool
        Whether to save coordinate mapping
    grid_spacing : int
        Grid spacing for coordinate mapping (default=1 for every pixel)
        - grid_spacing=1: Maps EVERY MALDI pixel (recommended for full data)
        - grid_spacing=5-10: Faster, smaller file, use for preview/testing
    tissue_only : bool
        If True, only map tissue regions (default=True, recommended)
        If False, map entire image including background
    
    Returns:
    --------
    registration : MALDIRegistration
        Registration object with coordinate transformation methods
    """
    print(f"\n{'='*60}")
    print(f"MALDI-MSI TO H&E REGISTRATION PIPELINE")
    print(f"{'='*60}\n")
    
    # Initialize
    print(f"Step 1/7: Loading and preprocessing images...")
    reg = MALDIRegistration(he_path, maldi_path)
    
    # Landmark selection
    print(f"\nStep 2/7: Manual landmark selection...")
    reg.select_landmarks(n_points=n_landmarks)
    
    # Affine
    print(f"\nStep 3/7: Computing affine transformation...")
    reg.compute_affine_transform()
    
    # Refine
    print(f"\nStep 4/7: Refining affine transformation...")
    reg.refine_affine()
    
    # Non-rigid
    print(f"\nStep 5/7: Applying non-rigid deformation...")
    reg.apply_nonrigid_deformation()
    
    # Metrics
    print(f"\nStep 6/7: Computing registration quality metrics...")
    reg.compute_registration_metrics(pixel_size_um=pixel_size_um)

    # Visualize
    print(f"\nStep 7/7: Generating visualizations...")
    reg.visualize_results()
    
    # Save results
    reg.save_registered_image()
    
    # NEW: Save coordinate mapping
    if save_coords:
        reg.save_coordinate_mapping(grid_spacing=grid_spacing, tissue_only=tissue_only)
        reg.visualize_coordinate_mapping(pixel_size_um=pixel_size_um)
        reg.visualize_mi_heatmap(patch_size=64, n_bins=32,
                             pixel_size_um=2.0, output_path='mi_heatmap.html')
    
    print(f"\n{'='*60}")
    print(f"REGISTRATION COMPLETE!")
    print(f"{'='*60}")
    print(f"\nAvailable methods:")
    print(f"  - reg.compute_registration_metrics(pixel_size_um)")
    print(f"  - reg.transform_maldi_to_he_coordinates(maldi_coords)")
    print(f"  - reg.transform_he_to_maldi_coordinates(he_coords)")
    print(f"  - reg.create_coordinate_mapping_grid(grid_spacing, tissue_only)")
    print(f"{'='*60}\n")
    
    return reg


# Example usage:
if __name__ == "__main__":
    HE_PATH = "/home/krastegar0/MALDI_Metabolomics/img_folder/old_liver_10x.tiff"
    MALDI_PATH = "/home/krastegar0/MALDI_Metabolomics/img_folder/Taurine_img_withoutborders.tif"
    
    # Run pipeline with TISSUE-ONLY coordinate mapping
    registration = run_registration_pipeline(
        he_path=HE_PATH,
        maldi_path=MALDI_PATH,
        n_landmarks=8,
        save_coords=True,
        grid_spacing=1,      # Map every pixel
        tissue_only=True     # Only map tissue regions, not background!
    )

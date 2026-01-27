"""
MALDI-MSI to H&E Image Registration Pipeline with Coordinate Mapping
====================================================================
This script performs coarse-to-fine registration and provides coordinate
transformation functions to map MALDI coordinates to H&E space.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import cv2
from scipy import ndimage
from scipy.interpolate import RBFInterpolator
from scipy.optimize import minimize
from skimage import transform, filters
from pyimzml.ImzMLParser import ImzMLParser
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
    
    def visualize_coordinate_mapping(self):
        """
        Visualize the coordinate transformation as a vector field and grid deformation.
        
        Parameters:
        -----------
        n_arrows : int
            Number of arrows to display in each dimension
        """
        print(f"\nGenerating coordinate mapping visualization...")
        tissue_coords = np.asarray([coords[:2] for coords in self.maldi_df['coordinates']])
        he_grid = self.transform_maldi_to_he_coordinates(tissue_coords)
        # Reshape for grid visualization
        he_grid_x = he_grid[:, 0]
        he_grid_y = he_grid[:, 1]
        
        # Calculate displacement vectors (from affine-transformed position to final position)
        # First get affine-only transformation
        maldi_grid_homogeneous = np.column_stack([self.maldi_grid, np.ones(len(self.maldi_grid))])
        #print(f'\nmaldi grid shape: {self.maldi_grid.shape}, \nrefined_afine_shape: {self.refined_affine.shape}')
        affine_only = (self.refined_affine @ maldi_grid_homogeneous.T).T[:, :2]
        
        # Displacement is the NON-RIGID component only
        displacement = he_grid - affine_only
        
        fig = plt.figure(figsize=(25, 15))
        gs = fig.add_gridspec(1, 2, width_ratios=[1, 1])
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        
        # Left: Deformed grid overlay on H&E
        ax1.imshow(self.he_image, alpha=0.8)

        # Plot grid points showing how regular MALDI grid deforms to H&E space
        ax1.plot(he_grid_x, he_grid_y, 'bo', alpha=0.1, markersize=1, label='MALDI_points')

        # Mark landmarks
        ax1.plot(self.he_landmarks[:, 0], self.he_landmarks[:, 1], 
                'go', markersize=8, markeredgecolor='white', markeredgewidth=2, 
                label='H&E landmarks')
        ax1.set_title('MALDI Grid Deformed to H&E Space\n(Shows where MALDI spots map to)', 
                     fontsize=12, fontweight='bold')
        ax1.set_xlabel('H&E X (pixels)')
        ax1.set_ylabel('H&E Y (pixels)')
        ax1.legend()
        ax1.axis('equal')
        
        # Middle: Non-rigid displacement vectors only
        displacement_mag = np.sqrt(displacement[:, 0]**2 + displacement[:, 1]**2)
        max_displacement = np.max(displacement_mag)
        
        ax2.imshow(self.he_image, alpha=0.7)
        # Only show vectors where there's significant non-rigid deformation
        # ...existing code...
        mask = displacement_mag > 1.0  # Only show displacements > 1 pixel
        quiv = None
        if np.any(mask):
            quiv = ax2.quiver(
                affine_only[mask, 0], affine_only[mask, 1], 
                displacement[mask, 0], displacement[mask, 1],
                displacement_mag[mask], cmap='Spectral', alpha=0.8, 
                scale=1, scale_units='xy', angles='xy', width=0.003
            )
            plt.colorbar(quiv, label="Displacement Magnitude (pixels)", ax=ax2)  # ✓ This is correct
        ax2.set_title(f'Non-Rigid Displacement Vectors\n(Max: {max_displacement:.1f} pixels)', 
                    fontsize=12, fontweight='bold')
        ax2.set_xlabel('X (pixels)')
        ax2.set_ylabel('Y (pixels)')

# ...existing code...
        plt.tight_layout()
        plt.savefig('coordinate_mapping_accuracy.png', dpi=150, bbox_inches='tight')
        print(f"Saved coordinate mapping visualization to 'coordinate_mapping_accuracy.png'")
        print(f"\nRegistration Accuracy:")
        print(f"  Max non-rigid displacement: {max_displacement:.1f} pixels")
        plt.show()
    
    # ========== END NEW METHODS ==========
    
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
                              save_coords=True, grid_spacing=1, tissue_only=True):
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
    print(f"Step 1/6: Loading and preprocessing images...")
    reg = MALDIRegistration(he_path, maldi_path)
    
    # Landmark selection
    print(f"\nStep 2/6: Manual landmark selection...")
    reg.select_landmarks(n_points=n_landmarks)
    
    # Affine
    print(f"\nStep 3/6: Computing affine transformation...")
    reg.compute_affine_transform()
    
    # Refine
    print(f"\nStep 4/6: Refining affine transformation...")
    reg.refine_affine()
    
    # Non-rigid
    print(f"\nStep 5/6: Applying non-rigid deformation...")
    reg.apply_nonrigid_deformation()
    
    # Visualize
    print(f"\nStep 6/6: Generating visualizations...")
    reg.visualize_results()
    
    # Save results
    reg.save_registered_image()
    
    # NEW: Save coordinate mapping
    if save_coords:
        reg.save_coordinate_mapping(grid_spacing=grid_spacing, tissue_only=tissue_only)
        reg.visualize_coordinate_mapping()
    
    print(f"\n{'='*60}")
    print(f"REGISTRATION COMPLETE!")
    print(f"{'='*60}")
    print(f"\nAvailable coordinate transformation methods:")
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
        n_landmarks=5,
        save_coords=True,
        grid_spacing=1,      # Map every pixel
        tissue_only=True     # Only map tissue regions, not background!
    )
    

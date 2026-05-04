"""

from skimage import io, color
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
#from __future__ import print_function

import histomicstk as htk

import numpy as np

import skimage.io
import skimage.measure
import skimage.color

import matplotlib.pyplot as plt

# Load H&E image
img = io.imread("/home/krastegar0/Lipidomics/img_folder/old_liver_10x.tiff")

# plot functions without sub plots 
plt.figure(figsize=(12,10))
plt.imshow(img)
plt.axis('off')
plt.show()
"""

"""
MALDI-MSI to H&E Image Registration Pipeline
============================================
This script performs coarse-to-fine registration of a MALDI ion map to an H&E stained image.

Pipeline Steps:
1. Load and preprocess images
2. Manual landmark selection (interactive)
3. Compute initial affine transformation
4. Refine with optimization
5. Apply non-rigid deformation
6. Visualize results
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import cv2
from scipy import ndimage
from scipy.interpolate import RBFInterpolator
from scipy.optimize import minimize
from skimage import transform, filters
import warnings
warnings.filterwarnings('ignore')

class MALDIRegistration:
    """
    Main registration class for aligning MALDI-MSI data to H&E histology images.
    """
    
    def __init__(self, he_path, maldi_path):
        """
        Initialize the registration object.
        
        Parameters:
        -----------
        he_path : str
            Path to the H&E stained image file
        maldi_path : str
            Path to the MALDI-MSI image file (RGBA format)
        """
        # Load the H&E image (typically RGB, shape: height x width x 3)
        self.he_image = cv2.imread(he_path)
        # OpenCV loads as BGR, convert to RGB for proper display in matplotlib
        self.he_image = cv2.cvtColor(self.he_image, cv2.COLOR_BGR2RGB)
        
        # Load the MALDI image (RGBA format, shape: height x width x 4)
        self.maldi_image = cv2.imread(maldi_path, cv2.IMREAD_UNCHANGED)
        # Convert from BGR(A) to RGB(A) if needed
        if self.maldi_image.shape[2] == 4:  # Has alpha channel
            self.maldi_image = cv2.cvtColor(self.maldi_image, cv2.COLOR_BGRA2RGBA)
        
        # Store original dimensions for later reference
        self.he_shape = self.he_image.shape[:2]  # (height, width)
        self.maldi_shape = self.maldi_image.shape[:2]  # (height, width)
        
        # Convert MALDI RGBA to grayscale (this will be our "TIC" for registration)
        # Drop the alpha channel first
        maldi_rgb = self.maldi_image[:, :, :3]  # Take only RGB channels
        
        # Convert to grayscale using luminance formula (standard RGB to grayscale conversion)
        # Weights are based on human perception: green is perceived brighter than red, red brighter than blue
        self.maldi_gray = (0.299 * maldi_rgb[:, :, 0] +   # Red channel contribution
                          0.587 * maldi_rgb[:, :, 1] +   # Green channel contribution (highest weight)
                          0.114 * maldi_rgb[:, :, 2])    # Blue channel contribution
        
        # Convert H&E to grayscale for registration purposes
        # We use the same luminance formula for consistency
        self.he_gray = (0.299 * self.he_image[:, :, 0] + 
                       0.587 * self.he_image[:, :, 1] + 
                       0.114 * self.he_image[:, :, 2])
        
        # Normalize both grayscale images to [0, 1] range for consistent processing
        # This prevents intensity scale differences from affecting registration
        self.maldi_gray = self.maldi_gray / 255.0
        self.he_gray = self.he_gray / 255.0
        
        # Initialize storage for landmarks (corresponding points selected by user)
        # Each will be a list of (x, y) coordinates
        self.he_landmarks = []      # Points clicked on H&E image
        self.maldi_landmarks = []   # Corresponding points clicked on MALDI image
        
        # Storage for transformation matrices
        self.affine_matrix = None       # Initial affine transformation (6 parameters)
        self.refined_affine = None      # Optimized affine transformation
        
        # Storage for registered images at different stages
        self.registered_affine = None      # MALDI after affine transformation
        self.registered_nonrigid = None    # MALDI after non-rigid deformation
        
        print(f"Loaded H&E image: {self.he_shape}")
        print(f"Loaded MALDI image: {self.maldi_shape}")
        print(f"Images preprocessed and ready for landmark selection")
    
    def select_landmarks(self, n_points=5):
        """
        Interactive tool for manual landmark selection.
        User clicks corresponding points on both images to establish initial correspondence.
        
        Parameters:
        -----------
        n_points : int
            Number of landmark pairs to collect (default: 5)
            Minimum 3 required for affine, 4-6 recommended for robustness
        """
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
        
        # Create a figure with two subplots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Display H&E image on the left
        ax1.imshow(self.he_image)  # Show in color for easier landmark identification
        ax1.set_title('H&E Image - Click HERE FIRST', fontsize=14, fontweight='bold')
        ax1.axis('off')  # Hide axis ticks for cleaner display
        
        # Display MALDI image on the right
        ax2.imshow(self.maldi_image)  # Show in color
        ax2.set_title('MALDI Image - Click HERE SECOND', fontsize=14, fontweight='bold')
        ax2.axis('off')
        
        # Lists to store clicked points for visualization
        he_points_plot = []    # Will store matplotlib point objects for H&E
        maldi_points_plot = []  # Will store matplotlib point objects for MALDI
        
        # Counter to track which image we're currently clicking on
        current_image = 'he'  # Start with H&E
        he_count = 0          # Number of points clicked on H&E
        maldi_count = 0       # Number of points clicked on MALDI
        
        def onclick(event):
            """
            Callback function triggered when user clicks on either image.
            Records the clicked coordinates and displays them visually.
            """
            nonlocal current_image, he_count, maldi_count
            
            # Check if click was inside an axis (not on border or outside)
            if event.inaxes is None:
                return
            
            # Get the x, y coordinates of the click
            x, y = event.xdata, event.ydata
            
            # Determine which image was clicked based on which axis
            if event.inaxes == ax1 and current_image == 'he' and he_count < n_points:
                # Click on H&E image
                self.he_landmarks.append([x, y])  # Store coordinates
                # Plot a red circle with white border at clicked location
                point, = ax1.plot(x, y, 'ro', markersize= 2, markeredgecolor='white', markeredgewidth=2)
                # Add a number label next to the point
                ax1.text(x, y, str(he_count + 1), color='yellow', fontsize=12, 
                        fontweight='bold', ha='center', va='center')
                he_points_plot.append(point)
                he_count += 1
                print(f"H&E landmark {he_count}/{n_points}: ({x:.1f}, {y:.1f})")
                
                # Switch to MALDI mode after collecting all H&E points
                if he_count == n_points:
                    current_image = 'maldi'
                    ax2.set_title('MALDI Image - CLICK NOW', fontsize=14, 
                                fontweight='bold', color='red')
                    print(f"\n>>> Now click {n_points} CORRESPONDING points on MALDI image <<<\n")
            
            elif event.inaxes == ax2 and current_image == 'maldi' and maldi_count < n_points:
                # Click on MALDI image
                self.maldi_landmarks.append([x, y])  # Store coordinates
                # Plot a red circle with white border at clicked location
                point, = ax2.plot(x, y, 'ro', markersize=10, markeredgecolor='white', markeredgewidth=2)
                # Add a number label next to the point
                ax2.text(x, y, str(maldi_count + 1), color='yellow', fontsize=12, 
                        fontweight='bold', ha='center', va='center')
                maldi_points_plot.append(point)
                maldi_count += 1
                print(f"MALDI landmark {maldi_count}/{n_points}: ({x:.1f}, {y:.1f})")
                
                # When all points collected, inform user and change title
                if maldi_count == n_points:
                    ax2.set_title('MALDI Image - COMPLETE! Close window.', 
                                fontsize=14, fontweight='bold', color='green')
                    print(f"\n{'='*60}")
                    print(f"Landmark selection complete! Close the window to continue.")
                    print(f"{'='*60}\n")
            
            # Redraw the figure to show the new point
            fig.canvas.draw()
        
        # Connect the click event to our callback function
        fig.canvas.mpl_connect('button_press_event', onclick)
        
        # Adjust spacing between subplots
        plt.tight_layout()
        # Display the figure (blocks until user closes window)
        plt.show()
        
        # Convert landmark lists to numpy arrays for easier computation
        self.he_landmarks = np.array(self.he_landmarks)
        self.maldi_landmarks = np.array(self.maldi_landmarks)
        
        print(f"Collected {len(self.he_landmarks)} landmark pairs")
        
    def compute_affine_transform(self):
        """
        Compute the initial affine transformation matrix from landmarks.
        
        An affine transformation preserves:
        - Parallel lines remain parallel
        - Ratios of distances along lines
        
        It includes: translation, rotation, scaling, and shearing (6 parameters total)
        This is sufficient for initial alignment before non-rigid deformation.
        """
        print(f"\nComputing affine transformation from landmarks...")
        
        # Check if we have enough landmarks (minimum 3 needed for affine)
        if len(self.he_landmarks) < 3:
            raise ValueError(f"Need at least 3 landmark pairs, got {len(self.he_landmarks)}")
        
        # Use skimage's estimate_transform to compute affine matrix
        # This uses least-squares to find the best affine transformation
        # that maps maldi_landmarks -> he_landmarks
        tform = transform.SimilarityTransform()
        
        # The function estimates parameters that map source (MALDI) to destination (H&E)
        # Returns True if successful
        success = tform.estimate(self.maldi_landmarks, self.he_landmarks)
        
        if not success:
            raise RuntimeError("Failed to estimate affine transformation")
        
        # Store the 3x3 affine transformation matrix
        # Format: [[a, b, tx],
        #          [c, d, ty],
        #          [0, 0, 1 ]]
        # where (a,b,c,d) handle rotation/scale/shear and (tx,ty) handle translation
        self.affine_matrix = tform.params
        
        print(f"Affine matrix computed:")
        print(self.affine_matrix)
        
        # Apply the transformation to the MALDI grayscale image
        # This warps the MALDI image to align with H&E
        self.registered_affine = cv2.warpAffine(
            self.maldi_gray,  # Source image (what we're transforming)
            self.affine_matrix[:2, :],  # Transformation matrix (2x3 format for OpenCV)
            (self.he_shape[1], self.he_shape[0]),  # Output size (width, height)
            flags=cv2.INTER_LINEAR,  # Use bilinear interpolation for smooth result
            borderMode=cv2.BORDER_CONSTANT,  # Fill outside regions with zeros
            borderValue=0  # Black background
        )
        print(f"Initial affine transformation applied")
        
    def extract_tissue_mask(self, image, threshold=0.1):
        """
        Extract binary mask of tissue region from grayscale image.
        This helps focus registration on actual tissue rather than background.
        
        Parameters:
        -----------
        image : ndarray
            Grayscale image (values in [0, 1])
        threshold : float
            Intensity threshold to separate tissue from background
            
        Returns:
        --------
        mask : ndarray (boolean)
            Binary mask where True = tissue, False = background
        """
        # Simple thresholding: anything above threshold is considered tissue
        mask = image > threshold
        
        # Apply morphological operations to clean up the mask
        # 1. Closing: fills small holes in tissue
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel_close)
        
        # 2. Opening: removes small noise outside tissue
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
        
        return mask.astype(bool)
    
    def refine_affine(self):
        """
        Refine the initial affine transformation using boundary-based optimization.
        
        Strategy:
        - Extract tissue boundaries from both images
        - Optimize affine parameters to maximize boundary overlap
        - Uses the initial landmark-based affine as starting point
        """
        print(f"\nRefining affine transformation...")
        
        # Extract tissue masks from both images
        he_mask = self.extract_tissue_mask(self.he_gray)
        maldi_mask = self.extract_tissue_mask(self.maldi_gray)
        
        # Extract boundaries (edges) using Sobel filter
        # Boundaries are most informative for alignment
        he_edges = filters.sobel(self.he_gray * he_mask)
        
        # Initial affine parameters from the landmark-based transformation
        # Convert 3x3 matrix to parameter vector [a, b, c, d, tx, ty]
        initial_params = np.array([
            self.affine_matrix[0, 0],  # a: x-scaling and rotation
            self.affine_matrix[0, 1],  # b: x-shearing and rotation
            self.affine_matrix[1, 0],  # c: y-shearing and rotation
            self.affine_matrix[1, 1],  # d: y-scaling and rotation
            self.affine_matrix[0, 2],  # tx: x-translation
            self.affine_matrix[1, 2]   # ty: y-translation
        ])
        
        def cost_function(params):
            """
            Cost function to minimize: negative overlap between boundaries.
            Lower cost = better alignment.
            
            Parameters:
            -----------
            params : array [a, b, c, d, tx, ty]
                Affine transformation parameters
                
            Returns:
            --------
            cost : float
                Negative correlation between aligned boundaries (lower is better)
            """
            # Reconstruct affine matrix from parameters
            affine = np.array([
                [params[0], params[1], params[4]],  # [a, b, tx]
                [params[2], params[3], params[5]],  # [c, d, ty]
            ])
            
            # Apply transformation to MALDI mask
            transformed_mask = cv2.warpAffine(
                maldi_mask.astype(float),
                affine,
                (self.he_shape[1], self.he_shape[0]),
                flags=cv2.INTER_LINEAR
            )
            
            # Extract edges from transformed MALDI
            transformed_edges = filters.sobel(transformed_mask)
            
            # Compute normalized cross-correlation between edge maps
            # Higher correlation = better alignment
            # We return negative because optimizer minimizes
            overlap = np.corrcoef(he_edges.flatten(), transformed_edges.flatten())[0, 1]
            
            # Return negative correlation (we want to maximize, optimizer minimizes)
            return -overlap
        
        # Optimize using Nelder-Mead (derivative-free method, robust to noise)
        print(f"Optimizing affine parameters...")
        result = minimize(
            cost_function,
            initial_params,
            method='Nelder-Mead',
            options={'maxiter': 100, 'disp': False}
        )
        
        # Reconstruct refined affine matrix from optimized parameters
        refined_params = result.x
        self.refined_affine = np.array([
            [refined_params[0], refined_params[1], refined_params[4]],
            [refined_params[2], refined_params[3], refined_params[5]],
            [0, 0, 1]
        ])
        
        # Apply refined transformation
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
        """
        Apply non-rigid (elastic) deformation to handle local tissue distortions.
        
        Uses Thin Plate Spline (TPS) interpolation based on landmarks.
        This allows local warping while keeping the overall structure intact.
        """
        print(f"\nApplying non-rigid deformation...")
        
        # For non-rigid, we need the transformed landmark positions
        # Transform MALDI landmarks using the refined affine
        maldi_landmarks_transformed = cv2.transform(
            self.maldi_landmarks.reshape(-1, 1, 2),
            self.refined_affine[:2, :]
        ).reshape(-1, 2)
        
        # Create displacement field using Radial Basis Function interpolation
        # This creates smooth deformations based on how landmarks should move
        
        # Calculate displacement vectors (where each point should move to)
        displacements = self.he_landmarks - maldi_landmarks_transformed
        
        # Create RBF interpolators for x and y displacements
        # These will predict displacement at any point based on nearby landmarks
        rbf_x = RBFInterpolator(
            maldi_landmarks_transformed,  # Source positions
            displacements[:, 0],  # Target x displacements
            kernel='thin_plate_spline',  # TPS kernel (smooth, no parameters)
            smoothing=0.0  # No additional smoothing
        )
        
        rbf_y = RBFInterpolator(
            maldi_landmarks_transformed,
            displacements[:, 1],  # Target y displacements
            kernel='thin_plate_spline',
            smoothing=0.0
        )
        
        # Create a dense grid of points covering the entire H&E image
        # We'll compute displacement at each point
        y_coords, x_coords = np.mgrid[0:self.he_shape[0], 0:self.he_shape[1]]
        
        # Flatten coordinates for vectorized computation
        points = np.column_stack([x_coords.ravel(), y_coords.ravel()])
        
        # Predict displacement at each pixel
        print(f"Computing displacement field ({len(points)} points)...")
        dx = rbf_x(points).reshape(self.he_shape)
        dy = rbf_y(points).reshape(self.he_shape)
        
        # Create mapping grid: where to sample from in the affine-registered image
        # map_x[i,j] tells us which x-coordinate to sample from source
        # map_y[i,j] tells us which y-coordinate to sample from source
        map_x = (x_coords - dx).astype(np.float32)
        map_y = (y_coords - dy).astype(np.float32)
        
        # Apply the deformation field using OpenCV's remap
        # This warps the affine-registered MALDI to match H&E locally
        self.registered_nonrigid = cv2.remap(
            self.registered_affine,  # Source image (affine-registered MALDI)
            map_x,  # X coordinates to sample from
            map_y,  # Y coordinates to sample from
            interpolation=cv2.INTER_LINEAR,  # Bilinear interpolation
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        
        print(f"Non-rigid deformation complete")
        
    def visualize_results(self):
        """
        Visualize registration results at different stages:
        1. Original images
        2. After affine transformation
        3. After non-rigid deformation
        
        Uses checkerboard pattern to highlight alignment quality.
        """
        print(f"\nGenerating visualization...")
        
        # Create a figure with 4 subplots showing progression
        fig, axes = plt.subplots(2, 2, figsize=(16, 16))
        
        # 1. Original MALDI overlaid on H&E (no registration)
        # Resize MALDI to roughly match H&E for visualization
        maldi_resized = cv2.resize(self.maldi_gray, 
                                   (self.he_shape[1], self.he_shape[0]))
        overlay_original = self.create_overlay(self.he_gray, maldi_resized)
        axes[0, 0].imshow(overlay_original, cmap='gray')
        axes[0, 0].set_title('Original (No Registration)', fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')
        
        # 2. Landmarks visualization
        axes[0, 1].imshow(self.he_image)
        # Transform MALDI landmarks to H&E space for visualization
        if self.refined_affine is not None:
            maldi_transformed = cv2.transform(
                self.maldi_landmarks.reshape(-1, 1, 2),
                self.refined_affine[:2, :]
            ).reshape(-1, 2)
            # Draw lines connecting corresponding landmarks
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
        
        # 3. After affine transformation
        if self.registered_affine is not None:
            overlay_affine = self.create_overlay(self.he_gray, self.registered_affine)
            axes[1, 0].imshow(overlay_affine, cmap='gray')
            axes[1, 0].set_title('After Affine Registration', fontsize=14, fontweight='bold')
            axes[1, 0].axis('off')
        
        # 4. After non-rigid deformation
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
        """
        Create a blended overlay of two images for visualization.
        
        Parameters:
        -----------
        img1, img2 : ndarray
            Grayscale images to overlay
        alpha : float
            Blending factor (0.5 = equal mix)
            
        Returns:
        --------
        overlay : ndarray
            Blended image
        """
        # Normalize both images to [0, 1]
        img1_norm = (img1 - img1.min()) / (img1.max() - img1.min() + 1e-8)
        img2_norm = (img2 - img2.min()) / (img2.max() - img2.min() + 1e-8)
        
        # Simple alpha blending
        overlay = alpha * img1_norm + (1 - alpha) * img2_norm
        
        return overlay
    
    def save_registered_image(self, output_path='registered_maldi.tif'):
        """
        Save the final registered MALDI image.
        
        Parameters:
        -----------
        output_path : str
            Path where to save the registered image
        """
        if self.registered_nonrigid is not None:
            # Convert to 8-bit for saving
            output = (self.registered_nonrigid * 255).astype(np.uint8)
            cv2.imwrite(output_path, output)
            print(f"\nSaved registered image to '{output_path}'")
        else:
            print(f"No registered image available yet. Run the full pipeline first.")


def run_registration_pipeline(he_path, maldi_path, n_landmarks=5):
    """
    Complete registration pipeline from start to finish.
    
    Parameters:
    -----------
    he_path : str
        Path to H&E image
    maldi_path : str
        Path to MALDI image
    n_landmarks : int
        Number of landmark pairs to use (default: 5)
    
    Returns:
    --------
    registration : MALDIRegistration
        Registration object with all results
    """
    print(f"\n{'='*60}")
    print(f"MALDI-MSI TO H&E REGISTRATION PIPELINE")
    print(f"{'='*60}\n")
    
    # Step 1: Initialize and load images
    print(f"Step 1/5: Loading and preprocessing images...")
    reg = MALDIRegistration(he_path, maldi_path)
    
    # Step 2: Manual landmark selection
    print(f"\nStep 2/5: Manual landmark selection...")
    reg.select_landmarks(n_points=n_landmarks)
    
    # Step 3: Compute initial affine transformation
    print(f"\nStep 3/5: Computing affine transformation...")
    reg.compute_affine_transform()
    
    # Step 4: Refine affine using boundary optimization
    print(f"\nStep 4/5: Refining affine transformation...")
    reg.refine_affine()
    
    # Step 5: Apply non-rigid deformation
    print(f"\nStep 5/5: Applying non-rigid deformation...")
    reg.apply_nonrigid_deformation()
    
    # Visualize results
    print(f"\nGenerating visualization...")
    reg.visualize_results()
    
    # Save result
    reg.save_registered_image()
    
    print(f"\n{'='*60}")
    print(f"REGISTRATION COMPLETE!")
    print(f"{'='*60}\n")
    
    return reg


# Example usage:
if __name__ == "__main__":
    # Replace these paths with your actual file paths
    HE_PATH = "/home/krastegar0/Lipidomics/img_folder/old_liver_10x.tiff"
    MALDI_PATH = "/home/krastegar0/Lipidomics/img_folder/Taurine_img_withoutborders.tif"
    
    # Run the complete pipeline
    # This will:
    # 1. Load images
    # 2. Let you select landmarks interactively
    # 3. Compute and refine affine transformation
    # 4. Apply non-rigid deformation
    # 5. Show and save results
    
    registration = run_registration_pipeline(
        he_path=HE_PATH,
        maldi_path=MALDI_PATH,
        n_landmarks=5  # You can adjust this (4-6 recommended)
    )
    
    # Access results:
    # registration.registered_affine - image after affine only
    # registration.registered_nonrigid - final registered image
    # registration.affine_matrix - transformation matrix

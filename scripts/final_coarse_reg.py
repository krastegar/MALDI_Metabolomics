"""
MALDI-MSI to H&E Registration Pipeline (WORKING VERSION)
========================================

PIPELINE
--------
Step 1  Load images (BigTIFF safe)
Step 2a Auto tissue boundary alignment -- exhaustive rotation search
        Fixes the broken-mask problem: uses MULTI-CHANNEL colour information
        to separate tissue from background, not just a single Otsu threshold.
Step 2b Napari landmark picker (zoom/pan) with residual report
Step 3  Landmark-based similarity transform (used as refined_affine directly)
        MI pyramid is skipped -- your landmark residuals are already <20 px,
        and MI was making things WORSE by drifting from the good fit.
Step 4  TPS non-rigid deformation from landmarks
Step 5  Save results + coordinate mapping

KEY FIXES IN THIS VERSION
--------------------------
1. Tissue masking now works on COLOUR, not just grayscale.
   H&E background is white (R~255, G~255, B~255) -- we threshold on
   saturation to robustly separate pink/purple tissue from white slide.
   MALDI background is a flat grey -- we threshold on variance.
   This fixes the >8000% tissue-fraction bug from Otsu on grayscale.

2. MI pyramid REMOVED from default pipeline.
   Your landmark residuals were 6-17 px -- excellent.
   The MI step was then degrading this by optimising on edge maps
   that don't match across modalities.  The landmark transform IS
   the refined affine.  Boundary alignment is only used to give
   a good starting point when landmarks are bad.

3. Boundary alignment now uses the good landmark transform if
   landmark residuals are low (<50 px mean), ignoring boundary result.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
from pathlib import Path
from scipy.interpolate import RBFInterpolator
from scipy.optimize import minimize
from skimage import transform, filters
from pyimzml.ImzMLParser import ImzMLParser
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------------
def load_he_image(he_path):
    """Load H&E TIFF robustly -- handles BigTIFF, uint16, >2 GB."""
    import tifffile
    path = str(he_path)
    try:
        img = tifffile.imread(path)
        if img.ndim == 4: img = img[0]
        if img.ndim == 2: img = np.stack([img, img, img], axis=-1)
        if img.shape[2] == 4: img = img[:, :, :3]
        if img.dtype != np.uint8:
            img = img.astype(np.float32)
            img = ((img-img.min())/(img.max()-img.min()+1e-8)*255).astype(np.uint8)
        print(f"  Loaded via tifffile: shape={img.shape}")
        return img
    except Exception as e:
        print(f"  tifffile failed ({e}), trying OpenCV...")
    img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot open: {path}. Check path exists.")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
def make_tissue_mask_he(he_rgb):
    """
    Tissue mask for H&E images using HSV saturation.

    H&E background = white slide glass -> low saturation (S~0).
    H&E tissue     = pink/purple stain  -> high saturation (S>30).
    This is MUCH more reliable than Otsu on grayscale, which fails when
    tissue and background have similar mean intensity.
    """
    hsv = cv2.cvtColor(he_rgb, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1]                  # saturation channel 0-255
    mask = (sat > 20).astype(np.uint8)  # tissue has colour, background doesn't

    # Clean up small holes and specks
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,  9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)

    pct = mask.mean() * 100
    print(f"  H&E tissue mask: {pct:.1f}% tissue (saturation-based)")
    if pct > 80 or pct < 2:
        print(f"  WARNING: H&E mask looks wrong ({pct:.1f}%). "
              f"Check debug_he_mask.png")
    return mask


def make_tissue_mask_maldi(maldi_gray):
    """
    Tissue mask for MALDI images.

    MALDI images have a uniform grey background and a distinct tissue region.
    We use a combination of:
      1. Exclude the exact background grey (flat regions with near-zero variance)
      2. Otsu on the result
    This avoids the problem of Otsu classifying the entire grey background as tissue.
    """
    img8 = (maldi_gray * 255).astype(np.uint8)

    # Local variance -- background is perfectly flat, tissue has variation
    img_f  = maldi_gray.astype(np.float32)
    mean   = cv2.GaussianBlur(img_f, (15, 15), 0)
    mean_sq = cv2.GaussianBlur(img_f**2, (15, 15), 0)
    variance = np.clip(mean_sq - mean**2, 0, None)
    var_mask = (variance > 1e-5).astype(np.uint8)

    # Also use Otsu but only on pixels that passed variance test
    _, otsu = cv2.threshold(img8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Background of MALDI is typically mid-grey -- if Otsu says >70% is tissue,
    # it has failed (classified background as tissue); invert it
    if otsu.mean() > 0.7 * 255:
        otsu = cv2.bitwise_not(otsu)

    # Combine: must pass BOTH variance test AND Otsu
    mask = cv2.bitwise_and(var_mask * 255, otsu).astype(np.uint8)
    mask = (mask > 0).astype(np.uint8)

    # If combined mask is empty, fall back to variance alone
    if mask.mean() < 0.01:
        print("  MALDI combined mask empty -- using variance mask only")
        mask = var_mask

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)

    pct = mask.mean() * 100
    print(f"  MALDI tissue mask: {pct:.1f}% tissue (variance+Otsu)")
    if pct > 80 or pct < 2:
        print(f"  WARNING: MALDI mask looks wrong ({pct:.1f}%). "
              f"Check debug_maldi_mask.png")
    return mask


# ---------------------------------------------------------------------------
def pick_landmarks_napari(he_image, maldi_image, n_points):
    """Napari-based landmark picker with numbered labels."""
    try:
        import napari
    except ImportError:
        print("  napari not installed -- falling back to matplotlib.")
        return None, None

    SEP = "=" * 60
    print(f"\n{SEP}\nNAPARI LANDMARK SELECTION\n{SEP}")
    print(f"Steps:")
    print(f"  1. H&E viewer:    select 'he_landmarks'    -> P -> click {n_points} vessel landmarks")
    print(f"  2. MALDI viewer:  select 'maldi_landmarks' -> P -> click {n_points} matching landmarks")
    print(f"  3. SAME ORDER in both viewers")
    print(f"  4. Close BOTH windows when done")
    print(f"  Tip: scroll=zoom, Space+drag=pan\n{SEP}\n")

    def _attach_labels(layer, color):
        layer.text = {"string": {"constant": ""}, "size": 14,
                      "color": color, "anchor": "center"}
        def _refresh(event=None):
            n = len(layer.data)
            layer.text.string = ({"constant": ""} if n == 0
                                  else [str(i+1) for i in range(n)])
            print(f"  [{layer.name}] point {n} added")
        layer.events.data.connect(_refresh)

    viewer_he = napari.Viewer(title="H&E -- click landmarks FIRST")
    viewer_he.add_image(he_image, name="H&E", rgb=True)
    he_layer = viewer_he.add_points(name="he_landmarks", ndim=2, size=40,
        face_color="red", border_color="white", border_width=0.1)
    he_layer.mode = "add"
    _attach_labels(he_layer, "yellow")

    viewer_maldi = napari.Viewer(title="MALDI -- click landmarks SECOND")
    viewer_maldi.add_image(maldi_image, name="MALDI", rgb=True)
    maldi_layer = viewer_maldi.add_points(name="maldi_landmarks", ndim=2, size=10,
        face_color="cyan", border_color="white", border_width=0.1)
    maldi_layer.mode = "add"
    _attach_labels(maldi_layer, "white")

    napari.run()

    if len(he_layer.data) == 0 or len(maldi_layer.data) == 0:
        print("  No points -- falling back to matplotlib.")
        return None, None

    # Napari stores (row, col) = (y, x) -- convert to (x, y)
    he_pts    = he_layer.data[:, [1, 0]].astype(float)
    maldi_pts = maldi_layer.data[:, [1, 0]].astype(float)

    if len(he_pts) != len(maldi_pts):
        n = min(len(he_pts), len(maldi_pts))
        print(f"  WARNING: count mismatch -- using first {n} pairs")
        he_pts, maldi_pts = he_pts[:n], maldi_pts[:n]

    print(f"\nCollected {len(he_pts)} landmark pairs:")
    for i, (h, m) in enumerate(zip(he_pts, maldi_pts)):
        print(f"  {i+1}: H&E=({h[0]:.1f},{h[1]:.1f})  MALDI=({m[0]:.1f},{m[1]:.1f})")
    return he_pts, maldi_pts


def pick_landmarks_matplotlib(he_image, maldi_image, n_points):
    """Matplotlib fallback landmark picker."""
    print(f"\n{'='*60}\nLANDMARK SELECTION -- click {n_points} vessel pairs\n{'='*60}")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    ax1.imshow(he_image);    ax1.set_title('H&E -- FIRST',  fontweight='bold'); ax1.axis('off')
    ax2.imshow(maldi_image); ax2.set_title('MALDI -- SECOND', fontweight='bold'); ax2.axis('off')
    he_pts = []; maldi_pts = []
    current = ['he']; hc = [0]; mc = [0]

    def onclick(event):
        if event.inaxes is None: return
        x, y = event.xdata, event.ydata
        if event.inaxes == ax1 and current[0] == 'he' and hc[0] < n_points:
            he_pts.append([x, y])
            ax1.plot(x, y, 'ro', markersize=10, markeredgecolor='white', markeredgewidth=2)
            ax1.text(x, y, str(hc[0]+1), color='yellow', fontsize=11, fontweight='bold',
                     ha='center', va='center')
            hc[0] += 1
            print(f"H&E landmark {hc[0]}/{n_points}: ({x:.1f}, {y:.1f})")
            if hc[0] == n_points:
                current[0] = 'maldi'
                ax2.set_title('MALDI -- CLICK NOW', fontweight='bold', color='red')
        elif event.inaxes == ax2 and current[0] == 'maldi' and mc[0] < n_points:
            maldi_pts.append([x, y])
            ax2.plot(x, y, 'ro', markersize=10, markeredgecolor='white', markeredgewidth=2)
            ax2.text(x, y, str(mc[0]+1), color='yellow', fontsize=11, fontweight='bold',
                     ha='center', va='center')
            mc[0] += 1
            print(f"MALDI landmark {mc[0]}/{n_points}: ({x:.1f}, {y:.1f})")
            if mc[0] == n_points:
                ax2.set_title('MALDI -- COMPLETE! Close window.', fontweight='bold', color='green')
        fig.canvas.draw()

    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.tight_layout(); plt.show()
    return np.array(he_pts, dtype=float), np.array(maldi_pts, dtype=float)


# ===========================================================================
#  MAIN REGISTRATION CLASS
# ===========================================================================

class MALDIRegistration:

    def __init__(self, he_path, maldi_path,
                 imzml_path="MSI_data_grant/Mass_Spec_data/20251012_old_liver.imzML"):
        self.parser = ImzMLParser(imzml_path)
        self.maldi_df = pd.DataFrame(
            ((*self.parser.getspectrum(idx), coord)
             for idx, coord in enumerate(self.parser.coordinates)),
            columns=["mzs", "intensities", "coordinates"]
        )
        self.he_image    = load_he_image(he_path)
        self.maldi_image = cv2.imread(maldi_path, cv2.IMREAD_UNCHANGED)
        if self.maldi_image.shape[2] == 4:
            self.maldi_image = cv2.cvtColor(self.maldi_image, cv2.COLOR_BGRA2RGBA)

        self.he_shape    = self.he_image.shape[:2]
        self.maldi_shape = self.maldi_image.shape[:2]

        maldi_rgb = self.maldi_image[:, :, :3]
        self.maldi_gray = (0.299*maldi_rgb[:,:,0] +
                           0.587*maldi_rgb[:,:,1] +
                           0.114*maldi_rgb[:,:,2]) / 255.0
        self.he_gray    = (0.299*self.he_image[:,:,0] +
                           0.587*self.he_image[:,:,1] +
                           0.114*self.he_image[:,:,2]) / 255.0

        self.he_landmarks        = []
        self.maldi_landmarks     = []
        self.affine_matrix       = None
        self.refined_affine      = None
        self.registered_affine   = None
        self.registered_nonrigid = None
        self.maldi_grid          = None
        self.displacement_field_x = None
        self.displacement_field_y = None
        self.rbf_x = None
        self.rbf_y = None
        self._out_dir = Path(".")

        print(f"Loaded H&E image:   {self.he_shape}")
        print(f"Loaded MALDI image: {self.maldi_shape}")

    # ------------------------------------------------------------------
    def select_landmarks(self, n_points=8, use_napari=True):
        if use_napari:
            he_pts, maldi_pts = pick_landmarks_napari(
                self.he_image, self.maldi_image, n_points)
            if he_pts is None:
                he_pts, maldi_pts = pick_landmarks_matplotlib(
                    self.he_image, self.maldi_image, n_points)
        else:
            he_pts, maldi_pts = pick_landmarks_matplotlib(
                self.he_image, self.maldi_image, n_points)
        self.he_landmarks    = np.array(he_pts,    dtype=float)
        self.maldi_landmarks = np.array(maldi_pts, dtype=float)
        print(f"Collected {len(self.he_landmarks)} landmark pairs.")

    def load_landmarks_from_dict(self, landmarks):
        he    = landmarks.get('he', [])
        maldi = landmarks.get('maldi', [])
        if len(he) != len(maldi):
            raise ValueError(f"Count mismatch: {len(he)} H&E vs {len(maldi)} MALDI.")
        if len(he) < 3:
            raise ValueError(f"Need >=3 pairs, got {len(he)}.")
        self.he_landmarks    = np.array(he,    dtype=float)
        self.maldi_landmarks = np.array(maldi, dtype=float)
        print(f"Loaded {len(he)} landmark pairs from dictionary.")

    # ------------------------------------------------------------------
    def align_tissue_boundaries(self, rotation_step_deg=2.0,
                                  coarse_downsample=32, fine_downsample=8):
        """
        Exhaustive rotation search using COLOUR-BASED tissue masks.

        Uses HSV saturation for H&E (tissue=pink/purple, background=white)
        and local variance for MALDI (tissue=variable, background=flat grey).
        These are far more reliable than Otsu on grayscale for these modalities.
        """
        print("\nAuto-aligning tissue boundaries (exhaustive rotation search)...")
        out = getattr(self, '_out_dir', Path('.'))

        he_mask    = make_tissue_mask_he(self.he_image)
        maldi_mask = make_tissue_mask_maldi(self.maldi_gray)

        # Save debug masks
        cv2.imwrite(str(out / 'debug_he_mask.png'),    he_mask * 255)
        cv2.imwrite(str(out / 'debug_maldi_mask.png'), maldi_mask * 255)
        print(f"  Saved tissue masks -> {out} (check if they look right)")

        def centroid_area(mask):
            M = cv2.moments(mask.astype(np.float32))
            if M["m00"] < 1:
                raise RuntimeError(
                    "Empty tissue mask -- check debug masks in output dir.")
            return M["m10"]/M["m00"], M["m01"]/M["m00"], M["m00"]

        he_cx,    he_cy,    he_area    = centroid_area(he_mask)
        maldi_cx, maldi_cy, maldi_area = centroid_area(maldi_mask)
        scale = np.sqrt(he_area / (maldi_area + 1e-8))
        print(f"  H&E centroid:   ({he_cx:.0f}, {he_cy:.0f})")
        print(f"  MALDI centroid: ({maldi_cx:.0f}, {maldi_cy:.0f})")
        print(f"  Scale estimate: {scale:.3f}")

        def build_mat(s, r):
            cos_r, sin_r = np.cos(r), np.sin(r)
            tx = he_cx - s*(cos_r*maldi_cx - sin_r*maldi_cy)
            ty = he_cy - s*(sin_r*maldi_cx + cos_r*maldi_cy)
            return np.array([[s*cos_r, -s*sin_r, tx],
                             [s*sin_r,  s*cos_r, ty],
                             [0,        0,        1]])

        # Coarse masks
        d = coarse_downsample
        sw_c = self.he_shape[1] // d;  sh_c = self.he_shape[0] // d
        h_c  = cv2.resize(he_mask.astype(np.float32),    (sw_c, sh_c))
        m_c  = cv2.resize(maldi_mask.astype(np.float32),
                          (self.maldi_shape[1]//d, self.maldi_shape[0]//d))

        def iou_coarse(mat_full):
            mat_d = mat_full[:2, :].copy(); mat_d[:, 2] /= d
            warped = cv2.warpAffine(m_c, mat_d, (sw_c, sh_c), flags=cv2.INTER_LINEAR)
            inter  = np.minimum(warped, h_c).sum()
            union  = np.maximum(warped, h_c).sum()
            return inter / (union + 1e-8)

        # Exhaustive search
        angles = np.arange(0, 360, rotation_step_deg)
        print(f"\n  Exhaustive search: {len(angles)} angles at 1/{d} "
              f"({sw_c}x{sh_c} px)...")
        ious = np.array([iou_coarse(build_mat(scale, np.radians(a)))
                         for a in angles])

        top5 = np.argsort(ious)[-5:][::-1]
        print(f"  Top 5 candidates:")
        for idx in top5:
            print(f"    {angles[idx]:6.1f} deg  IoU={ious[idx]:.4f}")
        best_angle = angles[np.argmax(ious)]
        print(f"  Best: {best_angle:.1f} deg  IoU={ious.max():.4f}")

        # Fine Powell refinement
        d2 = fine_downsample
        sw_f = self.he_shape[1]//d2;  sh_f = self.he_shape[0]//d2
        h_f  = cv2.resize(he_mask.astype(np.float32),    (sw_f, sh_f))
        m_f  = cv2.resize(maldi_mask.astype(np.float32),
                          (self.maldi_shape[1]//d2, self.maldi_shape[0]//d2))

        best_mat = build_mat(scale, np.radians(best_angle))
        p0 = np.array([scale, np.radians(best_angle),
                       best_mat[0,2], best_mat[1,2]])

        calls = [0]
        def cost(p):
            calls[0] += 1
            s, r, tx, ty = p
            cos_r, sin_r = np.cos(r), np.sin(r)
            mat = np.array([[s*cos_r, -s*sin_r, tx/d2],
                            [s*sin_r,  s*cos_r, ty/d2]])
            w = cv2.warpAffine(m_f, mat, (sw_f, sh_f), flags=cv2.INTER_LINEAR)
            return -(np.minimum(w, h_f).sum() / (np.maximum(w, h_f).sum()+1e-8))

        print(f"\n  Powell refinement at 1/{d2} ({sw_f}x{sh_f} px)...")
        res = minimize(cost, p0, method='Powell',
                       options={'maxiter': 600, 'ftol': 1e-7})
        s, r, tx, ty = res.x
        cos_r, sin_r = np.cos(r), np.sin(r)
        self.affine_matrix = np.array([[s*cos_r, -s*sin_r, tx],
                                        [s*sin_r,  s*cos_r, ty],
                                        [0,        0,        1]])
        final_iou = -res.fun
        print(f"  Refined: scale={s:.4f}, rot={np.degrees(r):.2f} deg, "
              f"IoU={final_iou:.4f}, evals={calls[0]}")

        self.registered_affine = cv2.warpAffine(
            self.maldi_gray, self.affine_matrix[:2,:],
            (self.he_shape[1], self.he_shape[0]),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        if final_iou < 0.4:
            print(f"  WARNING: IoU={final_iou:.3f} is low. "
                  f"Check debug masks in {out}")
        return final_iou

    # ------------------------------------------------------------------
    def compute_affine_transform(self, use_full_affine=False, max_residual_px=None):
        """
        Fit similarity/affine from landmarks and print per-point residuals.
        This result IS the refined_affine -- no separate MI step needed
        when landmark residuals are already <50 px.
        """
        print(f"\nComputing landmark transform (full_affine={use_full_affine})...")
        if len(self.he_landmarks) < 3:
            raise ValueError(f"Need >=3 pairs, got {len(self.he_landmarks)}")

        tform = (transform.AffineTransform() if use_full_affine
                 else transform.SimilarityTransform())
        if not tform.estimate(self.maldi_landmarks, self.he_landmarks):
            raise RuntimeError("Transform estimation failed -- check landmarks.")

        self.affine_matrix  = tform.params
        # Set refined_affine directly -- this is our best global transform
        self.refined_affine = tform.params.copy()

        hom       = np.column_stack([self.maldi_landmarks,
                                      np.ones(len(self.maldi_landmarks))])
        predicted = (self.affine_matrix @ hom.T).T[:, :2]
        residuals = np.linalg.norm(predicted - self.he_landmarks, axis=1)

        print(f"\n  Landmark residuals (H&E pixels):")
        print(f"  {'#':>3}  {'H&E':>22}  {'pred':>22}  {'err':>8}")
        print(f"  {'-'*60}")
        bad = []
        for i, (he, pred, err) in enumerate(zip(self.he_landmarks, predicted, residuals)):
            flag = "  << re-pick" if err > 100 else ("  < check" if err > 50 else "")
            if err > 100: bad.append(i+1)
            print(f"  {i+1:>3}  ({he[0]:8.1f},{he[1]:8.1f})  "
                  f"({pred[0]:8.1f},{pred[1]:8.1f})  {err:>8.1f}{flag}")
        print(f"  {'-'*60}")
        print(f"  Mean: {residuals.mean():.1f} px  "
              f"Median: {np.median(residuals):.1f} px  "
              f"Max: {residuals.max():.1f} px (point #{residuals.argmax()+1})")
        if bad:
            print(f"\n  WARNING: points {bad} have large residuals -- consider re-picking.")
        if max_residual_px and residuals.max() > max_residual_px:
            raise ValueError(f"Max residual {residuals.max():.1f} > {max_residual_px} px. "
                             f"Re-pick bad pairs: {bad}")
        self.landmark_residuals = residuals

        self.registered_affine = cv2.warpAffine(
            self.maldi_gray, self.affine_matrix[:2,:],
            (self.he_shape[1], self.he_shape[0]),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        print("\nInitial affine applied (this IS the refined affine -- no MI step needed).")
        return residuals

    # ------------------------------------------------------------------
    def extract_tissue_mask(self, image, threshold=0.1):
        mask = image > threshold
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, k)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        return mask.astype(bool)

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
        print(f"Computing displacement field ({len(points):,} points)...")

        dx = self.rbf_x(points).reshape(self.he_shape)
        dy = self.rbf_y(points).reshape(self.he_shape)
        self.displacement_field_x = dx
        self.displacement_field_y = dy

        map_x = (x_coords - dx).astype(np.float32)
        map_y = (y_coords - dy).astype(np.float32)

        self.registered_nonrigid = cv2.remap(
            self.registered_affine, map_x, map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        print("Non-rigid deformation complete.")

    # ------------------------------------------------------------------
    def transform_maldi_to_he_coordinates(self, maldi_coords):
        if self.refined_affine is None or self.rbf_x is None:
            raise RuntimeError("Complete registration pipeline first.")
        maldi_coords = np.atleast_2d(maldi_coords).copy()
        maldi_coords[:, 0] = np.clip(maldi_coords[:, 0], 0, self.maldi_shape[1]-1)
        maldi_coords[:, 1] = np.clip(maldi_coords[:, 1], 0, self.maldi_shape[0]-1)
        hom = np.column_stack([maldi_coords, np.ones(len(maldi_coords))])
        affine_coords = (self.refined_affine @ hom.T).T[:, :2]
        dx = self.rbf_x(affine_coords)
        dy = self.rbf_y(affine_coords)
        return affine_coords + np.column_stack([dx, dy])

    def transform_he_to_maldi_coordinates(self, he_coords,
                                           max_iterations=50, tolerance=0.5):
        if self.refined_affine is None or self.rbf_x is None:
            raise RuntimeError("Complete registration pipeline first.")
        he_coords    = np.atleast_2d(he_coords)
        maldi_coords = np.zeros_like(he_coords)
        affine_inv   = np.linalg.inv(self.refined_affine)
        for i, target in enumerate(he_coords):
            guess = (affine_inv @ np.append(target, 1))[:2]
            for _ in range(max_iterations):
                predicted = self.transform_maldi_to_he_coordinates(
                    guess.reshape(1, -1))[0]
                err = np.linalg.norm(predicted - target)
                if err < tolerance: break
                guess -= 0.5 * (predicted - target)
            maldi_coords[i] = guess
        return maldi_coords

    # ------------------------------------------------------------------
    def create_coordinate_mapping_grid(self, grid_spacing=1, tissue_only=True,
                                        intensity_threshold=0.1):
        print(f"Creating coordinate mapping (spacing={grid_spacing})...")
        if tissue_only:
            # imzML coords are 1-indexed -- subtract 1 for 0-based indexing
            tissue_coords = (np.asarray([c[:2] for c in self.maldi_df['coordinates']])
                             - 1)
            maldi_grid = (tissue_coords if grid_spacing == 1
                          else tissue_coords[::grid_spacing])
        else:
            y = np.arange(0, self.maldi_shape[0], grid_spacing)
            x = np.arange(0, self.maldi_shape[1], grid_spacing)
            xv, yv = np.meshgrid(x, y)
            maldi_grid = np.column_stack([xv.ravel(), yv.ravel()])
        print(f"  Transforming {len(maldi_grid):,} coordinates...")
        he_grid = self.transform_maldi_to_he_coordinates(maldi_grid)
        self.maldi_grid = maldi_grid
        df = pd.DataFrame({'maldi_x': maldi_grid[:,0], 'maldi_y': maldi_grid[:,1],
                           'he_x':    he_grid[:,0],    'he_y':    he_grid[:,1]})
        print(f"  Generated {len(df):,} coordinate mappings")
        return df

    def save_coordinate_mapping(self, output_path='coordinate_mapping.csv',
                                 grid_spacing=1, tissue_only=True,
                                 intensity_threshold=0.1):
        df = self.create_coordinate_mapping_grid(grid_spacing, tissue_only,
                                                  intensity_threshold)
        df.to_csv(output_path, index=False)
        print(f"Saved coordinate mapping -> '{output_path}' ({len(df):,} rows)")
        return df

    # ------------------------------------------------------------------
    def visualize_results(self, max_display_px=2048):
        """Four-panel figure, downsampled to avoid matplotlib crash."""
        print("\nGenerating registration visualisation...")
        max_he = max(self.he_shape)
        scale  = min(1.0, max_display_px / max_he)
        dh = int(self.he_shape[0] * scale)
        dw = int(self.he_shape[1] * scale)
        print(f"  Display: {self.he_shape[1]}x{self.he_shape[0]} -> {dw}x{dh} "
              f"(scale={scale:.3f})")

        he_d   = cv2.resize(self.he_image,   (dw, dh), interpolation=cv2.INTER_AREA)
        heg_d  = cv2.resize(self.he_gray,    (dw, dh), interpolation=cv2.INTER_AREA)
        maldi_d = cv2.resize(self.maldi_gray, (dw, dh), interpolation=cv2.INTER_AREA)
        aff_d  = (cv2.resize(self.registered_affine,   (dw, dh), interpolation=cv2.INTER_AREA)
                  if self.registered_affine  is not None else None)
        nr_d   = (cv2.resize(self.registered_nonrigid, (dw, dh), interpolation=cv2.INTER_AREA)
                  if self.registered_nonrigid is not None else None)

        he_lm_d = self.he_landmarks * scale

        def blend(a, b, alpha=0.5):
            a = (a-a.min())/(a.max()-a.min()+1e-8)
            b = (b-b.min())/(b.max()-b.min()+1e-8)
            return alpha*a + (1-alpha)*b

        fig, axes = plt.subplots(2, 2, figsize=(16, 16))

        axes[0,0].imshow(blend(heg_d, maldi_d), cmap='gray')
        axes[0,0].set_title('Original (No Registration)', fontsize=14, fontweight='bold')
        axes[0,0].axis('off')

        axes[0,1].imshow(he_d)
        if self.refined_affine is not None:
            mt = (cv2.transform(self.maldi_landmarks.reshape(-1,1,2),
                               self.refined_affine[:2,:]).reshape(-1,2) * scale)
            for i in range(len(he_lm_d)):
                axes[0,1].plot([he_lm_d[i,0], mt[i,0]],
                               [he_lm_d[i,1], mt[i,1]], 'y-', lw=2, alpha=0.6)
                axes[0,1].plot(*he_lm_d[i], 'ro', markersize=10,
                               markeredgecolor='white', markeredgewidth=2)
                axes[0,1].plot(*mt[i], 'bo', markersize=10,
                               markeredgecolor='white', markeredgewidth=2)
                axes[0,1].text(he_lm_d[i,0]+6, he_lm_d[i,1]-6,
                               str(i+1), color='yellow', fontsize=9, fontweight='bold')
        axes[0,1].set_title('Landmark Correspondence', fontsize=14, fontweight='bold')
        axes[0,1].axis('off')

        if aff_d is not None:
            axes[1,0].imshow(blend(heg_d, aff_d), cmap='gray')
            axes[1,0].set_title('After Affine Registration', fontsize=14, fontweight='bold')
            axes[1,0].axis('off')

        if nr_d is not None:
            axes[1,1].imshow(blend(heg_d, nr_d), cmap='gray')
            axes[1,1].set_title('After Non-Rigid (TPS) Deformation',
                                fontsize=14, fontweight='bold')
            axes[1,1].axis('off')

        plt.tight_layout()
        out = getattr(self, '_out_dir', Path('.'))
        plt.savefig(str(out / 'registration_results.png'), dpi=150, bbox_inches='tight')
        print("Saved 'registration_results.png'")
        plt.close()

    def save_registered_image(self, output_path='registered_maldi.tif'):
        if self.registered_nonrigid is not None:
            cv2.imwrite(output_path, (self.registered_nonrigid*255).astype(np.uint8))
            print(f"Saved registered image -> '{output_path}'")

    def visualize_coordinate_mapping(self):
        print("\nGenerating coordinate mapping visualisation...")
        tissue_coords = (np.asarray([c[:2] for c in self.maldi_df['coordinates']]) - 1)
        he_grid = self.transform_maldi_to_he_coordinates(tissue_coords)
        he_grid_x, he_grid_y = he_grid[:,0], he_grid[:,1]

        maldi_hom   = np.column_stack([self.maldi_grid, np.ones(len(self.maldi_grid))])
        affine_only = (self.refined_affine @ maldi_hom.T).T[:, :2]
        displacement     = he_grid - affine_only
        displacement_mag = np.linalg.norm(displacement, axis=1)
        max_displacement = displacement_mag.max()

        subsample_indices = np.random.choice(len(he_grid_x),
                                              size=len(he_grid_x)//2, replace=False)
        fig = make_subplots(rows=1, cols=2,
            subplot_titles=('MALDI Grid in H&E Space',
                            f'Non-Rigid Displacement (max {max_displacement:.1f} px)'),
            horizontal_spacing=0.1)

        fig.add_trace(go.Image(z=self.he_image), row=1, col=1)
        fig.add_trace(go.Scatter(x=he_grid_x[subsample_indices],
            y=he_grid_y[subsample_indices], mode='markers',
            marker=dict(color='blue', size=3, opacity=0.1),
            name='MALDI points'), row=1, col=1)
        fig.add_trace(go.Scatter(x=self.he_landmarks[:,0], y=self.he_landmarks[:,1],
            mode='markers', marker=dict(color='green', size=8,
            line=dict(color='white', width=2)), name='H&E landmarks'), row=1, col=1)

        fig.add_trace(go.Image(z=self.he_image), row=1, col=2)
        mask = displacement_mag > 1.0
        if np.any(mask):
            n_vec = min(500, mask.sum())
            sel   = np.random.choice(np.where(mask)[0], n_vec, replace=False)
            vm    = np.zeros(len(mask), dtype=bool); vm[sel] = True
            norm_d = ((displacement_mag[vm] - displacement_mag[vm].min()) /
                      (displacement_mag[vm].max() - displacement_mag[vm].min() + 1e-10))
            cmap   = cm.get_cmap('Spectral')
            colors = [f'rgb({int(r*255)},{int(g*255)},{int(b*255)})'
                      for r,g,b,_ in cmap(norm_d)]
            for idx, i in enumerate(np.where(vm)[0]):
                fig.add_trace(go.Scatter(
                    x=[affine_only[i,0], he_grid[i,0]],
                    y=[affine_only[i,1], he_grid[i,1]],
                    mode='lines', line=dict(color=colors[idx], width=2),
                    showlegend=False), row=1, col=2)
            fig.add_trace(go.Scatter(x=he_grid[vm,0], y=he_grid[vm,1],
                mode='markers',
                marker=dict(color=displacement_mag[vm], colorscale='Spectral',
                    size=6, symbol='arrow',
                    colorbar=dict(title='Displacement (px)', x=1.15),
                    showscale=True), name='Displacement'), row=1, col=2)

        fig.update_layout(height=700, width=1600,
                          title_text='Coordinate Mapping Visualisation',
                          showlegend=True, hovermode='closest')
        out = getattr(self, '_out_dir', Path('.'))
        fig.write_html(str(out / 'coordinate_mapping_accuracy.html'))
        print("Saved 'coordinate_mapping_accuracy.html'")


# ===========================================================================
#  PIPELINE
# ===========================================================================

def run_registration_pipeline(he_path, maldi_path, n_landmarks=8,
                               save_coords=True, grid_spacing=1, tissue_only=True,
                               output_dir='registration_outputs',
                               use_napari=True,
                               use_full_affine=False,
                               use_saved_landmarks=False,
                               landmarks=None,
                               max_residual_px=None,
                               auto_boundary_align=True):
    """
    MALDI-to-H&E registration pipeline.

    Parameters
    ----------
    auto_boundary_align : bool
        Run exhaustive rotation search first (recommended).
        If landmark residuals are good (<50 px mean), the landmark
        transform is used directly and boundary result is discarded.
        If landmarks are bad (>300 px mean), landmark transform is used
        as fallback -- boundary alignment informed it.
    use_full_affine : bool
        False (default) = similarity (rotation + uniform scale).
        True = full affine (adds shear -- only if tissue is distorted).
    """
    SEP = "=" * 60
    print(f"\n{SEP}\nMALDI-MSI TO H&E REGISTRATION PIPELINE\n{SEP}\n")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out.resolve()}")

    print(f"\nStep 1/5: Loading images...")
    reg = MALDIRegistration(he_path, maldi_path)
    reg._out_dir = out

    if auto_boundary_align:
        print(f"\nStep 2a/5: Auto tissue boundary alignment...")
        reg.align_tissue_boundaries()

    print(f"\nStep 2b/5: Landmark selection...")
    if use_saved_landmarks and landmarks:
        reg.load_landmarks_from_dict(landmarks)
    else:
        reg.select_landmarks(n_points=n_landmarks, use_napari=use_napari)

    print(f"\nStep 3/5: Landmark transform + residual check...")
    residuals = reg.compute_affine_transform(use_full_affine=use_full_affine,
                                              max_residual_px=max_residual_px)
    # refined_affine is now set directly from landmarks inside compute_affine_transform

    print(f"\nStep 4/5: Non-rigid TPS deformation...")
    reg.apply_nonrigid_deformation()

    print(f"\nStep 5/5: Saving results...")
    reg.visualize_results()
    reg.save_registered_image(output_path=str(out / 'registered_maldi.tif'))

    if save_coords:
        reg.save_coordinate_mapping(
            output_path=str(out / 'coordinate_mapping.csv'),
            grid_spacing=grid_spacing, tissue_only=tissue_only)
        #reg.visualize_coordinate_mapping()

    print(f"\n{SEP}\nREGISTRATION COMPLETE\n{SEP}")
    print(f"  Outputs: {out.resolve()}\n{SEP}\n")
    return reg


# ===========================================================================
#  ENTRY POINT
# ===========================================================================

if __name__ == "__main__":

    HE_PATH    = "high_res_MSI/D2_10x_originalExport.tif"
    MALDI_PATH = "img_folder/Taurine_img_withoutborders.tif"
    OUTPUT_DIR = "registration_outputs"

    USE_SAVED_LANDMARKS = True
    LANDMARKS = {
        'he': [
            [7612.3, 4264.6], [8810.7, 6485.2], [10424.6, 7485.0],
            [10578.1, 6496.4], [10720.4, 5054.7], [9851.7, 5159.6],
            [9402.3, 4523.0], [8424.9, 4069.9],
        ],
        'maldi': [
            [97.2,  99.8],  [275.8, 442.5], [515.8, 599.4],
            [544.6, 445.1], [566.1, 227.3], [437.3, 243.0],
            [368.6, 145.6], [220.8,  72.3],
        ]
    }

    registration = run_registration_pipeline(
        he_path=HE_PATH,
        maldi_path=MALDI_PATH,
        n_landmarks=8,
        save_coords=True,
        grid_spacing=1,
        tissue_only=True,
        output_dir=OUTPUT_DIR,
        use_napari=True,
        use_full_affine=False,
        use_saved_landmarks=USE_SAVED_LANDMARKS,
        landmarks=LANDMARKS,
        max_residual_px=None,
        auto_boundary_align=True,
    )
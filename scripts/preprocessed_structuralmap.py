#!/usr/bin/env python3
"""
maldi_he_register.py

Biologically constrained structural-field registration of MALDI MSI to H&E
histology, producing a MALDI-to-H&E coordinate mapping as the primary data
product.

WHY THIS APPROACH
  MALDI and H&E do not have reliable raw-intensity correspondence. At
  5 µm MALDI resolution, the biologically useful target is correct
  histological neighborhood alignment: vessels, sinusoids, anisotropic
  tissue organization, and local anatomical continuity. This script
  therefore registers structural fields rather than summed intensity
  reconstructions.

PIPELINE
  1. MALDI → NMF components, per-component denoising, structural fields.
  2. H&E → hematoxylin channel (color-deconvolved), structural fields.
  3. Build structural tissue masks for objective evaluation.
  4. Directly optimise vessel displacement + structural difference.
  5. Diagnostic visualisations and objective-score audit trail.
  6. Coordinate mapping export.

PRIMARY OUTPUTS (in --output_dir/)
  maldi_to_he_table.csv        coordinate mapping with columns:
                                maldi_x, maldi_y, he_x, he_y

DIAGNOSTIC OUTPUTS
  nmf_component_denoising.png  original vs denoised NMF components
  maldi_structural_fields.png  MALDI edge/orientation/anisotropy/vesselness
  he_structural_fields.png     H&E edge/orientation/anisotropy/vesselness
  he_at_registration_grid.png  H&E downsampled to MALDI grid
  vessel_overlay_pre/post.png  vessel-center alignment before/after transform
  vessel_displacement_qc.png   nearest-neighbor vessel displacement diagnostics
  registered_overlay.png       warped MALDI structural field over H&E field

DEPENDENCIES
  pip install pyimzml SimpleITK scikit-image scikit-learn numpy scipy pandas matplotlib

USAGE
  python maldi_he_register.py \\
      --imzml file.imzML --he he.png \\
      --output_dir results/ \\
      --maldi_um_per_pixel 5.0 \\
      [--he_um_per_pixel 0.5] [--n_components 8]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from scipy import optimize

from skimage import io as skio
from skimage.color import rgb2gray, rgb2hed
from skimage.filters import gaussian, median, sobel, frangi, threshold_otsu
from skimage.feature import structure_tensor
from skimage.measure import label, regionprops_table
from skimage.morphology import disk, remove_small_objects
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.transform import resize
from sklearn.decomposition import NMF
from sklearn.neighbors import NearestNeighbors
from pyimzml.ImzMLParser import ImzMLParser


# ──────────────────────────────────────────────────────────────────────────
# Structural-field registration config
# ──────────────────────────────────────────────────────────────────────────

DENOISE_METHOD = "gaussian"      # "gaussian", "median", "nl_means", or "none"
GAUSSIAN_SIGMA_PIXELS = 0.75     # MALDI pixels; keep small to preserve gradients
MEDIAN_RADIUS_PIXELS = 1
NL_MEANS_PATCH_SIZE = 3
NL_MEANS_PATCH_DISTANCE = 5
NL_MEANS_H_FACTOR = 0.8

STRUCTURAL_COMBINE_METHOD = "max"  # "max" or "weighted_mean"
REGISTRATION_FIELD = "edge"        # "edge", "vesselness", or "anisotropy"
STRUCTURE_TENSOR_SIGMA = 1.0
FRANGI_SIGMAS = (1.0, 2.0, 3.0)
FRANGI_ALPHA = 0.5
FRANGI_BETA = 0.5
FRANGI_GAMMA = None
VESSEL_THRESHOLD_PERCENTILE = 94.0
VESSEL_MIN_AREA = 8
VESSEL_MAX_AREA = 5000
VESSEL_MIN_ECCENTRICITY = 0.35
VESSEL_MIN_COUNT_FOR_OBJECTIVE = 8
TISSUE_MASK_THRESHOLD_PERCENTILE = 55.0
TISSUE_MASK_MIN_SIZE = 64

MAX_TRANSLATION_PIXELS = 30.0
MAX_ROTATION_DEGREES = 10.0
VESSEL_DISTANCE_SCALE_PIXELS = 10.0
MASK_DISTANCE_SCALE_PIXELS = 10.0
BIOLOGY_WEIGHT_VESSEL = 1.0
BIOLOGY_WEIGHT_STRUCTURAL = 1.0
BIOLOGY_WEIGHT_MASK_CHAMFER = 2.0
BIOLOGY_WEIGHT_TISSUE_OVERLAP = 1.5
BIOLOGY_WEIGHT_TRANSFORM_SIZE = 0.35
BIOLOGY_OPT_MAXITER = 35
BIOLOGY_OPT_POPSIZE = 8


# ──────────────────────────────────────────────────────────────────────────
# Stage 1 — MALDI: spectra → NMF components → structural fields
# ──────────────────────────────────────────────────────────────────────────

def normalize01(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32, copy=False)
    lo = float(np.nanmin(arr))
    hi = float(np.nanmax(arr))
    if hi > lo:
        return ((arr - lo) / (hi - lo)).astype(np.float32)
    return np.zeros_like(arr, dtype=np.float32)


def robust_normalize01(arr: np.ndarray,
                       lower: float = 1.0,
                       upper: float = 99.0) -> np.ndarray:
    """Percentile-normalize one image so outliers do not set the full scale."""
    arr = arr.astype(np.float32, copy=False)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros_like(arr, dtype=np.float32)
    lo, hi = np.percentile(finite, [lower, upper])
    if hi <= lo:
        return normalize01(arr)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def denoise_component(component: np.ndarray,
                      method: str = DENOISE_METHOD,
                      gaussian_sigma: float = GAUSSIAN_SIGMA_PIXELS,
                      median_radius: int = MEDIAN_RADIUS_PIXELS) -> np.ndarray:
    """Denoise one normalized NMF spatial component while preserving scale."""
    method = method.lower()
    if method in ("none", "off", "false"):
        denoised = component
    elif method == "gaussian":
        denoised = gaussian(component, sigma=gaussian_sigma,
                            preserve_range=True)
    elif method == "median":
        denoised = median(component, footprint=disk(median_radius))
    elif method in ("nl_means", "nlm", "nonlocal_means"):
        sigma = float(np.mean(estimate_sigma(component, channel_axis=None)))
        denoised = denoise_nl_means(
            component,
            h=NL_MEANS_H_FACTOR * max(sigma, 1e-6),
            patch_size=NL_MEANS_PATCH_SIZE,
            patch_distance=NL_MEANS_PATCH_DISTANCE,
            fast_mode=True,
            preserve_range=True,
            channel_axis=None,
        )
    else:
        raise ValueError(f"Unknown NMF denoise method: {method}")
    return np.clip(np.asarray(denoised, dtype=np.float32), 0.0, 1.0)


def compute_structural_fields(image: np.ndarray,
                              tensor_sigma: float = STRUCTURE_TENSOR_SIGMA,
                              frangi_sigmas: tuple = FRANGI_SIGMAS) -> dict:
    """
    Convert one image into structural fields.

    The fields describe tissue geometry rather than raw intensity:
      edge        Sobel gradient magnitude
      orientation undirected local gradient angle in radians
      anisotropy  structure-tensor coherence, high for oriented structures
      vesselness  Frangi response for line/vessel-like structures
    """
    img = robust_normalize01(image)
    gy, gx = np.gradient(img)
    edge = robust_normalize01(sobel(img))
    orientation = np.arctan2(gy, gx).astype(np.float32)

    Axx, Axy, Ayy = structure_tensor(img, sigma=tensor_sigma)
    trace = Axx + Ayy
    det_term = np.sqrt(np.maximum((Axx - Ayy) ** 2 + 4.0 * Axy ** 2, 0.0))
    lambda1 = 0.5 * (trace + det_term)
    lambda2 = 0.5 * (trace - det_term)
    anisotropy = (lambda1 - lambda2) / (lambda1 + lambda2 + 1e-8)
    anisotropy = robust_normalize01(anisotropy)

    # H&E vessels may appear dark while MALDI line-like structures can be
    # bright or dark depending on component polarity, so keep the stronger
    # response from both ridge assumptions.
    vessel_bright = frangi(
        img,
        sigmas=frangi_sigmas,
        alpha=FRANGI_ALPHA,
        beta=FRANGI_BETA,
        gamma=FRANGI_GAMMA,
        black_ridges=False,
    )
    vessel_dark = frangi(
        img,
        sigmas=frangi_sigmas,
        alpha=FRANGI_ALPHA,
        beta=FRANGI_BETA,
        gamma=FRANGI_GAMMA,
        black_ridges=True,
    )
    vesselness = robust_normalize01(np.maximum(vessel_bright, vessel_dark))

    return {
        "edge": edge.astype(np.float32),
        "orientation": orientation,
        "anisotropy": anisotropy.astype(np.float32),
        "vesselness": vesselness.astype(np.float32),
    }


def combine_component_structural_fields(component_fields: list,
                                        method: str = STRUCTURAL_COMBINE_METHOD) -> dict:
    """Combine per-component structural fields without averaging intensities."""
    method = method.lower()
    if method not in ("max", "weighted_mean"):
        raise ValueError("Structural combine method must be 'max' or 'weighted_mean'")

    edge_stack = np.stack([f["edge"] for f in component_fields], axis=0)
    anis_stack = np.stack([f["anisotropy"] for f in component_fields], axis=0)
    vessel_stack = np.stack([f["vesselness"] for f in component_fields], axis=0)

    if method == "max":
        edge = edge_stack.max(axis=0)
        anisotropy = anis_stack.max(axis=0)
        vesselness = vessel_stack.max(axis=0)
        weights = edge_stack * anis_stack
    else:
        weights = edge_stack * anis_stack + 1e-8
        edge = np.sum(edge_stack * weights, axis=0) / np.sum(weights, axis=0)
        anisotropy = np.sum(anis_stack * weights, axis=0) / np.sum(weights, axis=0)
        vesselness = np.sum(vessel_stack * weights, axis=0) / np.sum(weights, axis=0)

    # Orientations are axial, not directional: theta and theta + pi are
    # equivalent. Combine them with doubled-angle circular averaging.
    orientations = np.stack([f["orientation"] for f in component_fields], axis=0)
    weights = weights + 1e-8
    sin2 = np.sum(np.sin(2.0 * orientations) * weights, axis=0)
    cos2 = np.sum(np.cos(2.0 * orientations) * weights, axis=0)
    orientation = 0.5 * np.arctan2(sin2, cos2)

    return {
        "edge": robust_normalize01(edge),
        "orientation": orientation.astype(np.float32),
        "anisotropy": robust_normalize01(anisotropy),
        "vesselness": robust_normalize01(vesselness),
    }


def dense_mz_bins(mzs: np.ndarray,
                  intensities: np.ndarray,
                  pixel_indices: np.ndarray,
                  mz_tol: float,
                  min_count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Assign m/z values to dense-region bins using the DBSCAN-like 1D rule.

    Bins are contiguous sorted regions whose min_count-ahead m/z difference is
    below mz_tol. Sparse/noise regions and bins with fewer than min_count values
    are removed.
    """
    if min_count < 1:
        raise ValueError("min_count must be at least 1")

    order = np.argsort(mzs)
    sorted_mzs = mzs[order]
    sorted_intensities = intensities[order]
    sorted_pixel_indices = pixel_indices[order]
    n_values = len(sorted_mzs)

    if n_values <= min_count:
        return (
            np.array([], dtype=sorted_mzs.dtype),
            np.array([], dtype=sorted_intensities.dtype),
            np.array([], dtype=sorted_pixel_indices.dtype),
            np.array([], dtype=np.int64),
        )

    delta_values = -1 * sorted_mzs[:(n_values - min_count)] + sorted_mzs[min_count:n_values]
    delta_mz = np.concatenate([delta_values, np.zeros(min_count, dtype=sorted_mzs.dtype)])
    in_dense_region = (delta_mz < mz_tol).astype(np.int64)

    difference = np.diff(in_dense_region)
    difference_plus = np.maximum(difference, 0)
    running_bin_nr = np.cumsum(difference_plus)

    delta = in_dense_region
    bin_nr = delta[:-1] * running_bin_nr
    bin_labels = np.concatenate([[delta[0]], bin_nr]).astype(np.int64)

    keep = bin_labels != 0
    if not np.any(keep):
        return (
            np.array([], dtype=sorted_mzs.dtype),
            np.array([], dtype=sorted_intensities.dtype),
            np.array([], dtype=sorted_pixel_indices.dtype),
            np.array([], dtype=np.int64),
        )

    sorted_mzs = sorted_mzs[keep]
    sorted_intensities = sorted_intensities[keep]
    sorted_pixel_indices = sorted_pixel_indices[keep]
    bin_labels = bin_labels[keep]

    labels, counts = np.unique(bin_labels, return_counts=True)
    dense_labels = labels[counts >= min_count]
    keep = np.isin(bin_labels, dense_labels)

    return (
        sorted_mzs[keep],
        sorted_intensities[keep],
        sorted_pixel_indices[keep],
        bin_labels[keep],
    )


def build_maldi_data(imzml_path: str,
                     n_components: int = 8,
                     mz_bin_width: float = 0.1,
                     mz_bin_min_count: int = 100,
                     min_intensity: float = 5.0,
                     pixel_size_um: float = 5.0,
                     denoise_method: str = DENOISE_METHOD,
                     gaussian_sigma: float = GAUSSIAN_SIGMA_PIXELS,
                     median_radius: int = MEDIAN_RADIUS_PIXELS,
                     structural_combine: str = STRUCTURAL_COMBINE_METHOD,
                     registration_field: str = REGISTRATION_FIELD) -> dict:
    """
    Load MSI, run NMF, and build structural fields for registration.

    Returns dict with:
      components      [K, H, W] float32, raw per-component normalized [0, 1]
      denoised_components [K, H, W] float32, denoised components normalized [0, 1]
      structural_fields dict of MALDI edge/orientation/anisotropy/vesselness
      registration_image [H, W] selected structural field for registration
      component_spectra [K, n_bins] float32 - the H matrix of NMF
      mz_axis         [n_bins] float32 - bin-center m/z values
      pixel_size_um   float - physical size of one MALDI pixel
      grid_shape      (H, W)
      n_components    K
      grid_origin     (x0, y0) - offset of MALDI grid origin in imzML coords
    """
    parser = ImzMLParser(imzml_path)
    coords_list = list(parser.coordinates)
    xs = np.array([c[0] for c in coords_list])
    ys = np.array([c[1] for c in coords_list])
    x0, y0 = int(xs.min()), int(ys.min())
    H = int(ys.max() - y0 + 1)
    W = int(xs.max() - x0 + 1)
    n_px = len(coords_list)
    print(f"[1/7] Loaded {n_px} spectra | grid {H}×{W} | {pixel_size_um} μm/pixel")

    # First pass -- collect intensity-filtered peaks for dense m/z binning.
    all_mzs = []
    all_intensities = []
    all_pixel_indices = []
    for idx in range(n_px):
        mzs, ints = parser.getspectrum(idx)
        m = ints > min_intensity
        if m.any():
            all_mzs.append(mzs[m].astype(np.float64, copy=False))
            all_intensities.append(ints[m].astype(np.float32, copy=False))
            all_pixel_indices.append(np.full(int(m.sum()), idx, dtype=np.int64))
    if not all_mzs:
        raise RuntimeError("No spectra survived intensity threshold")

    all_mzs = np.concatenate(all_mzs)
    all_intensities = np.concatenate(all_intensities)
    all_pixel_indices = np.concatenate(all_pixel_indices)
    mz_min = float(all_mzs.min())
    mz_max = float(all_mzs.max())
    print(
        f"[1/7] m/z [{mz_min:.2f}, {mz_max:.2f}] | "
        f"dense bin tolerance {mz_bin_width} Da | min_count={mz_bin_min_count}"
    )

    binned_mzs, binned_intensities, binned_pixel_indices, bin_labels = dense_mz_bins(
        all_mzs,
        all_intensities,
        all_pixel_indices,
        mz_tol=mz_bin_width,
        min_count=mz_bin_min_count,
    )
    if len(bin_labels) == 0:
        raise RuntimeError(
            "Dense m/z binning produced no bins. Try increasing --mz_bin or "
            "lowering --mz_bin_min_count."
        )

    unique_bin_labels, bin_index = np.unique(bin_labels, return_inverse=True)
    n_bins = len(unique_bin_labels)
    bin_centers = np.bincount(bin_index, weights=binned_mzs) / np.bincount(bin_index)
    print(f"[1/7] Dense m/z binning kept {len(binned_mzs):,} values in {n_bins} bins")

    # Fill the [n_pixels, n_bins] matrix.
    X = np.zeros((n_px, n_bins), dtype=np.float32)
    np.add.at(X, (binned_pixel_indices, bin_index), binned_intensities)

    # Drop empty bins (saves memory and speeds up NMF)
    nonzero = X.sum(axis=0) > 0
    X = X[:, nonzero]
    bin_centers = bin_centers[nonzero]
    print(f"[1/7] Spectral matrix: {X.shape}")

    # NMF
    print(f"[2/7] NMF with k={n_components}")
    model = NMF(n_components=n_components, init="nndsvda",
                max_iter=500, random_state=42)
    W_mat = model.fit_transform(X)
    print(f"[2/7] NMF reconstruction error: {model.reconstruction_err_:.3f}")

    # Build per-component images, normalize each to [0, 1]
    comp_imgs = np.zeros((n_components, H, W), dtype=np.float32)
    rows = np.array([int(c[1] - y0) for c in coords_list], dtype=int)
    cols = np.array([int(c[0] - x0) for c in coords_list], dtype=int)
    for k in range(n_components):
        c = W_mat[:, k]
        c = c / (c.max() + 1e-8)
        comp_imgs[k, rows, cols] = c

    print(f"[2/7] Denoising NMF components with method='{denoise_method}'")
    denoised_comp_imgs = np.zeros_like(comp_imgs, dtype=np.float32)
    for k in range(n_components):
        denoised_comp_imgs[k] = denoise_component(
            comp_imgs[k],
            method=denoise_method,
            gaussian_sigma=gaussian_sigma,
            median_radius=median_radius,
        )

    print(f"[2/7] Computing MALDI structural fields from {n_components} NMF components")
    component_fields = [
        compute_structural_fields(denoised_comp_imgs[k])
        for k in range(n_components)
    ]
    structural_fields = combine_component_structural_fields(
        component_fields, method=structural_combine)
    if registration_field not in structural_fields or registration_field == "orientation":
        raise ValueError("registration_field must be 'edge', 'anisotropy', or 'vesselness'")

    return {
        "components":        comp_imgs.astype(np.float32),
        "denoised_components": denoised_comp_imgs.astype(np.float32),
        "component_structural_fields": component_fields,
        "structural_fields": structural_fields,
        "registration_image": structural_fields[registration_field].astype(np.float32),
        "component_spectra": model.components_.astype(np.float32),
        "mz_axis":           bin_centers.astype(np.float32),
        "pixel_size_um":     float(pixel_size_um),
        "grid_shape":        (H, W),
        "n_components":      int(n_components),
        "grid_origin":       (x0, y0),
    }


# ──────────────────────────────────────────────────────────────────────────
# Stage 2 — H&E: load, color-deconvolve, keep full-res AND resized copies
# ──────────────────────────────────────────────────────────────────────────

def prepare_he(he_path: str,
               target_shape: tuple,
               use_hematoxylin: bool = True,
               pixel_size_um: float = None,
               maldi_um_per_pixel: float = 5.0,
               registration_field: str = REGISTRATION_FIELD) -> dict:
    """
    Load H&E, optionally extract hematoxylin channel, and produce two
    versions: full resolution (preserved for cellular queries) and a
    downsampled copy on the MALDI grid (used for registration).

    Returns dict with:
      full_resolution      [H_full, W_full] float32 in [0, 1]
      registration_grid    [H, W] float32 in [0, 1] (matches MALDI grid)
      structural_fields    dict of H&E edge/orientation/anisotropy/vesselness
      registration_image   selected structural field for registration
      downsample_factor    (dy, dx) — full → registration scale per axis
      pixel_size_um        float — H&E pixel size, inferred if not given
    """
    rgb = skio.imread(he_path)
    if rgb.ndim == 3 and rgb.shape[2] == 4:
        rgb = rgb[..., :3]

    if rgb.ndim == 3 and use_hematoxylin:
        hed = rgb2hed(rgb)
        gray_full = hed[..., 0].astype(np.float32)        # hematoxylin
    elif rgb.ndim == 3:
        gray_full = 1.0 - rgb2gray(rgb).astype(np.float32)  # invert so tissue is bright
    else:
        gray_full = rgb.astype(np.float32)
    gray_full = (gray_full - gray_full.min()) / (gray_full.max() - gray_full.min() + 1e-8)

    # Infer pixel size from area-matching assumption if not provided
    downsample_y = gray_full.shape[0] / target_shape[0]
    downsample_x = gray_full.shape[1] / target_shape[1]
    if pixel_size_um is None:
        pixel_size_um = maldi_um_per_pixel / ((downsample_y + downsample_x) / 2.0)
        print(f"[3/7] H&E pixel size inferred at {pixel_size_um:.3f} μm "
              f"(area-matched against MALDI)")
    else:
        print(f"[3/7] H&E pixel size given: {pixel_size_um:.3f} μm")

    print(f"[3/7] Resizing H&E {gray_full.shape} → {target_shape} "
          f"(downsample {downsample_y:.1f}×{downsample_x:.1f})")
    gray_reg = resize(gray_full, target_shape, anti_aliasing=True,
                      preserve_range=True).astype(np.float32)
    if gray_reg.max() > gray_reg.min():
        gray_reg = (gray_reg - gray_reg.min()) / (gray_reg.max() - gray_reg.min())

    print("[3/7] Computing H&E structural fields")
    structural_fields = compute_structural_fields(gray_reg)
    if registration_field not in structural_fields or registration_field == "orientation":
        raise ValueError("registration_field must be 'edge', 'anisotropy', or 'vesselness'")

    return {
        "full_resolution":    gray_full,
        "registration_grid":  gray_reg,
        "structural_fields":  structural_fields,
        "registration_image": structural_fields[registration_field].astype(np.float32),
        "downsample_factor":  (float(downsample_y), float(downsample_x)),
        "pixel_size_um":      float(pixel_size_um),
        "full_resolution_shape": tuple(gray_full.shape),
    }


def build_tissue_mask(fields: dict,
                      threshold_percentile: float = TISSUE_MASK_THRESHOLD_PERCENTILE,
                      min_size: int = TISSUE_MASK_MIN_SIZE) -> np.ndarray:
    """
    Build a tissue-support mask from structural evidence, not raw intensity.

    This mask limits structural-difference and tissue-overlap scoring to the
    tissue support, so background zeros do not reward bad transforms.
    """
    support = (
        0.45 * normalize01(fields["edge"]) +
        0.35 * normalize01(fields["anisotropy"]) +
        0.20 * normalize01(fields["vesselness"])
    )
    positive = support[support > 0]
    if positive.size == 0:
        return np.ones_like(support, dtype=bool)

    threshold = np.percentile(positive, threshold_percentile)
    mask = support >= threshold
    mask = ndi.binary_fill_holes(mask)
    mask = ndi.binary_closing(mask, structure=np.ones((3, 3), dtype=bool))
    mask = remove_small_objects(mask.astype(bool), min_size=min_size)
    if not np.any(mask):
        return np.ones_like(support, dtype=bool)
    return mask.astype(bool)


def compute_correspondence_table(maldi: dict,
                                  he: dict,
                                  transform: sitk.Transform,
                                  imzml_path: str) -> tuple:
    """
    Build a MALDI→H&E pixel correspondence table — one row per spectrum.

    Returns (table, columns) where:
      table   : [N, 4] float64 array
      columns : ["maldi_x", "maldi_y", "he_x", "he_y"]

    Columns:
      0  maldi_x        column in the MALDI native grid (0..W_m - 1)
      1  maldi_y        row in the MALDI native grid (0..H_m - 1)
      2  he_x           x in full-resolution H&E pixel space (fractional)
      3  he_y           y in full-resolution H&E pixel space (fractional)

    The MALDI→H&E direction uses the INVERSE of the registration transform
    (since the registration mapped fixed=H&E → moving=MALDI). For pure
    affine transforms this is computed analytically and vectorised; for
    composite transforms (B-spline) it falls back to per-point evaluation.
    """
    
    parser = ImzMLParser(imzml_path)
    coords = list(parser.coordinates)
    n = len(coords)

    x0, y0 = maldi["grid_origin"]
    dy_he, dx_he = he["downsample_factor"]

    imzml_x = np.array([c[0] for c in coords], dtype=np.float64)
    imzml_y = np.array([c[1] for c in coords], dtype=np.float64)
    grid_x  = imzml_x - x0
    grid_y  = imzml_y - y0

    # MALDI → H&E registration-grid coords via inverse transform.
    # Linear 2D transforms expose matrix/center/translation, so use the
    # analytic inverse and keep the table export fast.
    if all(hasattr(transform, name) for name in
           ("GetMatrix", "GetTranslation", "GetCenter")):
        M = np.array(transform.GetMatrix()).reshape(2, 2)
        t = np.array(transform.GetTranslation())
        c = np.array(transform.GetCenter())
        Minv = np.linalg.inv(M)
        pts = np.stack([grid_x, grid_y], axis=1)
        he_reg = (pts - t - c) @ Minv.T + c
    else:
        # Composite/B-spline transforms may not expose an analytic inverse.
        # Use it if available; otherwise use nearest inverse lookup over the
        # H&E registration grid.
        he_reg = np.zeros((n, 2), dtype=np.float64)
        try:
            inverse = transform.GetInverse()
            for i in range(n):
                p = inverse.TransformPoint((float(grid_x[i]), float(grid_y[i])))
                he_reg[i] = p
        except RuntimeError:
            print("[7/7] Transform has no analytic inverse; using grid-based inverse lookup")
            he_reg = approximate_inverse_on_grid(
                transform, maldi_points=np.stack([grid_x, grid_y], axis=1),
                he_shape=he["registration_grid"].shape,
            )

    # Registration-grid → full H&E pixel space (multiply by downsample factor)
    he_full_x = he_reg[:, 0] * dx_he
    he_full_y = he_reg[:, 1] * dy_he

    table = np.column_stack([grid_x, grid_y, he_full_x, he_full_y])
    columns = ["maldi_x", "maldi_y", "he_x", "he_y"]
    return table, columns


def approximate_inverse_on_grid(transform: sitk.Transform,
                                maldi_points: np.ndarray,
                                he_shape: tuple) -> np.ndarray:
    """
    Approximate MALDI→H&E registration-grid coordinates for non-analytic
    inverses by mapping every H&E grid point forward and nearest-matching
    requested MALDI points.
    """
    he_h, he_w = he_shape
    yy, xx = np.mgrid[0:he_h, 0:he_w]
    he_points = np.column_stack([xx.ravel(), yy.ravel()]).astype(np.float64)
    mapped = np.zeros_like(he_points)
    for i, (x, y) in enumerate(he_points):
        mapped[i] = transform.TransformPoint((float(x), float(y)))

    nn = NearestNeighbors(n_neighbors=1, algorithm="auto")
    nn.fit(mapped)
    nearest = nn.kneighbors(maldi_points, return_distance=False).ravel()
    return he_points[nearest]


def save_correspondence_csv(table: np.ndarray, columns: list,
                             path: Path) -> None:
    """Write the lookup table to CSV with appropriate per-column formats."""
    fmt = ["%d", "%d", "%.4f", "%.4f"]
    np.savetxt(path, table, delimiter=",",
                header=",".join(columns), comments="", fmt=fmt)
    print(f"[7/7] Correspondence CSV → {path} ({len(table)} rows)")


# ──────────────────────────────────────────────────────────────────────────
# Helpers + visualization
# ──────────────────────────────────────────────────────────────────────────

def np_to_sitk(arr: np.ndarray) -> sitk.Image:
    return sitk.GetImageFromArray(arr.astype(np.float32))

def mask_to_sitk(mask: np.ndarray) -> sitk.Image:
    return sitk.GetImageFromArray(mask.astype(np.uint8))

def sitk_to_np(img: sitk.Image) -> np.ndarray:
    return sitk.GetArrayFromImage(img)

def warp_to_fixed(moving: sitk.Image, fixed: sitk.Image,
                  tx: sitk.Transform) -> np.ndarray:
    return sitk_to_np(sitk.Resample(moving, fixed, tx, sitk.sitkLinear, 0.0))

def warp_mask_to_fixed(mask: np.ndarray, fixed: sitk.Image,
                       tx: sitk.Transform) -> np.ndarray:
    moving = sitk.GetImageFromArray(mask.astype(np.uint8))
    warped = sitk.Resample(moving, fixed, tx, sitk.sitkNearestNeighbor, 0)
    return sitk_to_np(warped).astype(bool)

def euler_transform_from_params(params: np.ndarray,
                                fixed_shape: tuple) -> sitk.Euler2DTransform:
    """
    Build a fixed->moving Euler transform from [angle_deg, tx_px, ty_px].

    Scale is fixed at 1.0 and no shear is allowed. This keeps optimization in
    the biologically plausible regime requested for local tissue geometry.
    """
    angle_deg, tx, ty = [float(v) for v in params]
    h, w = fixed_shape
    transform = sitk.Euler2DTransform()
    transform.SetCenter(((w - 1) / 2.0, (h - 1) / 2.0))
    transform.SetAngle(np.deg2rad(angle_deg))
    transform.SetTranslation((tx, ty))
    return transform

def save_image(arr: np.ndarray, path: Path, cmap: str = "gray",
               title: str = None) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(arr, cmap=cmap); ax.axis("off")
    ax.set_title(title or path.stem)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)

def save_component_denoising_qc(raw_components: np.ndarray,
                                denoised_components: np.ndarray,
                                path: Path) -> None:
    """Side-by-side raw/denoised QC panel for every NMF component."""
    K = raw_components.shape[0]
    fig, axes = plt.subplots(K, 2, figsize=(8, 3.2 * K), squeeze=False)
    for k in range(K):
        axes[k, 0].imshow(raw_components[k], cmap="viridis", vmin=0, vmax=1)
        axes[k, 0].set_title(f"NMF component {k + 1}: original")
        axes[k, 0].axis("off")
        axes[k, 1].imshow(denoised_components[k], cmap="viridis", vmin=0, vmax=1)
        axes[k, 1].set_title(f"NMF component {k + 1}: denoised")
        axes[k, 1].axis("off")
    fig.suptitle("NMF Spatial Component Denoising QC", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

def save_overlay(maldi_warped: np.ndarray, he: np.ndarray,
                 path: Path, alpha: float = 0.5) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(he, cmap="gray")
    ax.imshow(maldi_warped, cmap="magma", alpha=alpha)
    ax.set_title("MALDI structural field (warped, magma) over H&E structural field")
    ax.axis("off")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)

def save_difference(a: np.ndarray, b: np.ndarray, path: Path,
                    title: str) -> None:
    residual = np.abs(normalize01(a) - normalize01(b))
    fig, ax = plt.subplots(figsize=(7.5, 6))
    im = ax.imshow(residual, cmap="magma", vmin=0, vmax=1)
    ax.set_title(title)
    ax.axis("off")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Absolute normalized structural-field difference |MALDI - H&E| (0-1)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_structural_fields(fields: dict, path: Path, title: str) -> None:
    """QC panel for the structural representation used by registration."""
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), constrained_layout=True)
    panels = [
        ("edge", "magma", "Edge magnitude"),
        ("orientation", "twilight", "Orientation"),
        ("anisotropy", "viridis", "Anisotropy"),
        ("vesselness", "magma", "Vesselness"),
    ]
    for ax, (key, cmap, label_text) in zip(axes, panels):
        arr = fields[key]
        if key != "orientation":
            arr = normalize01(arr)
        im = ax.imshow(arr, cmap=cmap)
        ax.set_title(label_text)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title, fontsize=12)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def detect_vessels(vesselness: np.ndarray,
                   min_area: int = VESSEL_MIN_AREA,
                   max_area: int = VESSEL_MAX_AREA,
                   threshold_percentile: float = VESSEL_THRESHOLD_PERCENTILE,
                   min_eccentricity: float = VESSEL_MIN_ECCENTRICITY) -> dict:
    """
    Detect vessel-like regions and return their centers and shape properties.

    The output coordinates are in registration-grid pixels: x is column,
    y is row. These are the same coordinates used by SimpleITK transforms.
    """
    vessel = normalize01(vesselness)
    if np.all(vessel == 0):
        empty = pd.DataFrame(columns=["x", "y", "area", "eccentricity", "orientation"])
        return {"mask": np.zeros_like(vessel, dtype=bool),
                "labels": np.zeros_like(vessel, dtype=np.int32),
                "table": empty}

    percentile_thr = np.percentile(vessel[vessel > 0], threshold_percentile) if np.any(vessel > 0) else 1.0
    try:
        otsu_thr = threshold_otsu(vessel)
        threshold = max(float(percentile_thr), float(otsu_thr))
    except ValueError:
        threshold = float(percentile_thr)

    mask = vessel >= threshold
    #mask = ndi.binary_fill_holes(mask) # Don't fill holes; vessels can have gaps and we want to keep them separate.
    labels_img = label(mask)
    props = regionprops_table(
        labels_img,
        properties=("label", "area", "centroid", "eccentricity", "orientation"),
    )
    table = pd.DataFrame(props)
    if table.empty:
        table = pd.DataFrame(columns=["x", "y", "area", "eccentricity", "orientation"])
        return {"mask": mask,
                "labels": labels_img.astype(np.int32),
                "table": table}

    table = table.rename(columns={"centroid-0": "y", "centroid-1": "x"})
    table = table[
        (table["area"] >= min_area) &
        (table["area"] <= max_area) &
        (table["eccentricity"] >= min_eccentricity)
    ].copy()
    keep_labels = set(table["label"].astype(int))
    labels_img = np.where(np.isin(labels_img, list(keep_labels)), labels_img, 0)
    mask = labels_img > 0
    table = table[["x", "y", "area", "eccentricity", "orientation"]].reset_index(drop=True)
    return {"mask": mask,
            "labels": labels_img.astype(np.int32),
            "table": table}


def transform_moving_points_to_fixed(points_xy: np.ndarray,
                                     transform: sitk.Transform,
                                     fixed_shape: tuple = None) -> np.ndarray:
    """
    Map MALDI/moving points into the H&E/fixed registration grid.

    The SimpleITK registration transform maps fixed → moving for resampling,
    so vessel-center validation uses its inverse.
    """
    if len(points_xy) == 0:
        return np.zeros((0, 2), dtype=np.float64)
    try:
        inverse = transform.GetInverse()
        transformed = np.zeros((len(points_xy), 2), dtype=np.float64)
        for i, (x, y) in enumerate(points_xy):
            transformed[i] = inverse.TransformPoint((float(x), float(y)))
        return transformed
    except RuntimeError:
        if fixed_shape is None:
            raise
        return approximate_inverse_on_grid(transform, points_xy, fixed_shape)


def vessel_displacement_stats(maldi_vessels: pd.DataFrame,
                              he_vessels: pd.DataFrame,
                              transform: sitk.Transform,
                              fixed_shape: tuple) -> tuple[pd.DataFrame, dict]:
    """Nearest-neighbor vessel-center displacement after registration."""
    if maldi_vessels.empty or he_vessels.empty:
        empty = pd.DataFrame(columns=[
            "maldi_x", "maldi_y", "maldi_registered_x", "maldi_registered_y",
            "he_x", "he_y", "distance",
        ])
        return empty, {
            "mean": np.nan,
            "std": np.nan,
            "median": np.nan,
            "p95": np.nan,
            "n": 0,
        }

    maldi_xy = maldi_vessels[["x", "y"]].to_numpy(dtype=np.float64)
    he_xy = he_vessels[["x", "y"]].to_numpy(dtype=np.float64)
    maldi_registered = transform_moving_points_to_fixed(maldi_xy, transform, fixed_shape)

    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(he_xy)
    distances, indices = nn.kneighbors(maldi_registered)
    distances = distances.ravel()
    matched_he = he_xy[indices.ravel()]

    table = pd.DataFrame({
        "maldi_x": maldi_xy[:, 0],
        "maldi_y": maldi_xy[:, 1],
        "maldi_registered_x": maldi_registered[:, 0],
        "maldi_registered_y": maldi_registered[:, 1],
        "he_x": matched_he[:, 0],
        "he_y": matched_he[:, 1],
        "distance": distances,
    })
    stats = {
        "mean": float(np.mean(distances)),
        "std": float(np.std(distances)),
        "median": float(np.median(distances)),
        "p95": float(np.percentile(distances, 95)),
        "n": int(len(distances)),
    }
    return table, stats


def bidirectional_vessel_chamfer(maldi_vessels: pd.DataFrame,
                                 he_vessels: pd.DataFrame,
                                 transform: sitk.Transform,
                                 fixed_shape: tuple,
                                 vessel_cache: dict = None) -> dict:
    """
    Symmetric vessel-center distance under a transform.

    MALDI centers are mapped moving->fixed via inverse transform and matched to
    H&E centers. H&E centers are mapped fixed->moving and matched back to MALDI
    centers. This avoids the one-way nearest-neighbor trap where dense H&E
    detections can make a poor alignment look acceptable.
    """
    if maldi_vessels.empty or he_vessels.empty:
        return {
            "vessel_chamfer": np.inf,
            "maldi_to_he_mean": np.inf,
            "he_to_maldi_mean": np.inf,
            "vessel_n_maldi": int(len(maldi_vessels)),
            "vessel_n_he": int(len(he_vessels)),
        }

    if vessel_cache is None:
        maldi_xy = maldi_vessels[["x", "y"]].to_numpy(dtype=np.float64)
        he_xy = he_vessels[["x", "y"]].to_numpy(dtype=np.float64)
        he_nn = NearestNeighbors(n_neighbors=1).fit(he_xy)
        maldi_nn = NearestNeighbors(n_neighbors=1).fit(maldi_xy)
    else:
        maldi_xy = vessel_cache["maldi_xy"]
        he_xy = vessel_cache["he_xy"]
        he_nn = vessel_cache["he_nn"]
        maldi_nn = vessel_cache["maldi_nn"]

    maldi_registered = transform_moving_points_to_fixed(maldi_xy, transform, fixed_shape)
    he_to_maldi = np.zeros_like(he_xy)
    for i, (x, y) in enumerate(he_xy):
        he_to_maldi[i] = transform.TransformPoint((float(x), float(y)))

    maldi_to_he_dist = he_nn.kneighbors(maldi_registered, return_distance=True)[0].ravel()
    he_to_maldi_dist = maldi_nn.kneighbors(he_to_maldi, return_distance=True)[0].ravel()

    maldi_to_he_mean = float(np.mean(maldi_to_he_dist))
    he_to_maldi_mean = float(np.mean(he_to_maldi_dist))
    return {
        "vessel_chamfer": 0.5 * (maldi_to_he_mean + he_to_maldi_mean),
        "maldi_to_he_mean": maldi_to_he_mean,
        "he_to_maldi_mean": he_to_maldi_mean,
        "vessel_n_maldi": int(len(maldi_xy)),
        "vessel_n_he": int(len(he_xy)),
    }


def biological_objective_components(params: np.ndarray,
                                    fixed_image: np.ndarray,
                                    moving_image: np.ndarray,
                                    fixed_mask: np.ndarray,
                                    moving_mask: np.ndarray,
                                    maldi_vessels: pd.DataFrame,
                                    he_vessels: pd.DataFrame,
                                    fixed_sitk: sitk.Image,
                                    moving_sitk: sitk.Image,
                                    vessel_cache: dict = None) -> tuple[float, dict, sitk.Transform]:
    """
    Direct structural/biological registration objective.

    Objective terms:
      vessel_chamfer      bidirectional vessel displacement in pixels
      structural_diff     mean |warped MALDI - H&E| within tissue overlap
      lost_overlap        1 - intersection/union of structural tissue masks
      transform_size      normalized rotation/translation magnitude
    """
    transform = euler_transform_from_params(params, fixed_image.shape)
    warped_moving = warp_to_fixed(moving_sitk, fixed_sitk, transform)
    warped_mask = warp_mask_to_fixed(moving_mask, fixed_sitk, transform)

    overlap = fixed_mask & warped_mask
    union = fixed_mask | warped_mask
    if np.any(overlap):
        structural_diff = float(np.mean(np.abs(
            normalize01(warped_moving)[overlap] - normalize01(fixed_image)[overlap]
        )))
    else:
        structural_diff = 1.0

    if np.any(union):
        tissue_overlap = float(np.sum(overlap) / np.sum(union))
    else:
        tissue_overlap = 0.0
    lost_overlap = 1.0 - tissue_overlap

    vessel_stats = bidirectional_vessel_chamfer(
        maldi_vessels, he_vessels, transform, fixed_image.shape, vessel_cache)
    vessel_term = vessel_stats["vessel_chamfer"] / VESSEL_DISTANCE_SCALE_PIXELS

    angle_deg, tx, ty = [float(v) for v in params]
    transform_size = np.sqrt(
        (angle_deg / MAX_ROTATION_DEGREES) ** 2 +
        (tx / MAX_TRANSLATION_PIXELS) ** 2 +
        (ty / MAX_TRANSLATION_PIXELS) ** 2
    )

    score = (
        BIOLOGY_WEIGHT_VESSEL * vessel_term +
        BIOLOGY_WEIGHT_STRUCTURAL * structural_diff +
        BIOLOGY_WEIGHT_TISSUE_OVERLAP * lost_overlap +
        BIOLOGY_WEIGHT_TRANSFORM_SIZE * transform_size
    )
    components = {
        "score": float(score),
        "angle_deg": angle_deg,
        "tx": tx,
        "ty": ty,
        "vessel_chamfer": float(vessel_stats["vessel_chamfer"]),
        "maldi_to_he_mean": float(vessel_stats["maldi_to_he_mean"]),
        "he_to_maldi_mean": float(vessel_stats["he_to_maldi_mean"]),
        "structural_diff": structural_diff,
        "tissue_overlap": tissue_overlap,
        "lost_overlap": lost_overlap,
        "transform_size": float(transform_size),
        "vessel_n_maldi": vessel_stats["vessel_n_maldi"],
        "vessel_n_he": vessel_stats["vessel_n_he"],
    }
    return float(score), components, transform


def optimize_biological_transform(fixed_image: np.ndarray,
                                  moving_image: np.ndarray,
                                  fixed_mask: np.ndarray,
                                  moving_mask: np.ndarray,
                                  maldi_vessels: pd.DataFrame,
                                  he_vessels: pd.DataFrame,
                                  fixed_sitk: sitk.Image,
                                  moving_sitk: sitk.Image) -> tuple[sitk.Transform, pd.DataFrame, dict]:
    """
    Optimize the direct biological objective over a bounded rigid transform.

    Search space:
      angle_deg in [-MAX_ROTATION_DEGREES, MAX_ROTATION_DEGREES]
      tx, ty    in [-MAX_TRANSLATION_PIXELS, MAX_TRANSLATION_PIXELS]
    """
    bounds = [
        (-MAX_ROTATION_DEGREES, MAX_ROTATION_DEGREES),
        (-MAX_TRANSLATION_PIXELS, MAX_TRANSLATION_PIXELS),
        (-MAX_TRANSLATION_PIXELS, MAX_TRANSLATION_PIXELS),
    ]
    trace = []
    vessel_cache = None
    if not maldi_vessels.empty and not he_vessels.empty:
        maldi_xy = maldi_vessels[["x", "y"]].to_numpy(dtype=np.float64)
        he_xy = he_vessels[["x", "y"]].to_numpy(dtype=np.float64)
        vessel_cache = {
            "maldi_xy": maldi_xy,
            "he_xy": he_xy,
            "maldi_nn": NearestNeighbors(n_neighbors=1).fit(maldi_xy),
            "he_nn": NearestNeighbors(n_neighbors=1).fit(he_xy),
        }

    def evaluate(params, stage):
        score, components, _ = biological_objective_components(
            np.asarray(params, dtype=float),
            fixed_image,
            moving_image,
            fixed_mask,
            moving_mask,
            maldi_vessels,
            he_vessels,
            fixed_sitk,
            moving_sitk,
            vessel_cache,
        )
        components["stage"] = stage
        trace.append(components)
        return score

    identity_params = np.array([0.0, 0.0, 0.0], dtype=float)
    identity_score = evaluate(identity_params, "identity")
    print(f"[4/7] Identity biological objective = {identity_score:.4f}")

    print("[4/7] Optimizing direct biological objective: vessel + structure + overlap")
    result = optimize.differential_evolution(
        lambda p: evaluate(p, "global"),
        bounds=bounds,
        maxiter=BIOLOGY_OPT_MAXITER,
        popsize=BIOLOGY_OPT_POPSIZE,
        tol=0.01,
        polish=False,
        seed=42,
        workers=1,
        updating="immediate",
    )
    local = optimize.minimize(
        lambda p: evaluate(p, "local"),
        x0=result.x,
        method="Powell",
        bounds=bounds,
        options={"maxiter": 80, "xtol": 1e-3, "ftol": 1e-4},
    )

    candidates = [
        ("identity", identity_params, identity_score),
        ("global", result.x, float(result.fun)),
        ("local", local.x, float(local.fun)),
    ]
    best_name, best_params, _ = min(candidates, key=lambda item: item[2])
    best_score, best_components, best_transform = biological_objective_components(
        best_params,
        fixed_image,
        moving_image,
        fixed_mask,
        moving_mask,
        maldi_vessels,
        he_vessels,
        fixed_sitk,
        moving_sitk,
        vessel_cache,
    )
    best_components["selected_stage"] = best_name
    best_components["score"] = best_score
    print(
        f"[4/7] Selected biological optimum from {best_name}: "
        f"score={best_score:.4f}, vessel={best_components['vessel_chamfer']:.3f}px, "
        f"struct={best_components['structural_diff']:.3f}, "
        f"overlap={best_components['tissue_overlap']:.3f}, "
        f"angle={best_components['angle_deg']:.3f}, "
        f"tx={best_components['tx']:.3f}, ty={best_components['ty']:.3f}"
    )
    return best_transform, pd.DataFrame(trace), best_components


def save_vessel_overlay(base: np.ndarray,
                        he_vessels: dict,
                        maldi_vessels: dict,
                        path: Path,
                        title: str,
                        maldi_points_xy: np.ndarray = None) -> None:
    """Overlay H&E and MALDI vessel masks/centers on a registration-grid image."""
    he_df = he_vessels["table"]
    maldi_df = maldi_vessels["table"]
    if maldi_points_xy is None and not maldi_df.empty:
        maldi_points_xy = maldi_df[["x", "y"]].to_numpy(dtype=np.float64)
    elif maldi_points_xy is None:
        maldi_points_xy = np.zeros((0, 2), dtype=np.float64)

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(normalize01(base), cmap="gray")
    if np.any(he_vessels["mask"]):
        ax.contour(he_vessels["mask"], colors="cyan", linewidths=0.8)
    if np.any(maldi_vessels["mask"]):
        ax.contour(maldi_vessels["mask"], colors="magenta", linewidths=0.8)
    if not he_df.empty:
        ax.scatter(he_df["x"], he_df["y"], s=24, c="cyan", marker="o", label="H&E vessels")
    if len(maldi_points_xy) > 0:
        ax.scatter(maldi_points_xy[:, 0], maldi_points_xy[:, 1],
                   s=24, c="magenta", marker="+", label="MALDI vessels")
    ax.set_title(title)
    ax.axis("off")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_vessel_displacement_qc(displacements: pd.DataFrame,
                                base: np.ndarray,
                                path: Path) -> None:
    """Histogram and vector plot for local anatomical vessel-center alignment."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    if displacements.empty:
        axes[0].text(0.5, 0.5, "No matched vessels", ha="center", va="center")
        axes[1].text(0.5, 0.5, "No displacement vectors", ha="center", va="center")
        for ax in axes:
            ax.axis("off")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return

    axes[0].hist(displacements["distance"], bins=30, color="steelblue", edgecolor="white")
    axes[0].set_title("Nearest H&E vessel displacement")
    axes[0].set_xlabel("Distance (registration-grid pixels)")
    axes[0].set_ylabel("MALDI vessel count")

    axes[1].imshow(normalize01(base), cmap="gray")
    dx = displacements["he_x"] - displacements["maldi_registered_x"]
    dy = displacements["he_y"] - displacements["maldi_registered_y"]
    axes[1].quiver(
        displacements["maldi_registered_x"],
        displacements["maldi_registered_y"],
        dx,
        dy,
        angles="xy",
        scale_units="xy",
        scale=1,
        color="yellow",
        width=0.003,
    )
    axes[1].scatter(displacements["he_x"], displacements["he_y"], s=18, c="cyan")
    axes[1].set_title("MALDI vessel → nearest H&E vessel")
    axes[1].axis("off")
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--imzml",        required=True, help="Path to .imzML")
    ap.add_argument("--he",           required=True, help="Path to H&E image")
    ap.add_argument("--output_dir",   default="reg_results")
    ap.add_argument("--n_components", type=int, default=8,
                    help="Number of NMF components")
    ap.add_argument("--mz_bin",       type=float, default=0.042,
                    help="m/z tolerance in Da for dense-region binning")
    ap.add_argument("--mz_bin_min_count", type=int, default=100,
                    help="Minimum number of m/z values required to keep a dense bin")
    ap.add_argument("--maldi_um_per_pixel", type=float, default=5.0,
                    help="Physical size of one MALDI pixel (μm)")
    ap.add_argument("--he_um_per_pixel", type=float, default=None,
                    help="Physical size of one H&E pixel (μm). "
                         "If omitted, inferred assuming same tissue area.")
    ap.add_argument("--bspline",      action="store_true",
                    help="Ignored; biological optimization is rigid rotation + translation only")
    ap.add_argument("--no_hematoxylin", action="store_true",
                    help="Use plain grayscale H&E instead of hematoxylin")
    ap.add_argument("--denoise_method", default=DENOISE_METHOD,
                    choices=["gaussian", "median", "nl_means", "none"],
                    help="Denoising method applied independently to each NMF component")
    ap.add_argument("--denoise_sigma", type=float, default=GAUSSIAN_SIGMA_PIXELS,
                    help="Gaussian sigma in MALDI pixels when --denoise_method=gaussian")
    ap.add_argument("--median_radius", type=int, default=MEDIAN_RADIUS_PIXELS,
                    help="Median filter radius in pixels when --denoise_method=median")
    ap.add_argument("--structural_combine", default=STRUCTURAL_COMBINE_METHOD,
                    choices=["max", "weighted_mean"],
                    help="How to combine per-component structural fields")
    ap.add_argument("--registration_field", default=REGISTRATION_FIELD,
                    choices=["edge", "anisotropy", "vesselness"],
                    help="Structural field used in the direct biological objective")
    ap.add_argument("--transform_model", default="euler",
                    choices=["euler", "similarity", "affine"],
                    help="Ignored; direct biological optimization uses a constrained Euler transform")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1+2. MALDI data
    maldi = build_maldi_data(args.imzml,
                              n_components=args.n_components,
                              mz_bin_width=args.mz_bin,
                              mz_bin_min_count=args.mz_bin_min_count,
                              pixel_size_um=args.maldi_um_per_pixel,
                              denoise_method=args.denoise_method,
                              gaussian_sigma=args.denoise_sigma,
                              median_radius=args.median_radius,
                              structural_combine=args.structural_combine,
                              registration_field=args.registration_field)
    save_component_denoising_qc(
        maldi["components"],
        maldi["denoised_components"],
        out / "nmf_component_denoising.png",
    )
    save_structural_fields(
        maldi["structural_fields"],
        out / "maldi_structural_fields.png",
        "MALDI structural fields from NMF components",
    )

    # 3. H&E
    he = prepare_he(args.he, maldi["grid_shape"],
                     use_hematoxylin=not args.no_hematoxylin,
                     pixel_size_um=args.he_um_per_pixel,
                     maldi_um_per_pixel=args.maldi_um_per_pixel,
                     registration_field=args.registration_field)
    save_image(he["registration_grid"], out / "he_at_registration_grid.png",
               cmap="gray", title="H&E at MALDI grid")
    save_structural_fields(
        he["structural_fields"],
        out / "he_structural_fields.png",
        "H&E hematoxylin structural fields",
    )

    # Vessel detection and pre-registration biological QC.
    maldi_vessels = detect_vessels(maldi["structural_fields"]["vesselness"])
    he_vessels = detect_vessels(he["structural_fields"]["vesselness"])
    maldi_vessels["table"].to_csv(out / "maldi_vessels.csv", index=False)
    he_vessels["table"].to_csv(out / "he_vessels.csv", index=False)
    save_vessel_overlay(
        he["structural_fields"]["edge"],
        he_vessels,
        maldi_vessels,
        out / "vessel_overlay_pre.png",
        "Pre-registration vessel centers and contours",
    )
    save_difference(maldi["registration_image"], he["registration_image"],
                     out / "structural_difference_pre.png",
                     title="Pre-registration structural-field difference")
    # ------------------------------------------------------------------
    # PRE-REGISTRATION COORDINATE EXPORT
    # Use identity transform but preserve EXACT registration geometry
    # ------------------------------------------------------------------

    print("[4/7] Exporting pre-registration coordinate map")

    identity_tx = sitk.Euler2DTransform()
    identity_tx.SetCenter((
        (maldi["grid_shape"][1] - 1) / 2.0,
        (maldi["grid_shape"][0] - 1) / 2.0
    ))
    identity_tx.SetAngle(0.0)
    identity_tx.SetTranslation((0.0, 0.0))

    corr_table, corr_cols = compute_correspondence_table(
        maldi,
        he,
        identity_tx,
        args.imzml,
    )

    save_correspondence_csv(
        corr_table,
        corr_cols,
        out / "pre_registration_coordinate_map.csv"
    )

    # 4. Direct biological optimization. No MI, no shear, no scale changes:
    # optimize only rotation + translation for local tissue geometry.
    fixed  = np_to_sitk(he["registration_image"])
    moving = np_to_sitk(maldi["registration_image"])
    he_tissue_mask = build_tissue_mask(he["structural_fields"])
    maldi_tissue_mask = build_tissue_mask(maldi["structural_fields"])
    save_image(maldi_tissue_mask.astype(np.float32), out / "maldi_tissue_mask.png",
               cmap="gray", title="MALDI structural tissue mask")
    save_image(he_tissue_mask.astype(np.float32), out / "he_tissue_mask.png",
               cmap="gray", title="H&E structural tissue mask")
    if args.bspline:
        print("[4/7] --bspline ignored: direct biological objective is constrained to rigid rotation + translation")

    final_tx, objective_trace, objective_stats = optimize_biological_transform(
        fixed_image=he["registration_image"],
        moving_image=maldi["registration_image"],
        fixed_mask=he_tissue_mask,
        moving_mask=maldi_tissue_mask,
        maldi_vessels=maldi_vessels["table"],
        he_vessels=he_vessels["table"],
        fixed_sitk=fixed,
        moving_sitk=moving,
    )
    objective_trace.to_csv(out / "biological_objective_trace.csv", index=False)
    pd.DataFrame([objective_stats]).to_csv(out / "biological_objective_summary.csv", index=False)
    with open(out / "selected_transform.txt", "w", encoding="utf-8") as f:
        f.write("selected_candidate: direct_biological_objective\n")
        f.write("reason: optimized vessel Chamfer + structural difference + tissue overlap directly\n")
        for key, value in objective_stats.items():
            f.write(f"{key}: {value}\n")

    # 6. Diagnostic visualizations
    warped_structural = warp_to_fixed(moving, fixed, final_tx)
    save_overlay(warped_structural, he["registration_image"],
                  out / "registered_overlay.png")
    save_difference(warped_structural, he["registration_image"],
                     out / "structural_difference_post.png",
                     title="Post-registration structural-field difference")

    displacement_df, displacement_stats = vessel_displacement_stats(
        maldi_vessels["table"],
        he_vessels["table"],
        final_tx,
        he["registration_image"].shape,
    )
    displacement_stats.update({
        "bidirectional_vessel_chamfer": objective_stats["vessel_chamfer"],
        "maldi_to_he_mean": objective_stats["maldi_to_he_mean"],
        "he_to_maldi_mean": objective_stats["he_to_maldi_mean"],
        "structural_diff": objective_stats["structural_diff"],
        "tissue_overlap": objective_stats["tissue_overlap"],
        "lost_overlap": objective_stats["lost_overlap"],
        "objective_score": objective_stats["score"],
        "angle_deg": objective_stats["angle_deg"],
        "tx": objective_stats["tx"],
        "ty": objective_stats["ty"],
    })
    displacement_df.to_csv(out / "vessel_displacements.csv", index=False)
    post_points = displacement_df[["maldi_registered_x", "maldi_registered_y"]].to_numpy(
        dtype=np.float64) if not displacement_df.empty else np.zeros((0, 2))
    warped_maldi_vessels = {
        "mask": warp_mask_to_fixed(maldi_vessels["mask"], fixed, final_tx),
        "labels": np.zeros_like(he_vessels["labels"]),
        "table": maldi_vessels["table"],
    }
    save_vessel_overlay(
        he["structural_fields"]["edge"],
        he_vessels,
        warped_maldi_vessels,
        out / "vessel_overlay_post.png",
        "Post-registration vessel centers and contours",
        maldi_points_xy=post_points,
    )
    save_vessel_displacement_qc(
        displacement_df,
        he["structural_fields"]["edge"],
        out / "vessel_displacement_qc.png",
    )
    pd.DataFrame([displacement_stats]).to_csv(out / "vessel_displacement_stats.csv", index=False)
    print(
        "[6/7] Selected vessel displacement: "
        f"n={displacement_stats['n']}, "
        f"mean={displacement_stats['mean']:.3f}, "
        f"std={displacement_stats['std']:.3f}, "
        f"median={displacement_stats['median']:.3f}, "
        f"p95={displacement_stats['p95']:.3f} registration-grid pixels"
    )

    # 7. Coordinate mapping export
    corr_table, corr_cols = compute_correspondence_table(
        maldi, he, final_tx, args.imzml)
    save_correspondence_csv(corr_table, corr_cols,
                             out / "maldi_to_he_table.csv")

    print(f"\n✓ All outputs in {out.resolve()}\n")
    print("Next steps:")
    print("  1. Inspect nmf_component_denoising.png for component-level smoothing effects")
    print("  2. Inspect maldi_structural_fields.png and he_structural_fields.png")
    print("  3. Inspect biological_objective_summary.csv and biological_objective_trace.csv")
    print("  4. Use vessel_displacement_stats.csv as local anatomical validation")
    print("  5. Use maldi_to_he_table.csv for MALDI-to-H&E coordinate mapping")
    print()


if __name__ == "__main__":
    main()

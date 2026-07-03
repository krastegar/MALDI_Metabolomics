"""
sinusoid_brightness_mask.py
===========================
Two-step pipeline:
  1. Find tissue regions (everything that is not white slide background).
  2. Select white pixels that fall inside those tissue regions → sinusoid mask.

No hyperparameters.  Otsu's method determines the tissue / background boundary
automatically.  Holes in the tissue mask are filled with binary_fill_holes so
that sinusoid lumens are enclosed within the tissue footprint, then white pixels
inside that footprint are the final mask.

Outputs saved to OUT_DIR:
  sinusoid_mask.png      – binary mask  (0 = background / tissue, 255 = sinusoid)
  sinusoid_mask.npy      – same, as a boolean NumPy array
  sinusoid_overview.png  – three-panel QC figure
"""

import numpy as np
import cv2
from pathlib import Path
from scipy import ndimage
import matplotlib.pyplot as plt

# ── paths ────────────────────────────────────────────────────────────────────
IMAGE_PATH = "/home/kia/MALDI_Metabolomics/metabol_data/high_res_MSI/D2_10x_originalExport.tif"
OUT_DIR    = "/home/kia/MALDI_Metabolomics/annotations_training"
# ─────────────────────────────────────────────────────────────────────────────


def build_tissue_mask(gray: np.ndarray) -> tuple[np.ndarray, int]:
    """
    Returns (tissue_mask, otsu_threshold).

    tissue_mask is boolean, True where tissue is present (holes filled so that
    sinusoid lumens are enclosed inside the mask).
    """
    # Otsu automatically finds the threshold between stained tissue and white background
    otsu_thresh, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    tissue_raw = gray < otsu_thresh  # True = stained, False = white/background

    # Fill enclosed holes (sinusoid lumens) so the mask is a solid tissue footprint
    tissue_filled = ndimage.binary_fill_holes(tissue_raw)

    # Drop fragments smaller than 0.1 % of the image (dust, mounting medium specks)
    labeled, n = ndimage.label(tissue_filled)
    min_area = tissue_filled.size * 0.001
    tissue_clean = np.zeros_like(tissue_filled)
    for i, area in enumerate(ndimage.sum(tissue_filled, labeled, range(1, n + 1)), start=1):
        if area >= min_area:
            tissue_clean[labeled == i] = True

    return tissue_clean, int(otsu_thresh)


def build_sinusoid_mask(gray: np.ndarray,
                        tissue_mask: np.ndarray,
                        otsu_thresh: int) -> np.ndarray:
    """
    Returns binary uint8 mask: 1 = sinusoid (white pixel inside tissue).
    """
    white_pixels = gray >= otsu_thresh
    return (white_pixels & tissue_mask).astype(np.uint8)


def save_outputs(img_rgb, tissue_mask, sinusoid_mask, out_dir, scale=0.25):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Binary mask files
    cv2.imwrite(str(out / "sinusoid_mask.png"), sinusoid_mask * 255)
    np.save(str(out / "sinusoid_mask.npy"), sinusoid_mask.astype(bool))

    # QC figure at reduced resolution
    H, W = img_rgb.shape[:2]
    dW, dH = max(1, int(W * scale)), max(1, int(H * scale))
    thumb   = cv2.resize(img_rgb,      (dW, dH), interpolation=cv2.INTER_AREA)
    t_thumb = cv2.resize(tissue_mask.astype(np.uint8) * 255,
                          (dW, dH), interpolation=cv2.INTER_NEAREST)
    s_thumb = cv2.resize(sinusoid_mask * 255,
                          (dW, dH), interpolation=cv2.INTER_NEAREST)

    overlay = thumb.copy()
    overlay[s_thumb > 0] = [0, 210, 110]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(thumb);                          axes[0].set_title("Original H&E")
    axes[1].imshow(t_thumb, cmap="gray");           axes[1].set_title("Tissue mask")
    axes[2].imshow(overlay);                        axes[2].set_title("Sinusoid mask (green)")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(str(out / "sinusoid_overview.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    n_regions = int(ndimage.label(sinusoid_mask)[1])
    print(f"  sinusoid pixels : {sinusoid_mask.sum():,}")
    print(f"  sinusoid regions: {n_regions}")
    print(f"  saved to        : {out}")


def main():
    print(f"Loading: {IMAGE_PATH}")
    bgr  = cv2.imread(IMAGE_PATH)
    img_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    gray    = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    H, W    = gray.shape
    print(f"  {W} × {H} px")

    print("Step 1: building tissue mask…")
    tissue_mask, otsu_thresh = build_tissue_mask(gray)
    print(f"  Otsu threshold  : {otsu_thresh}")
    print(f"  tissue pixels   : {tissue_mask.sum():,}  ({100*tissue_mask.mean():.1f}%)")

    print("Step 2: selecting white pixels inside tissue…")
    sinusoid_mask = build_sinusoid_mask(gray, tissue_mask, otsu_thresh)

    print("Saving outputs…")
    save_outputs(img_rgb, tissue_mask, sinusoid_mask, OUT_DIR)
    print("Done.")


if __name__ == "__main__":
    main()

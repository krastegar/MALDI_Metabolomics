#!/usr/bin/env python3
"""
MALDI MSI -> H&E registration + PSF-based pixel->cell unmixing
with StarDist nuclei segmentation for cell labels.
"""
import argparse
import os
import numpy as np
import cv2
import SimpleITK as sitk
import matplotlib.pyplot as plt

from pyimzml.ImzMLParser import ImzMLParser
from skimage import filters, morphology, measure
from skimage.transform import resize
from scipy.ndimage import binary_fill_holes, gaussian_filter, distance_transform_edt, morphological_gradient
from scipy.sparse import lil_matrix
from scipy.optimize import lsq_linear
from scipy.spatial import cKDTree
from skimage.segmentation import watershed
from stardist.models import StarDist2D

try:
    import tifffile
    HAS_TIFF = True
except Exception:
    HAS_TIFF = False

# -------------------------
# 1. imzML -> ion image
# -------------------------
def extract_ion_image(imzml_path: str, target_mz: float, tol: float = 0.3) -> np.ndarray:
    parser = ImzMLParser(imzml_path)
    coords = parser.coordinates
    width  = max(c[0] for c in coords)
    height = max(c[1] for c in coords)
    ion_img = np.zeros((height, width), dtype=np.float32)
    for idx, (x, y, _) in enumerate(coords):
        mzs, intensities = parser.getspectrum(idx)
        mzs = np.asarray(mzs, dtype=np.float64)
        intensities = np.asarray(intensities, dtype=np.float32)
        hit = np.abs(mzs - target_mz) <= tol
        ion_img[y - 1, x - 1] = float(intensities[hit].sum()) if hit.any() else 0.0
    vmax = ion_img.max()
    if vmax > 0:
        ion_img /= vmax
    return ion_img

# -------------------------
# 2. Load H&E
# -------------------------
def load_he(he_path: str):
    path = str(he_path)
    if HAS_TIFF and path.lower().endswith((".tif", ".tiff")):
        rgb = tifffile.imread(path)
        if rgb.dtype != np.uint8:
            rgb = (rgb / rgb.max() * 255).astype(np.uint8)
        if rgb.ndim == 2:
            rgb = np.stack([rgb] * 3, axis=-1)
        if rgb.shape[-1] == 4:
            rgb = rgb[..., :3]
    else:
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"Cannot read H&E image: {path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    return rgb, grey

# -------------------------
# 3. Tissue masks
# -------------------------
def tissue_mask_he(grey: np.ndarray) -> np.ndarray:
    thresh = filters.threshold_otsu(grey)
    mask = binary_fill_holes(grey < thresh)
    return morphology.remove_small_objects(mask, min_size=500).astype(np.uint8)

def tissue_mask_msi(ion_img: np.ndarray, percentile: float = 10.0) -> np.ndarray:
    nonzero = ion_img[ion_img > 0]
    thresh = np.percentile(nonzero, percentile) if len(nonzero) else 0.0
    mask = ion_img > thresh
    return morphology.remove_small_objects(mask, min_size=4).astype(np.uint8)

# -------------------------
# 4. Coarse alignment
# -------------------------
def bounding_box(mask: np.ndarray):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        h, w = mask.shape
        return 0, 0, w, h
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    return x, y, w, h

def coarse_align(msi_mask: np.ndarray, he_mask: np.ndarray):
    mx, my, mw, mh = bounding_box(msi_mask)
    hx, hy, hw, hh = bounding_box(he_mask)
    src = np.float32([[mx, my], [mx + mw, my], [mx, my + mh]])
    dst = np.float32([[hx, hy], [hx + hw, hy], [hx, hy + hh]])
    return cv2.getAffineTransform(src, dst)

def cv_warp(image: np.ndarray, M: np.ndarray, out_shape: tuple) -> np.ndarray:
    h, w = out_shape[:2]
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)

# -------------------------
# 5. SimpleITK helpers
# -------------------------
def to_sitk(arr: np.ndarray) -> sitk.Image:
    return sitk.Cast(sitk.GetImageFromArray(arr.astype(np.float32)), sitk.sitkFloat32)

def from_sitk(img: sitk.Image) -> np.ndarray:
    return sitk.GetArrayFromImage(img).astype(np.float32)

def resample(moving: np.ndarray, transform: sitk.Transform, reference: np.ndarray) -> np.ndarray:
    return from_sitk(sitk.Resample(
        to_sitk(moving), to_sitk(reference),
        transform, sitk.sitkLinear, 0.0, sitk.sitkFloat32,
    ))

# -------------------------
# 6a. Rigid MI (SimpleITK)
# -------------------------
def sitk_rigid(moving: np.ndarray, fixed: np.ndarray) -> sitk.Transform:
    fixed_s  = to_sitk(fixed)
    moving_s = to_sitk(moving)
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.20)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsRegularStepGradientDescent(
        learningRate=1.0, minStep=1e-4,
        numberOfIterations=200, gradientMagnitudeTolerance=1e-6,
    )
    reg.SetOptimizerScalesFromPhysicalShift()
    init_tx = sitk.CenteredTransformInitializer(
        fixed_s, moving_s, sitk.Euler2DTransform(),
        sitk.CenteredTransformInitializerFilter.MOMENTS,
    )
    reg.SetInitialTransform(init_tx, inPlace=False)
    reg.SetShrinkFactorsPerLevel([4, 2, 1])
    reg.SetSmoothingSigmasPerLevel([2, 1, 0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    tx = reg.Execute(fixed_s, moving_s)
    print(f"      Rigid MI   -- metric: {reg.GetMetricValue():.5f} | {reg.GetOptimizerStopConditionDescription()}")
    return tx

# -------------------------
# 6b. Optional Demons
# -------------------------
def sitk_demons(moving: np.ndarray, fixed: np.ndarray, iterations: int = 50) -> sitk.DisplacementFieldTransform:
    demons = sitk.SymmetricForcesDemonsRegistrationFilter()
    demons.SetNumberOfIterations(iterations)
    demons.SetStandardDeviations(1.5)
    disp = demons.Execute(to_sitk(fixed), to_sitk(moving))
    print(f"      Demons     -- RMS: {demons.GetRMSChange():.5f} | metric: {demons.GetMetric():.5f}")
    return sitk.DisplacementFieldTransform(disp)

# -------------------------
# 7. PSF weight matrix and solver
# -------------------------
def compute_weights_from_labels(he_label: np.ndarray, maldi_coords: np.ndarray, psf_sigma: float, radius_factor: float = 3.0):
    H, Wimg = he_label.shape
    cell_ids = np.unique(he_label)
    cell_ids = cell_ids[cell_ids != 0]
    C = len(cell_ids)
    id_to_index = {cid: i for i, cid in enumerate(cell_ids)}
    P = maldi_coords.shape[0]
    Wmat = lil_matrix((P, C), dtype=np.float32)

    r = int(np.ceil(radius_factor * psf_sigma))
    xs = np.arange(-r, r+1)
    ys = np.arange(-r, r+1)
    xx, yy = np.meshgrid(xs, ys, indexing='xy')
    kernel = np.exp(-(xx**2 + yy**2) / (2 * psf_sigma**2))
    kernel /= kernel.sum()

    for p in range(P):
        xp, yp = maldi_coords[p]
        x0 = int(round(xp))
        y0 = int(round(yp))
        x1, x2 = max(0, x0-r), min(Wimg, x0+r+1)
        y1, y2 = max(0, y0-r), min(H, y0+r+1)
        if x1 >= x2 or y1 >= y2:
            continue
        kx1, kx2 = x1 - (x0-r), x2 - (x0-r)
        ky1, ky2 = y1 - (y0-r), y2 - (y0-r)
        subk = kernel[ky1:ky2, kx1:kx2]
        sublabels = he_label[y1:y2, x1:x2]
        if sublabels.size == 0:
            continue
        unique = np.unique(sublabels)
        for lab in unique:
            if lab == 0:
                continue
            mask = (sublabels == lab)
            wpc = float(subk[mask].sum())
            if wpc > 0:
                Wmat[p, id_to_index[lab]] = wpc

    row_sums = np.array(Wmat.sum(axis=1)).ravel()
    nonzero = row_sums > 0
    for i in np.where(nonzero)[0]:
        normalized = (np.array(Wmat.data[i]) / row_sums[i]).astype(np.float32)
        Wmat.data[i] = normalized.tolist()
    return Wmat.tocsr(), cell_ids

def build_cell_adjacency_laplacian(he_label: np.ndarray, k: int = 6):
    props = measure.regionprops(he_label)
    labels = [p.label for p in props]
    centroids = np.array([p.centroid for p in props])
    if len(labels) == 0:
        return None, np.array([])
    pts = np.vstack([centroids[:,1], centroids[:,0]]).T
    tree = cKDTree(pts)
    C = len(labels)
    rows, cols, data = [], [], []
    for i in range(C):
        dists, idxs = tree.query(pts[i], k=k+1)
        neighbors = idxs[idxs != i]
        for j in neighbors:
            rows.append(i)
            cols.append(j)
            data.append(1.0)
    from scipy.sparse import coo_matrix, diags
    A = coo_matrix((data, (rows, cols)), shape=(C, C))
    A = (A + A.T)
    A.data = np.clip(A.data, 0, 1)
    deg = np.array(A.sum(axis=1)).ravel()
    D = diags(deg)
    L = D - A
    return L.tocsr(), np.array(labels)

def solve_per_cell_nnls(W_csr, I_vec, laplacian=None, lam=0.0):
    if laplacian is not None and lam > 0:
        L = laplacian
        A_top = W_csr
        A_bot = (np.sqrt(lam) * L).tocsr()
        A = np.vstack([A_top.toarray(), A_bot.toarray()])
        b = np.concatenate([I_vec, np.zeros(A_bot.shape[0], dtype=np.float32)])
    else:
        A = W_csr.toarray()
        b = I_vec
    res = lsq_linear(A, b, bounds=(0, np.inf), lsmr_tol='auto', verbose=0)
    return res.x, res.cost, res.status

# -------------------------
# 8. Visualization helpers
# -------------------------
def save_overlay(he_rgb: np.ndarray, ion_reg: np.ndarray, out_path: str, alpha: float = 0.5, cmap: str = "hot"):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(he_rgb);  axes[0].set_title("H&E");                axes[0].axis("off")
    axes[1].imshow(ion_reg, cmap=cmap, vmin=0, vmax=1)
    axes[1].set_title("Registered Ion Image");                         axes[1].axis("off")
    axes[2].imshow(he_rgb)
    rgba = plt.get_cmap(cmap)(np.clip(ion_reg, 0, 1))
    rgba[..., 3] = np.clip(ion_reg, 0, 1) * alpha
    axes[2].imshow((rgba * 255).astype(np.uint8))
    axes[2].set_title(f"Overlay  alpha={alpha}");                      axes[2].axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved overlay  -> {out_path}")

def paint_per_cell_map(he_label: np.ndarray, cell_ids: np.ndarray, O_hat: np.ndarray):
    H, Wimg = he_label.shape
    out = np.zeros((H, Wimg), dtype=np.float32)
    for i, lab in enumerate(cell_ids):
        out[he_label == lab] = O_hat[i]
    if out.max() > 0:
        out_disp = out / out.max()
    else:
        out_disp = out
    return out_disp

# -------------------------
# 9. StarDist segmentation
# -------------------------
def run_stardist(he_rgb, prob_thresh=0.3, nms_thresh=0.8, n_tiles=(4, 4, 1), show_tile_progress=True):
    """
    Run StarDist 2D nuclei segmentation on H&E RGB image.
    he_rgb: uint8 RGB image (H, W, 3)
    """
    # normalize to [0,1] float
    if he_rgb.dtype != np.float32:
        he_norm = he_rgb.astype(np.float32) / 255.0
    else:
        he_norm = he_rgb
    model = StarDist2D.from_pretrained('2D_versatile_he')
    labels, details = model.predict_instances(
        he_norm,
        axes='YXC',
        prob_thresh=prob_thresh,
        nms_thresh=nms_thresh,
        n_tiles=n_tiles,
        show_tile_progress=show_tile_progress
    )
    return labels.astype(np.int32), details

def segmentation_pipeline(he_path: str, save_overlay: bool = False):
    """
    Run StarDist on the H&E image and optionally save an overlay and mask.
    Returns nuclei label image (he_label).
    """
    print("Loading H&E for StarDist segmentation...")
    he_rgb, _ = load_he(he_path)
    print("Running StarDist for nuclei segmentation...")
    nuclei_masks, _ = run_stardist(he_rgb, n_tiles=(8, 8, 1))

    if save_overlay and HAS_TIFF:
        print("Creating nuclei overlay...")
        he_uint8 = he_rgb.astype(np.uint8)
        boundaries = morphological_gradient(nuclei_masks, size=3) > 0
        overlay = he_uint8.copy()
        overlay[boundaries] = [255, 0, 0]
        base, ext = os.path.splitext(he_path)
        overlay_path = base + "_nuclei_overlay.tif"
        mask_path = base + "_nuclei_masks.tif"
        tifffile.imwrite(overlay_path, overlay, compression='lzma', photometric='rgb')
        tifffile.imwrite(mask_path, nuclei_masks.astype(np.uint16), compression='lzma')
        print(f"✓ Saved overlay: {overlay_path}")
        print(f"✓ Saved masks:   {mask_path}")

    print("✓ StarDist segmentation completed.")
    return nuclei_masks

# -------------------------
# 10. Main pipeline
# -------------------------
def run_pipeline(imzml_path, he_path, target_mz, mz_tol, out_dir,
                 alpha=0.5, use_demons=False, psf_sigma=1.0, knn=6, lam=0.01):
    os.makedirs(out_dir, exist_ok=True)
    print("\n" + "="*60)
    print("  MALDI MSI -> H&E  |  PSF-aware Registration + Unmixing (StarDist labels)")
    print("="*60)

    # 1. Ion image
    print("\n[1/8] Extracting ion image ...")
    ion = extract_ion_image(imzml_path, target_mz, mz_tol)
    print(f"      MSI grid: {ion.shape}  max={ion.max():.3f}")

    # 2. H&E
    print("[2/8] Loading H&E ...")
    he_rgb, he_grey = load_he(he_path)
    print(f"      H&E shape: {he_rgb.shape}")

    # 3. Pre-smooth MALDI with PSF (in MALDI pixel units)
    print("[3/8] Applying PSF smoothing to MALDI ...")
    if psf_sigma > 0:
        ion_smooth = gaussian_filter(ion, sigma=psf_sigma)
    else:
        ion_smooth = ion.copy()

    # 4. Tissue masks
    print("[4/8] Computing tissue masks ...")
    he_mask  = tissue_mask_he(he_grey)
    msi_mask = tissue_mask_msi(ion_smooth)
    msi_mask_up = resize(msi_mask.astype(float), he_grey.shape, order=0, preserve_range=True, anti_aliasing=False).astype(np.uint8)

    # 5. Coarse bounding-box alignment
    print("[5/8] Coarse bounding-box alignment ...")
    ion_up = resize(ion_smooth, he_grey.shape, order=1, preserve_range=True, anti_aliasing=True).astype(np.float32)
    M_coarse = coarse_align(msi_mask_up, he_mask)
    ion_coarse = cv_warp(ion_up, M_coarse, he_grey.shape)

    # 6. SimpleITK rigid MI registration
    print("[6/8] SimpleITK Mattes MI rigid registration ...")
    rigid_tx = sitk_rigid(ion_coarse, he_grey)
    ion_rigid = resample(ion_coarse, rigid_tx, he_grey)

    # 7. Optional Demons deformable
    if use_demons:
        print("      Running Symmetric Demons deformable refinement ...")
        demons_tx = sitk_demons(ion_rigid, he_grey)
        ion_final = resample(ion_rigid, demons_tx, he_grey)
    else:
        ion_final = ion_rigid
    ion_final = np.clip(ion_final, 0.0, 1.0)

    # Save intermediate visualizations
    plt.imsave(os.path.join(out_dir, "ion_raw.png"), ion, cmap="hot")
    plt.imsave(os.path.join(out_dir, "ion_smoothed.png"), ion_smooth, cmap="hot")
    plt.imsave(os.path.join(out_dir, "ion_coarse.png"), ion_coarse, cmap="hot")
    plt.imsave(os.path.join(out_dir, "ion_registered.png"), ion_final, cmap="hot")
    sitk.WriteTransform(rigid_tx, os.path.join(out_dir, "rigid_transform.tfm"))
    print("  Saved rigid transform -> rigid_transform.tfm")

    # 8. PSF-based pixel->cell mapping using StarDist labels
    print("[7/8] StarDist nuclei segmentation for cell labels ...")
    he_label = segmentation_pipeline(he_path, save_overlay=True)

    print("[8/8] PSF-based pixel->cell mapping and NNLS solve ...")
    # MALDI pixel coordinates in H&E pixel space
    H_m, W_m = ion.shape
    coords = []
    for y in range(H_m):
        for x in range(W_m):
            coords.append((x + 0.5, y + 0.5))
    coords = np.array(coords)
    scale_x = he_grey.shape[1] / float(W_m)
    scale_y = he_grey.shape[0] / float(H_m)
    coords_scaled = coords.copy()
    coords_scaled[:, 0] *= scale_x
    coords_scaled[:, 1] *= scale_y
    ones = np.ones((coords_scaled.shape[0], 1), dtype=np.float32)
    pts = np.hstack([coords_scaled, ones])
    M = M_coarse
    pts_warp = (M @ pts.T).T
    sitk_tx = rigid_tx
    final_coords = []
    for (xw, yw) in pts_warp:
        try:
            xf, yf = sitk_tx.TransformPoint((float(xw), float(yw)))
        except Exception:
            xf, yf = float(xw), float(yw)
        final_coords.append((xf, yf))
    final_coords = np.array(final_coords)

    H_he, W_he = he_grey.shape
    valid_mask = (final_coords[:,0] >= 0) & (final_coords[:,0] < W_he) & (final_coords[:,1] >= 0) & (final_coords[:,1] < H_he)
    maldi_coords = final_coords[valid_mask]
    maldi_values = ion.flatten()[valid_mask]

    W_sparse, cell_ids = compute_weights_from_labels(he_label, maldi_coords, psf_sigma=psf_sigma, radius_factor=3.0)
    print(f"      W shape: {W_sparse.shape}  nonzero rows: {W_sparse.getnnz(axis=1).astype(bool).sum()}")

    L, label_order = build_cell_adjacency_laplacian(he_label, k=knn)
    if L is None:
        print("      No cells found in segmentation; skipping regularization.")
        L = None

    print("      Solving regularized nonnegative least squares ...")
    O_hat, cost, status = solve_per_cell_nnls(W_sparse, maldi_values, laplacian=L, lam=lam if L is not None else 0.0)
    print(f"      Solver cost: {cost:.6f}  status: {status}")

    I_hat = W_sparse.dot(O_hat)
    mse = float(np.mean((maldi_values - I_hat)**2))
    corr = np.corrcoef(maldi_values, I_hat)[0,1] if maldi_values.size > 1 else np.nan
    print(f"      Forward check: RMSE={np.sqrt(mse):.6f}  corr={corr:.4f}")

    per_cell_map = paint_per_cell_map(he_label, cell_ids, O_hat)
    plt.imsave(os.path.join(out_dir, "per_cell_map.png"), per_cell_map, cmap="magma")
    np.save(os.path.join(out_dir, "per_cell_intensities.npy"), O_hat)
    np.save(os.path.join(out_dir, "maldi_coords.npy"), maldi_coords)
    np.save(os.path.join(out_dir, "maldi_values.npy"), maldi_values)

    save_overlay(he_rgb, ion_final, os.path.join(out_dir, "overlay.png"), alpha=alpha, cmap="hot")
    rgba = plt.get_cmap("magma")(per_cell_map)
    rgba[..., 3] = per_cell_map * 0.6
    fig, ax = plt.subplots(1,1, figsize=(8,8))
    ax.imshow(he_rgb)
    ax.imshow((rgba * 255).astype(np.uint8))
    ax.axis("off")
    plt.savefig(os.path.join(out_dir, "per_cell_overlay.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\n  Done. Outputs in:", out_dir)
    return {
        "ion_final": ion_final,
        "per_cell_map": per_cell_map,
        "per_cell_intensities": O_hat,
        "maldi_coords": maldi_coords,
        "maldi_values": maldi_values,
        "W": W_sparse
    }

# -------------------------
# 11. CLI
# -------------------------
def main():
    p = argparse.ArgumentParser(description="Register MALDI MSI to H&E and compute PSF-based per-cell intensities (StarDist labels).")
    p.add_argument("--imzml",  required=True, help=".imzML file path")
    p.add_argument("--he",     required=True, help="H&E image (TIF/PNG/JPG)")
    p.add_argument("--mz",     type=float, required=True, help="Target m/z")
    p.add_argument("--tol",    type=float, default=0.3, help="m/z tolerance Da (default 0.3)")
    p.add_argument("--out",    default="maldi_he_output", help="Output directory")
    p.add_argument("--alpha",  type=float, default=0.5, help="Overlay opacity (default 0.5)")
    p.add_argument("--demons", action="store_true", help="Add Symmetric Demons deformable step after rigid")
    p.add_argument("--psf_sigma", type=float, default=1.0, help="PSF sigma in MALDI pixels (default 1.0)")
    p.add_argument("--knn", type=int, default=6, help="k for k-NN adjacency when building Laplacian (default 6)")
    p.add_argument("--lam", type=float, default=0.01, help="Tikhonov regularization weight (default 0.01)")
    args = p.parse_args()
    run_pipeline(args.imzml, args.he, args.mz, args.tol, args.out,
                 alpha=args.alpha, use_demons=args.demons,
                 psf_sigma=args.psf_sigma, knn=args.knn, lam=args.lam)

if __name__ == "__main__":
    main()

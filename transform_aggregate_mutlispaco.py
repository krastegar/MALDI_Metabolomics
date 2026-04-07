"""
Multiscale SpaCo Pipeline
-------------------------
End-to-end pipeline: load -> mask -> pyramid -> aggregate -> eigensolver -> plot.
No classes. Pass data explicitly between steps.
"""

from __future__ import annotations
from pathlib import Path
from typing  import Optional
from scipy.spatial     import cKDTree
from scipy.sparse.linalg import LinearOperator, lobpcg
from scipy.linalg      import cholesky, eigh
from skimage.measure   import label as sk_label
from sklearn.neighbors import KDTree
from pyimzml.ImzMLParser import ImzMLParser
import numpy  as np
import pandas as pd
import scipy.sparse as sp
import scanpy as sc
import napari
from skimage import io
import matplotlib.cm as mcm
import matplotlib.colors as mcolors

# =============================================================================
# 1. DATA LOADING
# =============================================================================

def load_maldi(data_path: str) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
    """
    Load MALDI-MSI data from an imzML file.
    Returns count matrix (n_spots, n_mz), integer coords (n_spots, 2), m/z values.
    """
    parser     = ImzMLParser(Path(data_path))
    spectra    = [parser.getspectrum(i) for i in range(len(parser.coordinates))]
    mzs        = [s[0] for s in spectra]
    intensities= [s[1] for s in spectra]

    # Build COO triplets: row = spot index, col = unique m/z bin
    all_mzs        = np.concatenate(mzs)
    unique_mzs, col_idx = np.unique(all_mzs, return_inverse=True)
    row_idx        = np.repeat(np.arange(len(mzs)), [len(m) for m in mzs])
    coords         = np.array(parser.coordinates)[:, :2]  # integer (x, y)

    X = sp.coo_matrix((np.concatenate(intensities), (row_idx, col_idx)),
                      shape=(len(mzs), len(unique_mzs))).tocsr()
    print(f"Loaded MALDI | spots: {X.shape[0]:,} | m/z bins: {X.shape[1]:,}")
    return X, coords, unique_mzs


def load_visium(data_path: str) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
    """
    Load Visium HD data from a CellRanger output directory.
    Returns count matrix (n_spots, n_genes), float pixel coords, gene names.
    """
    path  = Path(data_path)
    adata = sc.read_10x_h5(path / "filtered_feature_bc_matrix.h5")
    adata.var_names_make_unique()

    # Align spatial coordinates to count matrix barcodes
    pos = (pd.read_parquet(path / "spatial" / "tissue_positions.parquet")
             .set_index("barcode").loc[adata.obs_names])
    coords = pos[["pxl_col_in_fullres", "pxl_row_in_fullres"]].to_numpy()

    X = sp.csr_matrix(adata.X)
    print(f"Loaded Visium | spots: {X.shape[0]:,} | genes: {X.shape[1]:,}")
    return X, coords, adata.var_names.to_numpy()


# =============================================================================
# 2. BASE MASK (level 0)
# =============================================================================
 
def build_base_mask(coords: np.ndarray, is_float: bool = False) -> tuple[sp.csr_matrix, np.ndarray]:
    """
    Build binary CSR mask M0 and discretized grid coords from raw spot coordinates.
    For MALDI: coords are already integers. For Visium: discretize via median NN distance.
 
    Returns
    -------
    M0          : sp.csr_matrix  binary mask (H, W)
    grid_coords : np.ndarray     zero-based integer grid indices (n_spots, 2)
    coord_info  : dict           offset and resolution needed to map back to raw pixel space
                  {
                    'offset'     : (row_min, col_min) in raw pixel coords,
                    'resolution' : bin size in raw pixel units (1.0 for MALDI),
                  }
    """
    if is_float:
        # Infer bin size from median nearest-neighbour distance (O(n log n) via KDTree C backend)
        distances, _ = KDTree(coords).query(coords, k=2)
        resolution   = float(np.median(distances[:, 1]))
        print(f"Inferred resolution: {resolution:.4f}px")
        offset       = coords.min(axis=0)                    # (row_min, col_min) in pixel space
        grid_coords  = np.floor((coords - offset) / resolution).astype(np.int32)
    else:
        resolution   = 1.0
        offset       = coords.min(axis=0).astype(np.float64)
        grid_coords  = (coords - coords.min(axis=0)).astype(np.int32)
 
    # coords[:, 0] = pxl_col, coords[:, 1] = pxl_row
    # grid rows = pxl_row direction, grid cols = pxl_col direction
    rows = grid_coords[:, 1]   # pxl_row -> grid row
    cols = grid_coords[:, 0]   # pxl_col -> grid col
    M0 = sp.coo_matrix((np.ones(len(rows), dtype=np.uint8), (rows, cols)),
                       shape=(rows.max() + 1, cols.max() + 1)).tocsr()
 
    # Update grid_coords to (row, col) convention — consistent with mask
    grid_coords = np.column_stack([rows, cols]).astype(np.int32)
 
    coord_info = {'offset': offset, 'resolution': resolution}
    return M0, grid_coords, coord_info

# =============================================================================
# 3. PYRAMID MASK CONSTRUCTION
# =============================================================================

def coarsen_mask(M: sp.csr_matrix) -> sp.csr_matrix:
    """
    Coarsen binary mask by 2x: a coarse cell is valid if >=2 of its 4 children are valid.
    Enforces 8-connectivity by keeping only the largest connected component.
    """
    # Dense detour required for reshape trick and sk_label — mask is small vs count matrix
    M_dense = M.toarray()
    H, W    = M_dense.shape[0] // 2, M_dense.shape[1] // 2

    # Count valid children in each 2x2 block via reshape — O(H*W)
    counts  = M_dense[:2*H, :2*W].reshape(H, 2, W, 2).sum(axis=(1, 3))
    coarse  = (counts >= 2).astype(np.uint8)

    # Keep only the largest 8-connected component (C backend via skimage)
    labels  = sk_label(coarse, connectivity=2)
    bc      = np.bincount(labels.flat)
    coarse  = (labels == np.argmax(bc[1:]) + 1).astype(np.uint8)

    return sp.coo_matrix(coarse).tocsr()


def build_pyramid_masks(M0: sp.csr_matrix, target_max: int,
                        plot: bool = True) -> dict[int, dict]:
    """
    Iteratively coarsen M0 until n_valid < target_max.
    Returns dict keyed by level with 'mask', 'tissue_idx', 'grid_shape'.
    """
    def _pack(mask: sp.csr_matrix) -> dict:
        # Flat tissue indices: row*W + col — O(nnz) via COO coords
        coo = mask.tocoo()
        return {'mask': mask, 'grid_shape': mask.shape,
                'tissue_idx': coo.row * mask.shape[1] + coo.col}

    levels, M = {0: _pack(M0)}, M0
    lvl = 0
    while M.nnz >= target_max:
        print(f"Level {lvl}: n = {M.nnz:,}")
        M   = coarsen_mask(M)
        lvl += 1
        levels[lvl] = _pack(M)

    print(f"✓ Pyramid: {lvl + 1} levels | coarsest n = {M.nnz:,}")

    if plot:
        import matplotlib.pyplot as plt
        n_cols = min(5, len(levels))
        n_rows = int(np.ceil(len(levels) / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(5 * n_cols, 5 * n_rows), squeeze=False)
        for idx, (l, d) in enumerate(levels.items()):
            ax = axes[idx // n_cols, idx % n_cols]
            ax.imshow(d['mask'].toarray(), cmap='Spectral', interpolation='nearest', vmin=0, vmax=1)
            ax.set_title(f"Level {l}: {d['grid_shape']}")
            ax.axis('off')
        for idx in range(len(levels), n_rows * n_cols):
            axes[idx // n_cols, idx % n_cols].axis('off')
        plt.tight_layout()
        plt.show()

    return levels


# =============================================================================
# 4. AGGREGATE COUNT DATA ONTO PYRAMID
# =============================================================================

def _transform(X: sp.csr_matrix, method: str) -> sp.csr_matrix:
    """
    Variance-stabilizing transform applied once to raw counts at level 0.
    Operates only on stored non-zero values — O(nnz), no dense materialisation.

    Options
    -------
    log1p   : log(1 + x)                      — simple, widely used
    anscombe: 2*sqrt(x + 3/8)                 — tailored to Poisson (Visium counts)
    arcsinh : arcsinh(x / median_nonzero)      — recommended for MALDI / mass-spec
    """
    X = X.tocsr().astype(np.float32)
    if method == 'log1p':
        X.data = np.log1p(X.data)
    elif method == 'anscombe':
        # Anscombe: variance-stabilises Poisson data, E[f(X)] ≈ 2*sqrt(lambda)
        X.data = 2.0 * np.sqrt(X.data + 0.375)
    elif method == 'arcsinh':
        # arcsinh(x/c): c = median of non-zero values — widely used in mass-spec
        c      = float(np.median(X.data))
        X.data = np.arcsinh(X.data / c)
    else:
        raise ValueError(f"Unknown transform '{method}'. Choose: log1p, anscombe, arcsinh.")
    return X


def aggregate_pyramid(X: sp.csr_matrix, grid_coords: np.ndarray,
                      levels: dict[int, dict],
                      data_type: str = 'visium',
                      transform: str = None) -> dict[int, dict]:
    """
    Transform raw data once at level 0, then aggregate by simple averaging at
    each coarser level — no re-transformation. This preserves the analytical
    properties of the transform and yields a clean multiscale kernel interpretation.

    Parameters
    ----------
    X          : raw count / intensity matrix (n_spots, n_feats)
    grid_coords: integer grid coordinates (n_spots, 2)
    levels     : output of build_pyramid_masks()
    data_type  : 'visium' → anscombe (Poisson counts)
                 'maldi'  → arcsinh  (mass-spec intensities)
    transform  : override default transform ('log1p', 'anscombe', 'arcsinh')
    """
    # Select default transform based on data type
    if transform is None:
        transform = 'anscombe' if data_type == 'visium' else 'arcsinh'
    print(f"Transform: {transform} (data_type={data_type})")

    X = X.tocsr() if not isinstance(X, sp.csr_matrix) else X

    # Apply variance-stabilising transform once to raw data — never again
    X_t = _transform(X, transform)

    fine_row = (grid_coords[:, 0] - grid_coords[:, 0].min()).astype(np.int32)
    fine_col = (grid_coords[:, 1] - grid_coords[:, 1].min()).astype(np.int32)

    for lvl, d in levels.items():
        H, W     = d['grid_shape']
        scale    = 2 ** lvl

        # Map each spot to its coarse grid cell at this level
        row_c    = fine_row // scale
        col_c    = fine_col // scale
        in_bounds= (row_c >= 0) & (row_c < H) & (col_c >= 0) & (col_c < W)
        flat_idx = (row_c[in_bounds] * W + col_c[in_bounds]).astype(np.int32)
        spot_idx = np.where(in_bounds)[0]

        # Sparse aggregation matrix A: A[pixel, spot] = 1/count — gives mean directly
        n_valid  = len(spot_idx)
        A        = sp.coo_matrix((np.ones(n_valid), (flat_idx, np.arange(n_valid))),
                                 shape=(H * W, n_valid)).tocsr()

        # Average transformed values — pure linear aggregation, no further transform
        counts   = np.asarray(A.sum(axis=1))
        occ      = np.where(counts.flatten() > 0)[0]
        means    = (A @ X_t[spot_idx])[occ].multiply(1.0 / counts[occ]).tocsr()

        # Reconstruct full pixel space via COO — never materialises dense
        rows_local, cols_out = means.nonzero()
        vals_out             = np.asarray(means[rows_local, cols_out]).ravel()
        d['data'] = sp.coo_matrix((vals_out, (occ[rows_local], cols_out)),
                                  shape=(H * W, X.shape[1])).tocsr()
        print(f"Level {lvl} | grid: {H}x{W} | tissue: {len(occ):,} | nnz: {d['data'].nnz:,}")

    return levels


# =============================================================================
# 5. SPACO EIGENSOLVER
# =============================================================================

def _make_operators(Y: sp.csr_matrix, coords: np.ndarray,
                    k: int = 8) -> tuple[LinearOperator, LinearOperator]:
    """
    Build sparse LHS (Y_c^T H Y_c) and RHS (Y_c^T Y_c) LinearOperators for LOBPCG.
    Spatial smoothing H uses packed KNN adjacency — O(n*k) memory.
    """
    mu   = np.asarray(Y.mean(axis=0)).ravel()           # column means
    n, p = Y.shape

    # KNN graph: src/dst edge lists + degree — O(n log n) C backend
    _, idx = cKDTree(coords).query(coords, k=k + 1)
    src    = np.repeat(np.arange(n), k).astype(np.int64)
    dst    = idx[:, 1:].reshape(-1).astype(np.int64)
    deg    = np.maximum(np.bincount(src, minlength=n).astype(float), 1.0)

    def _scores(V):
        # Y_c @ V = Y @ V - mu*(1^T V)
        V = np.asarray(V, dtype=np.float64).reshape(p, -1)
        return np.asarray(Y @ V) - (mu @ V)[None, :]

    def _backproject(U):
        # Y_c^T @ U = Y^T @ U - mu*(1^T U)
        return np.asarray(Y.T @ U) - np.outer(mu, U.sum(axis=0))

    def _apply_H(U):
        # H = P D^{-1/2} A D^{-1/2} P, applied as P H_raw P
        Uc  = U - U.mean(axis=0)
        out = np.zeros_like(Uc)
        np.add.at(out, src, 0.5 * (Uc[dst] / deg[src, None] + Uc[dst] / deg[dst, None]))
        return out - out.mean(axis=0)

    # Preserve Y dtype (float32 at fine levels saves ~2x memory per matvec)
    dtype = Y.dtype
    Aop = LinearOperator((p, p), dtype=dtype,
          matmat=lambda V: _backproject(_apply_H(_scores(V))),
          matvec=lambda v: _backproject(_apply_H(_scores(v))).ravel())

    Bop = LinearOperator((p, p), dtype=dtype,
          matmat=lambda V: _backproject(_scores(V)),
          matvec=lambda v: _backproject(_scores(v)).ravel())

    return Aop, Bop, mu


def _b_orthonormalize(X: np.ndarray, Bop: LinearOperator, eps: float = 1e-8) -> np.ndarray:
    """B-orthonormalize columns of X: X^T B X = I. Falls back to eigh if Cholesky fails."""
    G = X.T @ (Bop @ X);  G = 0.5 * (G + G.T) + eps * np.eye(G.shape[0])
    try:
        return X @ np.linalg.inv(cholesky(G, lower=False))
    except np.linalg.LinAlgError:
        evals, evecs = eigh(G)
        keep = evals > eps
        return X @ evecs[:, keep] @ np.diag(1.0 / np.sqrt(evals[keep]))


def solve_spaco_level(Y: sp.csr_matrix, coords: np.ndarray, keigs: int = 50,
                      b: int = 64, niter: int = 8, init_vecs: np.ndarray = None,
                      seed: int = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve one SpaCo level: Y_c^T H Y_c v = λ Y_c^T Y_c v via LOBPCG.
    Returns (eigvals, eigvecs, mu) — eigvecs shape (p, keigs).
    """
    Y        = Y.tocsr() if sp.issparse(Y) else Y
    Aop, Bop, mu = _make_operators(Y, coords)
    rng      = np.random.default_rng(seed)
    p        = Y.shape[1]

    # Warm-start: pad coarse eigenvectors with random columns, then B-orthonormalize
    X0 = np.zeros((p, b))
    if init_vecs is not None:
        nc = min(init_vecs.shape[1], b)
        X0[:, :nc] = init_vecs[:, :nc]
    X0[:, (0 if init_vecs is None else nc):] = rng.standard_normal((p, b - (0 if init_vecs is None else nc)))
    X0 = _b_orthonormalize(X0, Bop)

    eigvals, eigvecs = lobpcg(A=Aop, X=X0, B=Bop, largest=True, maxiter=niter, verbosityLevel=0)

    # Sort descending, trim to keigs, final B-orthonormalization
    order    = np.argsort(eigvals)[::-1]
    eigvals  = eigvals[order][:keigs]
    eigvecs  = _b_orthonormalize(eigvecs[:, order][:, :keigs], Bop)
    return eigvals, eigvecs, mu


def run_multiscale_spaco(levels: dict[int, dict], keigs: int = 50,
                         b: int = 64, niter: int = 8,
                         seed: int = 42) -> dict[int, dict]:
    """
    Run SpaCo coarsest -> finest, using each level's eigvecs to warm-start the next.
    Populates 'eigvals', 'eigvecs', 'mu' in each level dict.
    fine_b: override block size at the finest level to reduce memory (default: b // 2).
    """

    init_vecs = None
    finest    = min(levels.keys())
    for lvl in sorted(levels.keys(), reverse=True):   # coarsest first
        d         = levels[lvl]
        H, W      = d['grid_shape']
        tissue    = d['tissue_idx']
        # float32 halves matvec memory vs float64 — critical at fine levels
        Y         = d['data'][tissue].astype(np.float32)
        coords    = np.column_stack([tissue // W, tissue % W]).astype(np.float64)
        # Reduce block size at finest level to fit in RAM
        #level_b   = (fine_b if fine_b is not None else max(keigs + 2, b // 2)) if lvl == finest else b

        print(f"Solving level {lvl} | tissue spots: {len(tissue):,} | block size: {b}")
        eigvals, eigvecs, mu = solve_spaco_level(Y, coords, keigs=keigs,
                                                  b=b, niter=niter,
                                                  init_vecs=init_vecs, seed=seed + lvl)
        # Compute and store scores (n_tissue, keigs) before freeing data
        # S = Y_c @ V = Y @ V - mu*(1^T V) — stored in spot space for plotting
        scores = np.asarray(Y @ eigvecs) - (mu @ eigvecs)[None, :]
        d['eigvals'], d['eigvecs'], d['mu'], d['scores'] = eigvals, eigvecs, mu, scores
        init_vecs = eigvecs                            # warm-start for next finer level
        del d['data']                                  # free count matrix — no longer needed

    return levels


# =============================================================================
# 6. PLOTTING
# =============================================================================

def plot_spaco(levels: dict[int, dict],
               components: int | list[int] = 0,
               level: int | list[int] | None = None,
               cmap: str = "Spectral", panel_size: int = 500,
               max_cols: int = 2, invert_y: bool = False) -> None:
    """
    Datashader-based scatter plot of SpaCo scores — handles millions of points
    by rasterizing server-side before rendering. Outputs a static image grid.
 
    Install: uv pip install datashader matplotlib
 
    Parameters
    ----------
    levels     : output of run_multiscale_spaco
    components : single component index or list of indices (0-indexed)
    level      : single level, list of levels, or None for all solved levels
    cmap       : matplotlib colormap name (e.g. 'Spectral', 'viridis', 'RdBu')
    panel_size : pixel resolution of each rasterized panel
    max_cols   : maximum number of panels per row (default: 2)
    invert_y   : flip y-axis to match image orientation
    """
    import datashader as ds
    import datashader.transfer_functions as tf
    import matplotlib.pyplot as plt
    import matplotlib.cm as mcm
    import matplotlib.colors as mcolors
    from matplotlib.colors import Normalize
 
    comps  = [components] if isinstance(components, int) else list(components)
    solved = {k: v for k, v in sorted(levels.items()) if 'scores' in v}
    if not solved:
        print("No solved levels found — run run_multiscale_spaco first.")
        return
 
    if level is not None:
        keys   = [level] if isinstance(level, int) else list(level)
        solved = {k: v for k, v in solved.items() if k in keys}
        if not solved:
            print(f"No solved data found for level(s) {keys}.")
            return
 
    # Flatten all (level, component) pairs — at most 2 per row
    panels = [(lvl, d, c) for c in comps for lvl, d in solved.items()]
    n_cols = min(max_cols, len(panels))
    n_rows = int(np.ceil(len(panels) / n_cols))
 
    # constrained_layout handles colorbars automatically — no shifting or clipping
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(7.5 * n_cols, 6 * n_rows), squeeze=False,
                             constrained_layout=True)
 
    # Build colormap once — shared across all panels
    mpl_cmap   = mcm.get_cmap(cmap)
    hex_colors = [mcolors.to_hex(mpl_cmap(i / 255)) for i in range(256)]
 
    for idx, (lvl, d, comp) in enumerate(panels):
        ax     = axes[idx // n_cols, idx % n_cols]
        tissue = d['tissue_idx']
        H, W   = d['grid_shape']
        x      = (tissue % W).astype(np.float32)
        y      = (tissue // W).astype(np.float32)
        scores = d['scores'][:, comp].astype(np.float32)
        if invert_y:
            y = y.max() - y
 
        if len(tissue) < 50_000:
            # Coarse level: reconstruct (H, W) grid — imshow fills each bin correctly
            grid = np.full((H, W), np.nan)
            grid[y.astype(int), x.astype(int)] = scores
            im = ax.imshow(grid[::-1] if invert_y else grid, cmap=mpl_cmap,
                           interpolation='nearest', aspect='equal', origin='upper')
        else:
            # Fine level: rasterize via datashader — O(panel_size^2) not O(n_spots)
            df  = pd.DataFrame({'x': x.astype(float), 'y': y.astype(float), 'score': scores.astype(float)})
            cvs = ds.Canvas(plot_width=panel_size, plot_height=panel_size)
            agg = cvs.points(df, 'x', 'y', ds.mean('score'))
            img = tf.spread(tf.shade(agg, cmap=hex_colors, how='linear'), px=1)
            im  = ax.imshow(img.to_pil(), origin='upper' if invert_y else 'lower', aspect='equal')
 
        # Compact colorbar — no label, just tick values
        norm = Normalize(vmin=scores.min(), vmax=scores.max())
        sm   = mcm.ScalarMappable(norm=norm, cmap=mpl_cmap)
        cb   = plt.colorbar(sm, ax=ax, fraction=0.035, pad=0.02, shrink=0.8)
        cb.ax.tick_params(labelsize=7)
 
        # Concise title — small font to avoid overflow
        ax.set_title(f"L{lvl} | SpaC {comp + 1} | n={len(tissue):,}",
                     fontsize=9, pad=4)
        ax.axis('off')
 
    for idx in range(len(panels), n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].axis('off')
 
    fig.suptitle("Multiscale SpaCo Components", fontsize=12)
    plt.show()
 
def visium_grid_to_he_coords(tissue_idx: np.ndarray, grid_shape: tuple,
                      coord_info: dict, lvl: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert pyramid tissue indices to H&E pixel coordinates.
 
    Parameters
    ----------
    tissue_idx : np.ndarray  flat grid indices of tissue spots
    grid_shape : (H, W)      grid dimensions at this level
    coord_info : dict        output of build_base_mask — offset and resolution
    lvl        : int         pyramid level (used to scale back to full resolution)
 
    Returns
    -------
    he_row, he_col : np.ndarray  exact pixel coordinates in H&E image space
    """
    H, W       = grid_shape
    scale      = 2 ** lvl
    offset     = coord_info['offset']
    resolution = coord_info['resolution']
 
    # tissue_idx = row * W + col  (row = pxl_row direction, col = pxl_col direction)
    grid_row = (tissue_idx // W).astype(np.float32)   # pxl_row direction
    grid_col = (tissue_idx  % W).astype(np.float32)   # pxl_col direction
 
    # Invert discretization: grid_idx * scale * resolution + offset = raw pixel coord
    he_row = grid_row * scale * resolution + offset[1]  # offset[1] = row_min
    he_col = grid_col * scale * resolution + offset[0]  # offset[0] = col_min
 
    # Reflect col off y-axis to correct for mirror flip in H&E coordinate system
    he_col = he_col.max() - he_col
 
    return he_row, he_col
# =============================================================================
# Napari overlay function — Optional 
# =============================================================================
def spaco_scores_to_napari_points(level_data: dict, coord_info: dict,
                                  he_path: str, n_components: int = 9,
                                  n_points: int = None, 
                                  opacity: int = 0.5,
                                  point_size: int = 0.2) -> None:
    """
    Overlay SpaCo scores on H&E image in Napari.

    Parameters
    ----------
    level_data    : single level dict from levels[lvl]
    coord_info    : output of build_base_mask
    he_path       : path to tissue_hires_image.png
    n_components  : number of SpaCo components to add as layers
    n_points      : if not None, subsample every n_points-th spot for faster rendering
    """
    import napari
    from skimage.io import imread

    tissue = level_data['tissue_idx']
    H, W   = level_data['grid_shape']
    pcs    = level_data['scores']

    he_row, he_col = visium_grid_to_he_coords(tissue, (H, W), coord_info)
    points = np.column_stack([he_row, he_col])

    # Subsample if requested — every n_points-th spot
    if n_points is not None:
        points = points[::n_points]
        pcs    = pcs[::n_points]

    print(f"Rendering {len(points):,} points")

    he_img = imread(he_path)
    viewer = napari.Viewer()
    viewer.add_image(he_img, name="H&E")

    for i in range(min(pcs.shape[1], n_components)):
        viewer.add_points(
            points,
            features={'score': pcs[:, i]},
            face_color='score',
            face_colormap='Spectral',
            size=point_size,
            opacity=opacity,
            name=f"SpaC {i+1}",
            visible=(i == 0),
            blending='translucent'
        )

    napari.run()
# =============================================================================
# PIPELINE ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # call to napari overlay function — adjust n_points for faster rendering if needed
    #full_pyramid = np.load("full_data_pyramid.npy", allow_pickle=True).item()
    #spaco_scores_to_napari_points(level_data = full_pyramid[0], coord_info=coord_info, 
#                                  he_path='MSI_data_grant/cellranger/329537/outs/binned_outputs/square_002um/spatial/tissue_hires_image.png', 
#                                  n_points=2, opacity=0.6, point_size=1.5)
    
        # --- Visium HD ---
    X, coords, features = load_visium("MSI_data_grant/cellranger/329537/outs/binned_outputs/square_002um/")
    M0, visium_grid_coords, coord_info = build_base_mask(coords, is_float=True)
    levels              = build_pyramid_masks(M0, target_max=5_000, plot=True)
    vis_levels              = aggregate_pyramid(X, visium_grid_coords, levels, data_type='visium')
    subset_levels =  {k: v for k, v in sorted(vis_levels.items()) if k >= 1}  # skip finest level to save memory
    levels              = run_multiscale_spaco(subset_levels, keigs=20, b=32, niter=10)
    ##levels              = run_multiscale_spaco(levels, keigs=20, b=32, niter=10)
    #plot_spaco(levels, component=0, invert_y=True)
    
    
    # --- MALDI ---
    maldi_X, maldi_coords, features = load_maldi("MSI_data_grant/Mass_Spec_data/20251012_old_liver.imzML")
    M0_maldi, grid_coords, coords_info = build_base_mask(maldi_coords, is_float=False)
    levels              = build_pyramid_masks(M0_maldi, target_max=1_000, plot=True)
    levels              = aggregate_pyramid(maldi_X, grid_coords, levels, data_type='maldi')
    #subset_levels =  {k: v for k, v in sorted(levels.items()) if k >= 1}
    levels              = run_multiscale_spaco(levels, keigs=20, b=32, niter=10)
    plot_spaco(levels, components=0, invert_y=True)

import numpy as np

import matplotlib
try:
    matplotlib.use("TkAgg")
except Exception:
    pass

import matplotlib.pyplot as plt

from skimage import io as skio
from skimage.color import rgb2hed, rgba2rgb, gray2rgb
from skimage.transform import resize
from sklearn.decomposition import NMF


# -----------------------------
# Settings
# -----------------------------
he_path = "high_res_MSI/D2_10x_originalExport_cropped.tif"

n_components = 6
use_hed = True
downsample_shape = None   # Example: (800, 800), or None
random_state = 42
save_path = "he_nmf_components.png"


# -----------------------------
# Helpers
# -----------------------------
def normalize01(arr):
    arr = np.asarray(arr, dtype=np.float32)
    lo = np.nanmin(arr)
    hi = np.nanmax(arr)

    if hi > lo:
        return (arr - lo) / (hi - lo)

    return np.zeros_like(arr, dtype=np.float32)


def load_he_image(path, downsample_shape=None):
    img = skio.imread(path)

    if img.ndim == 2:
        img = gray2rgb(img)

    if img.ndim == 3 and img.shape[-1] == 4:
        img = rgba2rgb(img)

    img = img.astype(np.float32)

    if img.max() > 1:
        img /= 255.0

    img = np.clip(img, 0, 1)

    if downsample_shape is not None:
        img = resize(
            img,
            downsample_shape,
            preserve_range=True,
            anti_aliasing=True,
        ).astype(np.float32)

    return np.clip(img, 0, 1)


def nmf_he_components(
    he_path,
    n_components=3,
    use_hed=True,
    downsample_shape=None,
    random_state=42,
):
    img = load_he_image(he_path, downsample_shape=downsample_shape)
    H_img, W_img = img.shape[:2]

    if use_hed:
        feature_img = rgb2hed(img)

        # NMF requires nonnegative data.
        feature_img = feature_img.astype(np.float32)
        channel_mins = feature_img.reshape(-1, feature_img.shape[-1]).min(axis=0)
        feature_img = feature_img - channel_mins

        feature_name = "HED"
    else:
        feature_img = img.astype(np.float32)
        feature_name = "RGB"

    X = feature_img.reshape(-1, feature_img.shape[-1])
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X = np.maximum(X, 0)

    max_components = min(X.shape[0], X.shape[1])

    if n_components > max_components:
        print(
            f"Requested n_components={n_components}, but {feature_name} has only "
            f"{X.shape[1]} features. Using n_components={max_components}."
        )
        n_components = max_components

    model = NMF(
        n_components=n_components,
        init="nndsvda",
        max_iter=500,
        random_state=random_state,
    )

    W_mat = model.fit_transform(X)
    component_maps = W_mat.reshape(H_img, W_img, n_components)

    return img, component_maps, model.components_, feature_name


# -----------------------------
# Run NMF
# -----------------------------
img, component_maps, component_loadings, feature_name = nmf_he_components(
    he_path=he_path,
    n_components=n_components,
    use_hed=use_hed,
    downsample_shape=downsample_shape,
    random_state=random_state,
)


# -----------------------------
# Plot original + components
# -----------------------------
n_components_used = component_maps.shape[-1]

fig, axes = plt.subplots(
    1,
    n_components_used + 1,
    figsize=(4 * (n_components_used + 1), 4),
    constrained_layout=True,
)

if n_components_used + 1 == 1:
    axes = [axes]

axes[0].imshow(img)
axes[0].set_title("H&E image")
axes[0].axis("off")

for k in range(n_components_used):
    comp = normalize01(component_maps[..., k])

    axes[k + 1].imshow(comp, cmap="magma")
    axes[k + 1].set_title(f"NMF bin {k + 1}")
    axes[k + 1].axis("off")

fig.suptitle(f"H&E NMF components from {feature_name} features")

fig.savefig(save_path, dpi=200, bbox_inches="tight")
print(f"Saved plot to: {save_path}")
print("Matplotlib backend:", matplotlib.get_backend())

plt.show(block=True)


# -----------------------------
# Print loadings
# -----------------------------
print(f"NMF loadings from {feature_name} features:")
print(component_loadings)
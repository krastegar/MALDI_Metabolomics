# Biologically Constrained Structural-Field Registration of MALDI-MSI to H&E Histology

**Script:** `scripts/preprocessed_structuralmap.py`  
**Date:** June 2026

---

## Motivation

MALDI mass spectrometry imaging (MALDI-MSI) and hematoxylin & eosin (H&E) histology are complementary but physically incompatible modalities. MALDI-MSI measures molecular composition at each pixel — the acquired quantity is ion abundance as a function of mass-to-charge ratio (*m/z*) — while H&E measures light absorption by stained macromolecules. No reliable pointwise intensity correspondence exists between the two images, so classical intensity-based registration metrics (e.g., mutual information on raw pixel values) are ill-posed here.

The approach taken is to register **structural fields** — derived representations that encode tissue geometry rather than acquisition-specific intensities. The target of alignment is histological neighborhood correspondence: sinusoids, vessels, anisotropic fiber bundles, and the gross anatomical layout of the tissue section. Registration is therefore posed as an optimization over a biologically interpretable objective rather than a pixel-similarity surrogate.

---

## Pipeline Summary

The pipeline consists of six stages:

1. MALDI spectra → dense *m/z* bins → NMF decomposition → per-component denoising → structural fields
2. H&E → hematoxylin channel extraction (color deconvolution) → structural fields at MALDI resolution
3. Tissue support mask construction from structural evidence
4. Rigid registration by direct optimization of a composite biological objective
5. Diagnostic visualization and quality audit
6. Export of a MALDI → H&E pixel coordinate correspondence table

---

## Stage 1: MALDI Preprocessing

### 1.1 Dense *m/z* Binning

Each MALDI spectrum is a sparse vector over a continuous *m/z* axis. Before forming a spatial image, peaks must be aligned across pixels into a common feature axis. A DBSCAN-inspired 1D density criterion is used: a position $z_i$ in sorted *m/z* space is declared **dense** if $z_{i+k} - z_i < \delta$ for tolerance $\delta$ (default 0.042 Da) and lookahead $k$. Contiguous dense intervals define bins; bins with fewer than a minimum count of observations are discarded. This avoids the uniform-grid binning artefacts that arise when isotope patterns or adducts shift *m/z* centroids slightly across pixels.

The result is a data matrix $\mathbf{X} \in \mathbb{R}^{N \times B}_{\geq 0}$, where $N$ is the number of MALDI pixels and $B$ is the number of retained bins, with $X_{ij}$ the total ion current in bin $j$ at pixel $i$.

### 1.2 Non-negative Matrix Factorization (NMF)

$\mathbf{X}$ is factorized as

$$\mathbf{X} \approx \mathbf{W} \mathbf{H}, \qquad \mathbf{W} \in \mathbb{R}^{N \times K}_{\geq 0},\quad \mathbf{H} \in \mathbb{R}^{K \times B}_{\geq 0}$$

with rank $K$ (default 8) chosen to capture the dominant chemical tissue compartments. Non-negativity is both physically appropriate (ion abundances cannot be negative) and induces a **parts-based** representation: each component $k$ corresponds to a spatially localized chemical signature $\mathbf{H}_{k,\cdot}$ and a spatial loading image formed by reshaping $\mathbf{W}_{\cdot,k}$ onto the acquisition grid. The initialization uses NNDSVDA (non-negative double SVD with average fill), which empirically produces more interpretable components than random initialization.

### 1.3 Per-component Denoising

Each spatial loading image is denoised independently before structural field extraction. The default method is Gaussian smoothing with $\sigma = 0.75$ MALDI pixels, which suppresses salt-and-pepper noise introduced by shot noise in the ion detector while keeping gradients at structural boundaries intact. Alternatives include median filtering (disk structuring element) and non-local means (NLM), the latter estimating the noise level $\hat{\sigma}$ from the image and averaging patches weighted by their Euclidean distance in patch space.

---

## Stage 2: H&E Preprocessing

The H&E image is color-deconvolved using the Ruifrok–Johnston stain-separation model, which assumes that the optical density at each pixel is a linear superposition of the absorbance spectra of individual stains. The hematoxylin channel — the nuclear stain — is extracted because it carries the strongest structural signal for tissue architecture. The full-resolution hematoxylin image is then anti-aliasing downsampled to the MALDI acquisition grid so that both modalities share the same discrete domain during registration.

---

## Stage 3: Structural Fields

From each image (whether a MALDI NMF component or the H&E hematoxylin layer), four structural descriptors are computed:

**Edge magnitude.** The Sobel operator approximates $|\nabla f|$ via separable finite-difference kernels. The result highlights tissue–background and compartment boundaries.

**Orientation.** The local gradient direction $\theta = \arctan(g_y / g_x)$ is a circular quantity in $[0, \pi)$ (axial, not directional). Combining orientations across NMF components requires circular averaging at doubled angle: for a set of angles $\{\theta_k\}$ with scalar weights $\{w_k\}$,

$$\bar{\theta} = \frac{1}{2}\arctan\!\left(\frac{\sum_k w_k \sin 2\theta_k}{\sum_k w_k \cos 2\theta_k}\right)$$

which correctly handles the periodicity of undirected orientations.

**Anisotropy (coherence index).** The structure tensor at a point $\mathbf{x}$ is the outer-product integral $\mathbf{J}(\mathbf{x}) = G_\sigma * (\nabla f \nabla f^\top)$, smoothed by a Gaussian $G_\sigma$. Its eigenvalues $\lambda_1 \geq \lambda_2 \geq 0$ measure the gradient energy along the principal and orthogonal directions. The coherence index

$$C = \frac{\lambda_1 - \lambda_2}{\lambda_1 + \lambda_2 + \varepsilon}$$

is 1 where the gradient is perfectly unidirectional (oriented structures such as fiber bundles or vessel walls) and 0 where the gradient is isotropic (noise, uniform regions). It therefore provides a modality-agnostic measure of tissue anisotropy.

**Frangi vesselness.** The Frangi filter detects tubular (vessel-like) structures by analyzing the Hessian of the image at multiple scales $\sigma \in \{1, 2, 3\}$ pixels. For each scale, the Hessian eigenvalues $|\mu_1| \leq |\mu_2|$ enter two ratios: the **blob measure** $\mathcal{R}_B = |\mu_1|/|\mu_2|$, which distinguishes plate-like from tubular structures, and the **Frobenius norm** $\mathcal{S} = \|\mathbf{H}\|_F$, which suppresses background noise. The vesselness response is

$$\mathcal{V} = \exp\!\left(-\frac{\mathcal{R}_B^2}{2\beta^2}\right)\!\left(1 - \exp\!\left(-\frac{\mathcal{S}^2}{2\gamma^2}\right)\right)$$

with $\beta$ and $\gamma$ controlling sensitivity to blob asymmetry and second-order magnitude respectively. Both bright-ridge ($\mu_2 < 0$) and dark-ridge ($\mu_2 > 0$) responses are computed and the pointwise maximum retained, because vessels appear as bright lumen or dark walls depending on the modality and the NMF component polarity.

Per-component structural fields are combined across NMF components by taking the **pixelwise maximum**, which preserves the strongest structural evidence regardless of which chemical compartment expresses it, while avoiding the attenuation that a mean would introduce.

---

## Stage 4: Registration

### 4.1 Transform Space

Registration is restricted to a **rigid Euclidean (Euler) transform** in $\mathbb{R}^2$, parameterized by a rotation angle $\alpha$ and translation $(t_x, t_y)$. Scale and shear are held fixed at 1 and 0 respectively. This constraint is biologically justified: the tissue section undergoes at most minor physical distortions between sectioning and MALDI acquisition, so non-rigid deformation models would over-fit acquisition noise rather than correct biological misalignment.

### 4.2 Objective Function

Four terms form the composite objective $\mathcal{L}(\alpha, t_x, t_y)$:

$$\mathcal{L} = w_v \,\underbrace{\frac{d_C}{\delta_v}}_{\text{vessel Chamfer}} + w_s \,\underbrace{\bar{D}}_{\text{struct. diff.}} + w_o \,\underbrace{(1 - \text{IoU})}_{\text{overlap loss}} + w_r \,\underbrace{\|\mathbf{p}\|_{\text{norm}}}_{\text{reg. size}}$$

with weights $w_v = 1.0$, $w_s = 1.0$, $w_o = 1.5$, $w_r = 0.35$.

- **Vessel Chamfer distance** $d_C$. Vessel centers are detected from the vesselness field by thresholding and connected-component labeling, retaining components within area and eccentricity bounds. The bidirectional (symmetric) Chamfer distance between the MALDI vessel-center set $\mathcal{M}$ and H&E vessel-center set $\mathcal{H}$ under transform $T$ is $d_C = \tfrac{1}{2}(d(\mathcal{M} \xrightarrow{T} \mathcal{H}) + d(\mathcal{H} \xrightarrow{T^{-1}} \mathcal{M}))$, where each one-way term is the mean nearest-neighbor distance. Symmetry prevents the pathological case where a dense set of detections on one side makes a poor alignment appear acceptable.

- **Structural field difference** $\bar{D}$. The mean absolute pixelwise difference $|\hat{f}_{\text{MALDI}} - \hat{f}_{\text{H\&E}}|$ computed over the intersection of the two tissue support masks, where $\hat{f}$ denotes the selected structural field (edge by default) normalized to $[0,1]$.

- **Jaccard overlap loss.** The tissue support masks are binary images derived from a weighted combination of edge, anisotropy, and vesselness, thresholded at the 55th percentile. The intersection-over-union (IoU or Jaccard index) of the warped MALDI mask and the H&E mask is $J = |A \cap B| / |A \cup B|$; the loss term $1 - J$ penalizes transforms that displace tissue off the common support.

- **Transform size.** A soft regularizer $\|\mathbf{p}\|_{\text{norm}} = \sqrt{(\alpha / \alpha_{\max})^2 + (t_x / t_{\max})^2 + (t_y / t_{\max})^2}$ penalizes large transforms relative to the allowed bounds ($\pm 10°$, $\pm 30$ px), biasing the search toward the minimum necessary correction.

### 4.3 Optimization

The objective is non-convex and non-smooth (the Chamfer term involves discrete nearest-neighbor assignments). It is minimized in two stages. First, **differential evolution** (DE) globally samples the bounded three-dimensional parameter space using a population of candidate transforms that are evolved by mutation and crossover. DE makes no gradient assumptions and is robust to local minima introduced by the vessel-matching step. Second, the DE solution is refined by the **Powell method**, a derivative-free direction-set algorithm that converges quickly in smooth neighborhoods. The best of three candidates — identity transform, global DE solution, and Powell refinement — is selected by objective value.

---

## Output

The primary data product is `maldi_to_he_table.csv`, a per-spectrum correspondence table with columns `(maldi_x, maldi_y, he_x, he_y)` mapping each MALDI pixel to its registered location in full-resolution H&E pixel space. For analytic transforms the inverse is computed exactly; for non-analytic transforms a grid-based nearest-neighbor approximation is used. This table enables downstream co-localization of metabolite distributions with cellular features identified in H&E at single-cell resolution.

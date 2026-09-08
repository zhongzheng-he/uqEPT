<p align="center">
  <img src="assets/uqept-logo.svg" alt="uqEPT" width="340">
</p>

<p align="center">
  Uncertainty quantification for Helmholtz-based electrical properties tomography<br>
  under the local homogeneity assumption (LHA)
</p>

# uqEPT

`uqEPT` is a research-oriented Python toolkit for **uncertainty quantification
in Helmholtz-based electrical properties tomography (EPT)**. It supports
Laplacian and surface-integral formulations under the **local homogeneity
assumption (LHA)**, combining anatomically adaptive reconstruction with
voxel-wise uncertainty propagation and uncertainty-guided post-processing.

<p align="center">
  <img src="figure.png" alt="Laplacian-based uqEPT workflow: reconstruction, uncertainty propagation, and uncertainty-guided post-processing" width="730">
</p>

<p align="center">
  <em>Illustration of Laplacian-based EPT, uncertainty propagation, and uncertainty-guided post-processing.</em>
</p>

The current implementation supports:

- phase-based EPT for conductivity reconstruction;
- complex B-field-based EPT for conductivity and relative permittivity;
- Laplacian and surface-integral (SI) formulations;
- anatomically adaptive local polynomial fitting;
- voxel-wise standard-uncertainty maps;
- anatomical median filtering; and
- anatomical minimum-uncertainty weighted-mean filtering.

> **Research software:** this repository is under active development. The API
> and default parameters may change. Please validate parameters and outputs for
> your acquisition and application before quantitative interpretation.

## Model scope

Both the Laplacian and surface-integral formulations rely on the LHA: electrical
properties are assumed to be approximately constant over the effective local
reconstruction support. Surface integration changes how the Helmholtz-based
estimator is evaluated; it does **not** remove the LHA. Anatomical guidance helps
restrict the support to similar tissue regions, but does not guarantee that the
assumption holds at tissue boundaries or within heterogeneous regions.

## Reference

He Z, Lamy J, Arduino A, Zilberti L, Loureiro de Sousa P. Rigorous
Uncertainty Quantification for Helmholtz-based EPT: Application to
Uncertainty-Guided Post-processing. In: *2026 ISMRM-ISMRT Annual Meeting and
Exhibition*; Cape Town, South Africa; 2026.

[Download the conference abstract](https://hal.science/hal-05493748v1/file/Abstract%20%2300099.pdf)

If you use this code, please cite the reference above. A citation for the full
methodological paper will be added when available.

## Installation

Clone the development branch and install the required packages:

```bash
git clone --branch test https://github.com/zhongzheng-he/uqEPT.git
cd uqEPT
python -m pip install numpy scipy joblib tqdm numba connected-components-3d
```

Python 3.10 or newer is recommended. The repository can then be imported from
its root directory:

```python
from src import phase_based_surface_integral_EPT_with_uq
```

## Input conventions

All reconstruction inputs are three-dimensional NumPy arrays with identical
shapes.

| Input | Meaning | Convention |
|---|---|---|
| `PhiTR` | Transceive phase | Real-valued array in radians |
| `B` | Complex transmit-field surrogate | Complex-valued array |
| `Ref` | Anatomical guidance image or segmentation | Normalized internally when values exceed 1 |
| `ROI` | Reconstruction mask | Boolean array; defaults to `Ref > 0` |
| `h` | Voxel spacing | `[dx, dy, dz]` in metres |
| `omega` | Larmor angular frequency | Radians per second |

For standard complex Helmholtz-based EPT, a commonly used input is constructed
as

```python
B = np.abs(B1_plus) * np.exp(1j * PhiTR / 2.0)
```

where `PhiTR` is expressed in radians. For image-based EPT, `B` may instead be
a complex surrogate derived from an appropriate magnitude and phase image.
Such a surrogate introduces additional signal-model assumptions that must be
validated for the intended application.

## Quick start: surface-integral EPT

The SI formulation is implemented in two stages:

1. A local second-order polynomial fit estimates the field value and/or first
   spatial derivatives, together with their within-fit covariance.
2. These fitted quantities are aggregated over an integration domain using a
   volume-denominator surface-integral formulation.

The fitting kernel and integration kernel therefore have different roles and
are controlled separately. Their combined effective support remains subject to
the LHA described above.

### Phase-based SI-EPT

Phase-based SI-EPT reconstructs conductivity and its propagated standard
uncertainty:

```python
import numpy as np
from src import phase_based_surface_integral_EPT_with_uq

sigma, unc_sigma = phase_based_surface_integral_EPT_with_uq(
    PhiTR=transceive_phase,          # radians, real-valued 3D array
    Ref=reference_image,            # magnitude image or segmentation
    fit_kernel_size=[11, 11, 11],  # local polynomial-fitting support
    int_kernel_size=[15, 15, 15],  # surface-integral support
    fit_shape="cube",
    int_shape="cube",
    thresh=0.05,
    omega=2 * np.pi * 128e6,
    h=[1e-3, 1e-3, 1e-3],
    ROI=brain_mask,
    n_jobs=-1,
)
```

Outputs:

- `sigma`: conductivity in S/m;
- `unc_sigma`: propagated standard uncertainty of conductivity in S/m.

### Complex B-based SI-EPT

Complex SI-EPT reconstructs conductivity, relative permittivity, and their
propagated standard uncertainties:

```python
import numpy as np
from src import B1_based_surface_integral_EPT_with_uq

B = np.abs(B1_plus) * np.exp(1j * transceive_phase / 2.0)

sigma, epsilon_r, unc_sigma, unc_epsilon_r = (
    B1_based_surface_integral_EPT_with_uq(
        B=B,                         # complex-valued 3D field
        Ref=reference_image,
        fit_kernel_size=[11, 11, 11],
        int_kernel_size=[15, 15, 15],
        fit_shape="cube",
        int_shape="cube",
        thresh=0.05,
        omega=2 * np.pi * 128e6,
        h=[1e-3, 1e-3, 1e-3],
        ROI=brain_mask,
        n_jobs=-1,
    )
)
```

Outputs:

- `sigma`: conductivity in S/m;
- `epsilon_r`: dimensionless relative permittivity;
- `unc_sigma`: propagated standard uncertainty of conductivity in S/m;
- `unc_epsilon_r`: propagated standard uncertainty of relative permittivity.

Set `return_intermediates=True` to return a dictionary containing intermediate
SI quantities for method development and debugging. The dictionary keys differ
slightly between the phase-only and complex implementations; see the function
docstrings for the current contents.

## Laplacian EPT

The Laplacian formulation uses one anatomically adaptive polynomial-fitting
kernel.

### Phase-based Laplacian EPT

```python
from src import phase_based_Laplacian_EPT_with_uq

sigma, unc_sigma = phase_based_Laplacian_EPT_with_uq(
    PhiTR=transceive_phase,
    Ref=reference_image,
    kernel_size=[11, 11, 11],
    shape="cube",
    thresh=0.05,
    omega=2 * np.pi * 128e6,
    h=[1e-3, 1e-3, 1e-3],
    ROI=brain_mask,
    n_jobs=-1,
)
```

### Complex B-based Laplacian EPT

```python
from src import B1_based_Laplacian_EPT_with_uq

sigma, epsilon_r, unc_sigma, unc_epsilon_r = (
    B1_based_Laplacian_EPT_with_uq(
        B=B,
        Ref=reference_image,
        kernel_size=[11, 11, 11],
        shape="cube",
        thresh=0.05,
        omega=2 * np.pi * 128e6,
        h=[1e-3, 1e-3, 1e-3],
        ROI=brain_mask,
        n_jobs=-1,
    )
)
```

## Uncertainty-guided post-processing

The proposed post-processing method first restricts the local neighborhood
using the anatomical reference. It then retains the 25% of valid neighboring
voxels with the lowest uncertainty and computes an inverse-variance-weighted
mean from that subset.

```python
from src import anatomical_min_uncertainty_weighted_mean_filter

sigma_proposed = anatomical_min_uncertainty_weighted_mean_filter(
    Im=sigma,
    Ref=reference_image,
    uncertainty=unc_sigma,           # standard-uncertainty map
    kernel_size=[21, 21, 21],
    shape="cube",
    thresh=0.05,
    ROI=brain_mask,
    n_jobs=-1,
)
```

For comparison, the anatomically adaptive median filter is called as follows:

```python
from src import anatomical_median_filter

sigma_median = anatomical_median_filter(
    Im=sigma,
    Ref=reference_image,
    kernel_size=[21, 21, 21],
    shape="cube",
    thresh=0.05,
    ROI=brain_mask,
    n_jobs=-1,
)
```

The same post-processing functions can be applied to relative permittivity by
using `epsilon_r` and `unc_epsilon_r` as inputs.

## Choosing kernels and anatomical thresholds

- Every kernel dimension must be an odd integer.
- The second-order polynomial fit requires more than 10 included voxels for a
  finite residual-based covariance estimate.
- `fit_kernel_size` controls derivative estimation and the local bias-variance
  trade-off.
- `int_kernel_size` controls SI aggregation and its effective spatial support.
- Larger kernels generally suppress random fluctuations but increase boundary
  mixing and may reduce the recovery of small structures.
- Smaller kernels better preserve boundaries and the local-homogeneity
  assumption, but are more sensitive to noise.
- `thresh` is applied after internal normalization of `Ref` and should be
  selected for the contrast and scaling of the reference image.

Kernel sizes should be reported together with voxel spacing. They should be
validated for the SNR, anatomy, EPT formulation, and target structure rather
than treated as universal defaults.

## Interpretation of uncertainty

The returned uncertainty maps are propagated standard uncertainties under the
implemented local residual and covariance model. They should not be interpreted
as ground-truth reconstruction errors.

In particular:

- uncertainty can reflect measurement noise, fitting residuals, and numerical
  instability;
- smooth systematic bias may not produce large residuals and can therefore be
  underestimated;
- the code applies a biophysical-range penalty to implausible reconstructed
  values;
- in SI-EPT, within-fit covariance is propagated, whereas cross-covariance
  between neighboring overlapping polynomial fits is currently neglected; and
- quantitative calibration requires validation with repeated noise realizations
  or repeated measurements, normalized errors, and interval coverage.

Strong uncertainty-error correlation supports relative spatial reliability
ranking, but does not by itself establish calibration of uncertainty magnitude.

## Example notebook

Install the additional notebook and plotting dependencies, then launch Jupyter
from the repository root:

```bash
python -m pip install matplotlib jupyterlab
python -m jupyterlab
```

See [`examples/quickstart.ipynb`](examples/quickstart.ipynb) for an end-to-end
template covering input validation, phase-based and complex SI-EPT,
uncertainty-guided filtering, visualization, and saving outputs. Replace the
placeholder `.npy` paths with your own co-registered 3D arrays before running
the reconstruction cells.

## Notes on runtime and memory

Three-dimensional reconstruction can be computationally demanding, especially
for large fitting and integration kernels. Use `n_jobs` to control parallelism.
Using every available CPU core (`n_jobs=-1`) may require substantial memory; a
smaller value is often preferable on shared systems.

The first execution may include Numba compilation overhead.

## Repository structure

```text
uqEPT/
├── assets/
│   └── uqept-logo.svg
├── examples/
│   └── quickstart.ipynb
├── src/
│   ├── phase_based_Laplacian_EPT_with_uq.py
│   ├── B1_based_Laplacian_EPT_with_uq.py
│   ├── phase_based_surface_integral_EPT_with_uq.py
│   ├── B1_based_surface_integral_EPT_with_uq.py
│   ├── anatomical_median_filter.py
│   └── anatomical_min_uncertainty_weighted_mean_filter.py
├── figure.png
├── LICENSE
└── README.md
```

## License

See [`LICENSE`](LICENSE).

## Contact

Zhongzheng He, PhD<br>
ICube, Université de Strasbourg, Strasbourg, France<br>
zhongzheng.he@unistra.fr

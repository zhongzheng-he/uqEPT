# ///////////////////////////////////////////////////////////////////////////////////////////////
# // Zhongzheng He, PhD, ICube, Université de Strasbourg, Strasbourg, France
# // Contact: zhongzheng.he@unistra.fr
# ///////////////////////////////////////////////////////////////////////////////////////////////
import os
import time
from typing import Tuple
import numpy as np
from numpy.linalg import pinv
from joblib import Parallel, delayed
from tqdm import tqdm
import cc3d
from numba import njit

def phase_based_surface_integral_EPT_with_uq(PhiTR,Ref,fit_kernel_size=[11, 11, 11], int_kernel_size=[11,11,11],fit_shape="cube",int_shape="cube",thresh=0.05,omega = 128e6*2*np.pi, 
                                             h=None,ROI=None, n_jobs=-1,return_intermediates=False):
    """
    Phase-only surface-integral EPT with uncertainty quantification.
    Conductivity is estimated from the transceive phase as

        sigma_SI = S_PhiTR / (2 * mu0 * omega * V_eff)

    with

        S_phi = sum_p (wx_p*c1_p + wy_p*c3_p + wz_p*c6_p)

    For each voxel p, q = [c1, c3, c6] is obtained from a local second-order
    polynomial fit to the real-valued phase. The within-voxel covariance of q is
    retained. Cross-covariances between different voxel-wise fits are neglected.

    Parameters
    ----------
    PhiTR : numpy.ndarray
        3D array of the transceive phase image in radians.
    Ref : numpy.ndarray
        3D reference image for anatomical guidance, such as a magnitude image
        or a tissue segmentation map.
    fit_kernel_size : list of int, optional
        Dimensions [kx, ky, kz] of the 2nd order polynomial fitting kernel. All
        values must be odd. Default is [11, 11, 11].
    int_kernel_size : list of int, optional
        Dimensions [kx, ky, kz] of the surface integral kernel. All
        values must be odd. Default is [11, 11, 11].
    fit_shape : {'cube', 'ellipse', 'cross'}, optional
        The base shape of the 2nd order polynomial fitting kernel before anatomical adaptation.
        Default is 'cube'.
    int_shape : {'cube', 'ellipse', 'cross'}, optional
        The base shape of the surface integral kernel before anatomical adaptation.
        Default is 'cube'. 
    thresh : float, optional
        Threshold for anatomical adaptation (0 to 1). Only voxels in the
        `Ref` image with a relative intensity difference below this threshold
        (compared to the kernel's central voxel) are included in the fit.
        Default is 0.05.
    omega : float, optional
        Larmor frequency in rad/s (i.e., 2 * pi * frequency).
        Default is 128e6 * 2 * np.pi, corresponding to a 3T scanner.
    h : list of float, optional
        Voxel spacing [dx, dy, dz] in meters. If None, assumes an isotropic
        voxel size of 1 mm. Default is None.
    ROI : numpy.ndarray, optional
        3D binary mask defining the Region of Interest. If None, the ROI is
        automatically generated from non-zero voxels in the `Ref` image.
        Default is None.
    n_jobs : int, optional
        Number of CPU cores to use for parallel processing.
        -1 means using all available cores. Default is -1.

    Returns
    -------
    sigma : numpy.ndarray
        The reconstructed 3D electrical conductivity map in Siemens/meter (S/m).
    unc_sigma : numpy.ndarray
        A 3D map of the standard deviation of the reconstructed conductivity,
        representing the voxel-wise uncertainty of the estimation.

    """

    start_time = time.time()

    # ------------------------------------------------------------------------------------------
    # Input verification
    # ------------------------------------------------------------------------------------------
    if np.iscomplexobj(PhiTR):
        raise ValueError("PhiTR must be real-valued for the phase-only version.")

    PhiTR = np.asarray(PhiTR, dtype=np.float64)
    Ref = np.asarray(Ref, dtype=np.float64)

    if PhiTR.ndim != 3 or Ref.ndim != 3:
        raise ValueError("PhiTR and Ref must be 3D arrays.")
    if PhiTR.shape != Ref.shape:
        raise ValueError("PhiTR and Ref must have the same shape.")

    if any(k < 3 or k % 2 == 0 for k in fit_kernel_size):
        raise ValueError("fit_kernel_size must contain only odd integers >= 3.")
    if any(k < 3 or k % 2 == 0 for k in int_kernel_size):
        raise ValueError("int_kernel_size must contain only odd integers >= 3.")

    ROI = Ref > 0 if ROI is None else np.asarray(ROI).astype(bool)
    
    if ROI.shape != PhiTR.shape:
        raise ValueError("ROI must have the same shape as PhiTR.")
    if not np.any(ROI):
        raise ValueError("ROI is empty.")

    # Normalize reference image inside ROI.
    if np.max(Ref[ROI]) > 1:
        Ref = imrescale(Ref, ROI=ROI)

    cpu_cores = os.cpu_count()
    if n_jobs == -1:
        n_jobs = cpu_cores
        print(f"Number of CPU cores available and used: {cpu_cores}")
    else:
        print(f"Number of CPU cores available and used: {n_jobs}")

    # ------------------------------------------------------------------------------------------
    # Constants
    # ------------------------------------------------------------------------------------------
    mu0 = 4 * np.pi * 1e-7
    phase_scale_const = 1.0 / (2.0 * mu0 * omega)

    dx, dy, dz = h if h is not None else (1e-3, 1e-3, 1e-3)

    # ==========================================================================================
    # Stage 1/2: local real phase polynomial fitting and q covariance
    # ==========================================================================================
    kx, ky, kz = fit_kernel_size
    kx_radii, ky_radii, kz_radii = (kx - 1) // 2, (ky - 1) // 2, (kz - 1) // 2

    pad_fit = ((kx_radii, kx_radii), (ky_radii, ky_radii), (kz_radii, kz_radii))
    PhiTR_fit = np.pad(PhiTR, pad_fit, mode="constant")
    Ref_fit = np.pad(Ref, pad_fit, mode="constant")
    ROI_fit = np.pad(ROI, pad_fit, mode="constant")

    Shape_fit = make_kernel_shape(fit_kernel_size, fit_shape)
    n_shape_fit = np.count_nonzero(Shape_fit)
    if n_shape_fit <= 10:
        raise ValueError(
            "Uncertainty quantification requires more than 10 voxels in the "
            "second-order fitting kernel because dof = n_voxels - 10 must be > 0. "
            "Increase fit_kernel_size or use fit_shape='cube'."
        )

    # Create the 3D second-order polynomial design matrix.
    x, y, z = np.meshgrid(np.arange(kx), np.arange(ky), np.arange(kz), indexing="ij")
    x = (x - kx_radii) * dx
    y = (y - ky_radii) * dy
    z = (z - kz_radii) * dz

    x = x.ravel()
    y = y.ravel()
    z = z.ravel()

    F_fit = np.column_stack([np.ones_like(x),x, x ** 2,y, x * y,y ** 2,z,x * z,y * z,z ** 2])

    ind0_fit = np.where(Shape_fit.ravel())[0]
    F0_fit = F_fit[ind0_fit, :]
    F0_fit_pinv = pinv(F0_fit)
    M0_fit = pinv(F0_fit.T @ F0_fit)

    Indices_fit = np.where(ROI_fit)
    Indx_fit, Indy_fit, Indz_fit = Indices_fit[0], Indices_fit[1], Indices_fit[2]

    def process_fit_patch(j):
        """Fit q=[c1,c3,c6] and its 3x3 covariance for one voxel."""
        ix, iy, iz = Indx_fit[j], Indy_fit[j], Indz_fit[j]

        indx = slice(ix - kx_radii, ix + kx_radii + 1)
        indy = slice(iy - ky_radii, iy + ky_radii + 1)
        indz = slice(iz - kz_radii, iz + kz_radii + 1)

        PhiTR_patch = PhiTR_fit[indx, indy, indz]
        Ref_patch = Ref_fit[indx, indy, indz]

        Shape_patch = np.abs(Ref_patch - Ref_patch[kx_radii, ky_radii, kz_radii]) <= thresh
        Shape_patch &= Shape_fit

        # Avoid fitting with padded/background voxels.
        ROI_patch = ROI_fit[indx, indy, indz]
        Shape_patch &= ROI_patch

        # Check if all elements are in shape, i.e. Shape_patch == Shape_fit.
        if np.array_equal(Shape_patch, Shape_fit):
            PhiTR_patch_in_shape = PhiTR_patch.ravel()[ind0_fit]
            C = F0_fit_pinv @ PhiTR_patch_in_shape
            return calculate_phase_q_and_covariance_from_C(F0_fit, M0_fit, PhiTR_patch_in_shape, C)

        # Removing the non-connected components; cc3d is compiled with C++.
        labeled = cc3d.connected_components(Shape_patch, connectivity=6)
        center_label = labeled[kx_radii, ky_radii, kz_radii]
        Shape_patch = labeled == center_label

        ind = np.where(Shape_patch.ravel())[0]
        PhiTR_patch_in_shape = PhiTR_patch.ravel()[ind]
        F_adap = F_fit[ind, :]
        F_adap_pinv = pinv(F_adap)
        M_adap = pinv(F_adap.T @ F_adap)
        C = F_adap_pinv @ PhiTR_patch_in_shape

        return calculate_phase_q_and_covariance_from_C(F_adap, M_adap, PhiTR_patch_in_shape, C)

    print("Stage 1/2: fitting q=[c1,c3,c6] and covariance")
    fit_results = Parallel(n_jobs=n_jobs)(
        delayed(process_fit_patch)(j)
        for j in tqdm(range(len(Indx_fit)), desc="Fitting phase q", ncols=100, position=0, leave=True)
    )

    q_map = np.zeros(PhiTR_fit.shape + (3,), dtype=np.float64)
    cov_q_map = np.zeros(PhiTR_fit.shape + (3, 3), dtype=np.float64)

    temp_q, temp_cov_q = zip(*fit_results)
    q_map[Indx_fit, Indy_fit, Indz_fit, :] = temp_q
    cov_q_map[Indx_fit, Indy_fit, Indz_fit, :, :] = temp_cov_q

    crop_fit = (
        slice(kx_radii, -kx_radii),
        slice(ky_radii, -ky_radii),
        slice(kz_radii, -kz_radii),
    )
    q_map = q_map[crop_fit + (slice(None),)]
    cov_q_map = cov_q_map[crop_fit + (slice(None), slice(None))]
    valid_fit_map = np.all(np.isfinite(cov_q_map), axis=(-2, -1))
    valid_fit_map &= ROI

    # ==========================================================================================
    # Stage 2/2: phase surface integral and uncertainty propagation
    # ==========================================================================================
    ikx, iky, ikz = int_kernel_size
    ikx_radii, iky_radii, ikz_radii = (ikx - 1) // 2, (iky - 1) // 2, (ikz - 1) // 2

    pad_int_3d = ((ikx_radii, ikx_radii), (iky_radii, iky_radii), (ikz_radii, ikz_radii))
    q_int = np.pad(q_map, pad_int_3d + ((0, 0),), mode="constant")
    cov_q_int = np.pad(cov_q_map, pad_int_3d + ((0, 0), (0, 0)), mode="constant")
    valid_fit = np.pad(valid_fit_map, pad_int_3d, mode="constant")
    Ref_int = np.pad(Ref, pad_int_3d, mode="constant")
    ROI_int = np.pad(ROI, pad_int_3d, mode="constant")

    Shape_int = make_kernel_shape(int_kernel_size, int_shape)

    # Create effective-volume and signed-surface kernels matching the SI geometry.
    dsx = dy * dz
    dsy = dx * dz
    dsz = dx * dy
    dv = dx * dy * dz

    KV = np.zeros((3, 3, 3), dtype=np.float64)
    KV[[0, 2], 1, 1] = dsx
    KV[1, [0, 2], 1] = dsy
    KV[1, 1, [0, 2]] = dsz
    KV = KV / np.sum(KV) * dv

    KSx = np.zeros((3, 3, 3), dtype=np.float64)
    KSy = np.zeros((3, 3, 3), dtype=np.float64)
    KSz = np.zeros((3, 3, 3), dtype=np.float64)

    KSx[:, 1, 1] = np.array([-1.0, 0.0, 1.0]) * dsx
    KSy[1, :, 1] = np.array([-1.0, 0.0, 1.0]) * dsy
    KSz[1, 1, :] = np.array([-1.0, 0.0, 1.0]) * dsz

    Indices_int = np.where(ROI_int)
    Indx_int, Indy_int, Indz_int = Indices_int[0], Indices_int[1], Indices_int[2]

    def process_integral_patch(j):
        """Compute phase-only SI-EPT and voxel-wise uncertainty."""
        ix, iy, iz = Indx_int[j], Indy_int[j], Indz_int[j]

        indx = slice(ix - ikx_radii, ix + ikx_radii + 1)
        indy = slice(iy - iky_radii, iy + iky_radii + 1)
        indz = slice(iz - ikz_radii, iz + ikz_radii + 1)

        q_patch = q_int[indx, indy, indz, :]
        cov_q_patch = cov_q_int[indx, indy, indz, :, :]

        Ref_patch = Ref_int[indx, indy, indz]
        valid_fit_patch = valid_fit[indx, indy, indz]

        Shape_patch = np.abs(Ref_patch - Ref_patch[ikx_radii, iky_radii, ikz_radii]) <= thresh
        Shape_patch &= Shape_int

        # Keep the same conservative 6-connected center component style as the
        # refined complex-B implementation.
        if not np.array_equal(Shape_patch, Shape_int):
            labeled = cc3d.connected_components(Shape_patch, connectivity=6)
            center_label = labeled[ikx_radii, iky_radii, ikz_radii]
            Shape_patch = labeled == center_label

        return calculate_surface_integral_phase_and_uncertainty(q_patch,cov_q_patch,Shape_patch,valid_fit_patch,KV,KSx,KSy,KSz,phase_scale_const,ikx,iky,ikz,ikx_radii,iky_radii,ikz_radii)

    print("Stage 2/2: phase surface integral + UQ")
    si_results = Parallel(n_jobs=n_jobs)(
        delayed(process_integral_patch)(j)
        for j in tqdm(range(len(Indx_int)), desc="Phase SI-EPT + UQ", ncols=100, position=0, leave=True)
    )

    temp_sigma, temp_unc_sigma, temp_S, temp_V = zip(*si_results)

    # Allocate padded output maps and assign with Indices_int, exactly as in Stage 1.
    sigma = np.zeros(ROI_int.shape, dtype=np.float64)
    unc_sigma = np.zeros(ROI_int.shape, dtype=np.float64)
    S = np.zeros(ROI_int.shape, dtype=np.float64)
    V = np.zeros(ROI_int.shape, dtype=np.float64)

    sigma[Indx_int, Indy_int, Indz_int] = temp_sigma
    unc_sigma[Indx_int, Indy_int, Indz_int] = temp_unc_sigma
    S[Indx_int, Indy_int, Indz_int] = temp_S
    V[Indx_int, Indy_int, Indz_int] = temp_V

    crop_int = (
        slice(ikx_radii, -ikx_radii),
        slice(iky_radii, -iky_radii),
        slice(ikz_radii, -ikz_radii),
    )

    sigma = sigma[crop_int]
    unc_sigma = unc_sigma[crop_int]
    S = S[crop_int]
    V = V[crop_int]

    sigma[~np.isfinite(sigma)] = 0.0
    unc_sigma[~np.isfinite(unc_sigma)] = np.inf

    unc_sigma = unc_penalization(sigma, unc_sigma, Rmin=0, Rmax=2.5)

    print(f"Elapsed time: {time.time() - start_time:.2f} seconds")

    if return_intermediates:
        return {
            "sigma": sigma,
            "unc_sigma": unc_sigma,
            "S": S,
            "V": V,
            "q": q_map,
            "cov_q": cov_q_map
        }

    return sigma, unc_sigma


@njit(cache=True)
def calculate_phase_q_and_covariance_from_C(F_adap, M_adap, PhiTR_patch_in_shape, C):
    """
    Estimate q=[c1,c3,c6] and its 3x3 covariance from a precomputed C.

    If dof <= 0, q is still returned, but cov_q is set to inf because the
    residual variance cannot be statistically estimated.
    """
    n, m = F_adap.shape
    dof = n - m

    q = np.empty(3, dtype=np.float64)
    cov_q = np.empty((3, 3), dtype=np.float64)

    q[0] = C[1]
    q[1] = C[3]
    q[2] = C[6]

    if dof <= 0:
        cov_q[:, :] = np.inf
        return q, cov_q

    PhiTR_patch_fitted = F_adap @ C
    residual = PhiTR_patch_in_shape - PhiTR_patch_fitted
    sigma2 = np.sum(residual ** 2) / dof

    idx0 = 1
    idx1 = 3
    idx2 = 6

    cov_q[0, 0] = sigma2 * M_adap[idx0, idx0]
    cov_q[0, 1] = sigma2 * M_adap[idx0, idx1]
    cov_q[0, 2] = sigma2 * M_adap[idx0, idx2]

    cov_q[1, 0] = sigma2 * M_adap[idx1, idx0]
    cov_q[1, 1] = sigma2 * M_adap[idx1, idx1]
    cov_q[1, 2] = sigma2 * M_adap[idx1, idx2]

    cov_q[2, 0] = sigma2 * M_adap[idx2, idx0]
    cov_q[2, 1] = sigma2 * M_adap[idx2, idx1]
    cov_q[2, 2] = sigma2 * M_adap[idx2, idx2]

    return q, cov_q


@njit(cache=True)
def _conv3_same_at(mask, kernel, i, j, k):
    """3D MATLAB-like convn(mask, kernel, 'same') at one voxel, with flipped kernel."""
    nx, ny, nz = mask.shape
    out = 0.0

    for a in range(3):
        ii = i + a - 1
        if ii < 0 or ii >= nx:
            continue

        for b in range(3):
            jj = j + b - 1
            if jj < 0 or jj >= ny:
                continue

            for c in range(3):
                kk = k + c - 1
                if kk < 0 or kk >= nz:
                    continue

                if mask[ii, jj, kk]:
                    out += kernel[2 - a, 2 - b, 2 - c]

    return out


@njit(cache=True)
def calculate_surface_integral_phase_and_uncertainty(q_patch,cov_q_patch,Shape_patch, valid_fit_patch, KV,KSx,KSy,KSz,phase_scale_const,ikx,iky,ikz,ikx_radii,iky_radii,ikz_radii):
    """Numba-accelerated phase-only SI-EPT and UQ calculation."""
    S = 0.0
    V = 0.0

    # First pass: compute S and Veff.
    for i in range(ikx):
        for j in range(iky):
            for k in range(ikz):
                if Shape_patch[i, j, k]:
                    vw = _conv3_same_at(Shape_patch, KV, i, j, k)
                    wx = _conv3_same_at(Shape_patch, KSx, i, j, k)
                    wy = _conv3_same_at(Shape_patch, KSy, i, j, k)
                    wz = _conv3_same_at(Shape_patch, KSz, i, j, k)

                    S += wx * q_patch[i, j, k, 0]
                    S += wy * q_patch[i, j, k, 1]
                    S += wz * q_patch[i, j, k, 2]
                    V += vw

    if np.abs(V) == 0.0:
        return np.nan, np.inf, S, V

    scale = phase_scale_const / V
    sigma = scale * S

    # If the center voxel has no valid covariance, keep the mean but set UQ to inf.
    if not valid_fit_patch[ikx_radii, iky_radii, ikz_radii]:
        return sigma, np.inf, S, V

    # Include only voxels with finite q covariance for the uncertainty propagation.
    Shape_patch &= valid_fit_patch

    var_sigma = 0.0

    # Second pass: uncertainty propagation, neglecting cross-voxel covariance.
    for i in range(ikx):
        for j in range(iky):
            for k in range(ikz):
                if Shape_patch[i, j, k]:
                    wx = _conv3_same_at(Shape_patch, KSx, i, j, k)
                    wy = _conv3_same_at(Shape_patch, KSy, i, j, k)
                    wz = _conv3_same_at(Shape_patch, KSz, i, j, k)

                    d0 = scale * wx
                    d1 = scale * wy
                    d2 = scale * wz

                    var_sigma += d0 * cov_q_patch[i, j, k, 0, 0] * d0
                    var_sigma += d0 * cov_q_patch[i, j, k, 0, 1] * d1
                    var_sigma += d0 * cov_q_patch[i, j, k, 0, 2] * d2

                    var_sigma += d1 * cov_q_patch[i, j, k, 1, 0] * d0
                    var_sigma += d1 * cov_q_patch[i, j, k, 1, 1] * d1
                    var_sigma += d1 * cov_q_patch[i, j, k, 1, 2] * d2

                    var_sigma += d2 * cov_q_patch[i, j, k, 2, 0] * d0
                    var_sigma += d2 * cov_q_patch[i, j, k, 2, 1] * d1
                    var_sigma += d2 * cov_q_patch[i, j, k, 2, 2] * d2

    var_sigma = max(var_sigma, 0.0)
    unc_sigma = np.sqrt(var_sigma)

    return sigma, unc_sigma, S, V


def make_kernel_shape(kernel_size: Tuple[int, int, int], shape: str):
    """Create cube, ellipse, or cross base kernel shape."""
    kx, ky, kz = kernel_size
    kx_radii, ky_radii, kz_radii = (kx - 1) // 2, (ky - 1) // 2, (kz - 1) // 2

    x, y, z = np.meshgrid(np.arange(kx), np.arange(ky), np.arange(kz), indexing="ij")
    x = x - kx_radii
    y = y - ky_radii
    z = z - kz_radii

    if shape == "cube":
        Shape = np.ones((kx, ky, kz), dtype=bool)
    elif shape == "cross":
        Shape = np.zeros((kx, ky, kz), dtype=bool)
        Shape[kx_radii, :, kz_radii] = True
        Shape[kx_radii, ky_radii, :] = True
        Shape[:, ky_radii, kz_radii] = True
    elif shape == "ellipse":
        if kx_radii == 0 or ky_radii == 0 or kz_radii == 0:
            raise ValueError("ellipse shape requires kernel size >= 3 in all directions.")
        Shape = (x / kx_radii) ** 2 + (y / ky_radii) ** 2 + (z / kz_radii) ** 2 <= 1.0
    else:
        raise ValueError('Please specify shape = "ellipse", "cube", or "cross".')

    return Shape


def unc_penalization(mean, unc, Rmin=0, Rmax=2.5, k=1.0):
    """
    Corrects the uncertainty for non-biophysical results based on a coverage interval.

    A penalty is applied only if the k-sigma confidence interval of a measurement
    does not overlap with the plausible biophysical range [Rmin, Rmax]. The
    uncertainty is then increased to the minimum value required for the interval
    to touch the edge of the plausible range.

    Args:
        mean (np.ndarray): The map of estimated mean values (e.g., conductivity).
        unc (np.ndarray): The map of estimated uncertainties (std. dev.).
        Rmin (float): The minimum plausible biophysical value.
        Rmax (float): The maximum plausible biophysical value.
        k : The factor for the confidence interval (e.g., 1.0 for ~68%, 2.0 for ~98%).

    Returns:
        np.ndarray: The corrected uncertainty map.
    """
    # Create a copy to avoid modifying the original array in place
    corrected_unc = unc.copy()
    
    # --- Case 1: Value is TOO HIGH ---
    # Condition: The lower bound of the confidence interval is above the max plausible value.
    idx_high = (mean - k * corrected_unc) > Rmax
    
    # Correction: Set uncertainty to the distance from the mean to the max plausible value.
    # Using abs() makes it robust against any edge cases.
    corrected_unc[idx_high] = np.abs(mean[idx_high] - Rmax) / k

    # --- Case 2: Value is TOO LOW ---
    # Condition: The upper bound of the confidence interval is below the min plausible value.
    idx_low = (mean + k * corrected_unc) < Rmin
    
    # Correction: Set uncertainty to the distance from the mean to the min plausible value.
    corrected_unc[idx_low] = np.abs(Rmin - mean[idx_low]) / k

    return corrected_unc



def imrescale(image, new_min=0, new_max=1,ROI=None, window=(0.005, 0.995)): # Intensity normalization to [0,1]
    """
    Intensity normalization with optional ROI and percentile intensity window.

    Parameters
    ----------
    image : np.ndarray
        Input image.
    new_min, new_max : float
        Output intensity range.
    ROI : np.ndarray or None
        Optional mask. If provided, percentiles are estimated inside ROI.
    window : tuple(float, float)
        Quantile window, e.g. (0.005, 0.995) for 0.5%–99.5%.

    Returns
    -------
    np.ndarray
        Rescaled image.
    """
        
    image = np.asarray(image, dtype=np.float64)

    if ROI is not None:
        ROI = ROI > 0
        vals = image[ROI]
    else:
        vals = image.ravel()

    if vals.size == 0:
        raise ValueError("ROI is empty.")

    old_min, old_max = np.quantile(vals, window)

    if old_max == old_min:
        return np.full_like(image, new_min, dtype=np.float64)

    image_clipped = np.clip(image, old_min, old_max)

    return new_min + (image_clipped - old_min) * (new_max - new_min) / (old_max - old_min)
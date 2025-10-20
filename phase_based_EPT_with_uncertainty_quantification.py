# ///////////////////////////////////////////////////////////////////////////////////////////////
# // Zhongzheng He, PhD, ICube, Université de Strasbourg, Strasbourg, France
# // Contact: zhongzheng.he@unistra.fr
# ///////////////////////////////////////////////////////////////////////////////////////////////


import numpy as np
from numpy.linalg import pinv
from joblib import Parallel, delayed
from tqdm import tqdm  # For progress bar
import time
import os
import cc3d
from scipy.optimize import minimize
from numba import njit


def Phase_based_EPT_With_Uncertainty_Quantification(PhiTR, Ref, kernel_size=[5,5,5], shape="cube", thresh=0.1, omega=128e6*2*np.pi, h=None, ROI=None, n_jobs=-1):
    """
    Reconstructs electrical conductivity and its uncertainty using phase-based EPT.

    This function implements the Laplacian version of phase-based Electrical
    Properties Tomography (EPT), where conductivity is derived from the
    transceive phase (PhiTR) using the relation:
    sigma = Laplacian(PhiTR) / (2 * mu0 * omega).

    The Laplacian is estimated at each voxel using a 2nd order Savitzky-Golay
    filter. A key feature is the anatomically-adaptive kernel, which conforms
    to the local tissue structure. The kernel is shaped by including only
    neighboring voxels from the reference image (`Ref`) that have a similar
    intensity to the central voxel, defined by `thresh`. This adaptation
    minimizes filtering across different tissue boundaries, improving accuracy.

    Furthermore, the function quantifies the voxel-wise uncertainty of the
    conductivity reconstruction. This is achieved by propagating the statistical
    error from the Savitzky-Golay polynomial fit to the final conductivity
    value, providing a standard deviation map that serves as a confidence
    metric for the reconstruction.

    The computation is parallelized using Joblib to accelerate processing on
    multi-core systems.

    Parameters
    ----------
    PhiTR : numpy.ndarray
        3D array of the transceive phase image in radians.
    Ref : numpy.ndarray
        3D reference image for anatomical guidance, such as a magnitude image
        or a tissue segmentation map.
    kernel_size : list of int, optional
        Dimensions [kx, ky, kz] of the Savitzky-Golay filter kernel. All
        values must be odd. Default is [5, 5, 5].
    shape : {'cube', 'ellipse', 'cross'}, optional
        The base shape of the kernel before anatomical adaptation.
        Default is 'cube'.
    thresh : float, optional
        Threshold for anatomical adaptation (0 to 1). Only voxels in the
        `Ref` image with a relative intensity difference below this threshold
        (compared to the kernel's central voxel) are included in the fit.
        Default is 0.1.
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
    mu0       =  4*np.pi*1E-7
    coeff = 2*mu0*omega

    cpu_cores = os.cpu_count()
    if n_jobs==-1:
        n_jobs=cpu_cores
        print(f"Number of CPU cores available and used: {cpu_cores}")
    else:
        print(f"Number of CPU cores available and used: {n_jobs}")
        
    # Initial setup
    dx, dy, dz = h if h is not None else (1e-3, 1e-3, 1e-3)
    kx, ky, kz = kernel_size
    if (kx%2 ==0) | (ky%2 ==0) | (kz%2 ==0):
        raise ValueError('the kernel size [kx,ky,kz] should be odd')
        
    kx_radii, ky_radii, kz_radii = (kx - 1) // 2, (ky - 1) // 2, (kz - 1) // 2  

     # Pad arrays
    Ref = np.pad(Ref, ((kx_radii, kx_radii), (ky_radii, ky_radii), (kz_radii, kz_radii)), mode='constant')
    PhiTR  = np.pad(PhiTR, ((kx_radii, kx_radii), (ky_radii, ky_radii), (kz_radii, kz_radii)), mode='constant')
    ROI = Ref > 0 if ROI is None else np.pad(ROI, ((kx_radii, kx_radii), (ky_radii, ky_radii), (kz_radii, kz_radii)), mode='constant')

    #normalisation of reference image
    if np.max(np.ravel(Ref)) > 1 :
        Ref= imrescale(Ref,ROI=ROI)
        
   
    x, y, z = np.meshgrid(np.arange(kx), np.arange(ky), np.arange(kz), indexing='ij')
    x = x - kx_radii 
    y = y - ky_radii 
    z = z - kz_radii 

    if shape == "cube":
        Shape = np.ones((kx, ky, kz), dtype=bool)
    elif shape == "cross":
        Shape = np.zeros((kx, ky, kz), dtype=bool)
        Shape[kx_radii, :, kz_radii] = 1
        Shape[kx_radii , ky_radii, :] = 1
        Shape[:, ky_radii, kz_radii] = 1
    elif shape == "ellipse":
        Shape = (x / kx_radii) ** 2 + (y / ky_radii) ** 2 + (z /kz_radii) ** 2 <= 1
    else:
        raise ValueError('Please specify shape = "ellipse"/"cube"/"cross"')

    # considering the voxel spacings
    x=(x*dx).flatten()
    y=(y*dy).flatten()
    z=(z*dz).flatten()
    
    F = np.column_stack([np.ones_like(x), x, x**2,y, x * y, y**2,z, x * z, y * z, z**2])
    ind0 = np.where(Shape.flatten())[0]
    F0=F[ind0,:]
    F0_pinv = pinv(F0)

    # Jacobian for sigma = (2*(C₂ + C₅ + C₉)) / coeff
    J = np.zeros(10)  # Real-valued Jacobian
    J[[2,5,9]] = 2 / coeff  # ∂σ/∂C₂, ∂σ/∂C₅, ∂σ/∂C₉

    Indices = np.where(ROI)
    Indx, Indy, Indz = Indices[0], Indices[1], Indices[2]

    if len(Indx) == 0:
        raise ValueError("No valid ROI indices found.")
        
    def process_patch(j):
        """
        Process a single patch in parallel.
        """
        indx = slice(Indx[j] - kx_radii, Indx[j] + kx_radii + 1)
        indy = slice(Indy[j] - ky_radii, Indy[j] + ky_radii + 1)
        indz = slice(Indz[j] - kz_radii, Indz[j] + kz_radii + 1)
    
        PhiTR_patch  =  PhiTR[indx, indy, indz]
        Ref_patch = Ref[indx, indy, indz]
    
        # Ensure Shape_patch has the same dimensions as Shape
        Shape_patch = (np.abs(Ref_patch - Ref_patch[kx_radii, ky_radii, kz_radii]) <= thresh)
        Shape_patch &= Shape
    
        # Check if all elements are in shape, i.e., Shape_patch == Shape
        if np.array_equal(Shape_patch, Shape): 
            PhiTR_patch_in_shape = PhiTR_patch.flatten()[ind0]
            C = F0_pinv @ PhiTR_patch_in_shape # coefficients matrix 
            F_adap=F0   
 
        else:
            # removing the non-connected components, cc3d is compiled with C++  
            labeled = cc3d.connected_components(Shape_patch, connectivity=6)
            center_label = labeled[kx_radii, ky_radii, kz_radii]
            Shape_patch = (labeled == center_label)
        
            ind = np.where(Shape_patch.flatten())[0]
            PhiTR_patch_in_shape = PhiTR_patch.flatten()[ind] #vector
            F_adap = F[ind, :]
           
            C= pinv(F_adap) @ PhiTR_patch_in_shape     
   
        return  calculate_conductivity_and_uncertainty(F_adap, C, PhiTR_patch_in_shape, J,coeff)
    

    print("Processing...")
    results = Parallel(n_jobs=n_jobs)(delayed(process_patch)(j)
        for j in tqdm(range(len(Indx)), desc="Processing Patches", ncols=100, position=0, leave=True))

    temp_sigma, temp_uncertainty = zip(*results)

    sigma = np.zeros_like(PhiTR)
    sigma[Indices] = temp_sigma
    
    unc_sigma = np.zeros_like(PhiTR)
    unc_sigma[Indices] = temp_uncertainty

    crop_slice = (slice(kx_radii, -kx_radii), slice(ky_radii, -ky_radii), slice(kz_radii, -kz_radii))
    sigma, unc_sigma = sigma[crop_slice], unc_sigma[crop_slice]

    unc_sigma=unc_penalization(sigma,unc_sigma,Rmin=0,Rmax=2.5)

    print(f"Elapsed time: {time.time() - start_time:.2f} seconds")
    return sigma, unc_sigma


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


def imrescale(image, new_min=0, new_max=1,ROI=None): 
    # Intensity normalization to [0,1]
    if ROI is not None:
        ROI= ROI>0;
        old_min, old_max = np.min(image[ROI]), np.max(image[ROI])
    else:
        old_min, old_max = np.min(image.ravel()), np.max(image.ravel())
    return new_min + (image - old_min) * (new_max - new_min) / (old_max - old_min)
    
    
@njit(cache=True)
def calculate_conductivity_and_uncertainty(F_adap, C, PhiTR_patch_in_shape, J,coeff):
    Lap_PhiTR = 2 * (C[2] + C[5] + C[9]) 
    sigma = Lap_PhiTR/coeff
    
    n, rank = F_adap.shape 
    dof = n - rank
    if dof <= 0:
        return sigma, np.inf

    residuals = PhiTR_patch_in_shape - (F_adap @ C)
    var_residual = np.sum(residuals**2) / dof

    M= pinv(F_adap.T @ F_adap)
    # Propagate uncertainty as before
    cov_C = var_residual * M
    var_sigma = J @ cov_C @ J.T
    return sigma, np.sqrt(var_sigma)

import numpy as np
import numpy as np
from numpy.linalg import pinv
from joblib import Parallel, delayed
from tqdm import tqdm  # For progress bar
import time
import os
import cc3d
from scipy.optimize import minimize
from numba import njit
# last modified on 03/09/2025

def Phase_based_EPT_With_Uncertainty_Quantification(PhiTR, Ref, kernel_size=[5,5,5], shape="cube", thresh=0.1, omega=128e6*2*np.pi, h=None, ROI=None, n_jobs=-1):
    """
    Laplacian verison of Phase-based EPT:  conductivity = PhiTR_Lap / (2*mu0*omega)
    where PhiTR_Lap is the Laplacian of transceive phase.
    
    The Laplacian value is estimated by the adaptive 2nd order Savitzky-Golay filter in a cube/ellipse/cross window.
    The local kernel shape is anatomically adpated to the Ref image: only the voxels whose relative contrast
    with respect to the central voxel is lower than thresh (default= 0.1) are kept in the kernel.

     The uncertainty conductivity is quantified from covariation matrix of the SG fitted coefficients using the uncertainty propagation ( std_sigma = sqrt(J @ cov_C @ J.T))

    INPUTS:
    PhiTR: transceive phase image
    Ref: Magnitude or tissue segmentation
    shape: 'cube'/'ellipse'/'cross'
    thresh: 0.1 default
    h: [dx, dy, dz] : spacing of x, y, z, respectively
    ROI: Region of Interest mask (optional)
    n_jobs: Number of parallel jobs (default: -1, uses all available cores)

    OUTPUTS:
    sigma: reconstructed conductivity map [S/m]
    uncertainty: standard deviation map of reconstructed conductivity (based on the variance of estimated Lapacian)
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

    
    uncertainty = np.zeros_like(PhiTR)
    uncertainty[Indices] = temp_uncertainty

    
    uncertainty += np.abs(np.minimum(sigma,0)) + np.maximum(sigma-2.5,0)
    uncertainty *= ROI

    sigma = sigma[kx_radii:-kx_radii, ky_radii:-ky_radii, kz_radii:-kz_radii]
    uncertainty = uncertainty[kx_radii:-kx_radii, ky_radii:-ky_radii, kz_radii:-kz_radii]


    print(f"Elapsed time: {time.time() - start_time:.2f} seconds")
    return sigma, uncertainty
 
def imrescale(image, new_min=0, new_max=1,ROI=None): # Intensity normalization to [0,1]
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

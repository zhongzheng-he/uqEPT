import numpy as np
from numpy.linalg import pinv
from joblib import Parallel, delayed
from tqdm import tqdm  # For progress bar
import time
import os
import cc3d
from numba import njit

# last modified on 03/09/2025

def B1_based_EPT_With_Uncertainty_Quantification(B, Ref, kernel_size=[5,5,5], shape="cube", thresh=0.1, omega=128e6*2*np.pi, h=None, ROI=None, n_jobs=-1):
    """
    Adaptive Savitzky-Golay B1-based EPT with Bivariate Uncertainty Quantification
    B1-based EPT:  admittivity= Laplacian of B/(1j*mu0*omega*B)
                   conductivity = real(admittivity)
                   permittivity = imag(admittivity)/omega/eps_0
    The B can be the complex B1+ field for the standard Helmholtz EPT or the sqaure root of complex UTE/ZTE image for Image-based EPT. 

    The Laplacian value is estimated by the adaptive 2nd order Savitzky-Golay (SG) filter in a cube/ellipse/cross window.
    The local kernel shape is anatomically adpated to the Ref image: only the voxels whose relative contrast
    with respect to the central voxel is lower than thresh (default= 0.1) are kept in the kernel.

    The uncertainty conductivity/permittivity is quantified from the bivariate (real/imaginary) treatment of uncertainty propagation (cov_kappa = J_biv @ cov_C_bivariate @ J_biv.T)

    INPUTS:
    B: complex B1+ field or sqrt(complex image of UTE/ZTE)
    Ref: Magnitude or tissue segmentation
    shape: 'cube'/'ellipse'/'cross'
    thresh: 0.1 default
    h: [dx, dy, dz] : spacing of x, y, z, respectively
    ROI: Region of Interest mask (optional)
    n_jobs: Number of parallel jobs (default: -1, uses all available cores)

    OUTPUTS:
    sigma: reconstructed conductivity map [S/m]
    epsilon: relative permittivity
    unc_sigma: standard deviation map of reconstructed conductivity based on the covariance of the SG fitted coefficients.
    unc_epsilon: standard deviation map of reconstructed relative permittivity based on the covariance of the SG fitted coefficients.
    """
    start_time = time.time()

    if not isinstance(B, np.ndarray) or not np.iscomplexobj(B):
        raise ValueError("B must be a complex numpy array.")

    mu0       =  4*np.pi*1E-7
    eps0      =  8.854E-12
    j_mu0_omega = 1j*mu0*omega
    omega_eps0 =omega * eps0

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
    B  = np.pad(B, ((kx_radii, kx_radii), (ky_radii, ky_radii), (kz_radii, kz_radii)), mode='constant')
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

    J_c = np.zeros(10, dtype=np.complex128) # Initialization of Jacobian matrix of coefficient matrix of C estimated by the fitting  
    J_biv = np.zeros((2, 20), dtype=np.float64)  # Initialization of Bivariate Jacobian matrix : rows = [Re(kappa), Im(kappa)]; columns = [Re(C0..C9), Im(C0..C9)]     

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
    
        B_patch  =  B[indx, indy, indz]
        Ref_patch = Ref[indx, indy, indz]
    
        # Ensure Shape_patch has the same dimensions as Shape
        Shape_patch  = (np.abs(Ref_patch - Ref_patch[kx_radii, ky_radii, kz_radii]) <= thresh)
        Shape_patch &= Shape
    
        # Check if all elements are in shape, i.e., Shape_patch == Shape
        if np.array_equal(Shape_patch, Shape): 
            B_patch_in_shape = B_patch.flatten()[ind0]
            C = F0_pinv @ B_patch_in_shape # coefficients matrix 
            F_adap=F0   
        else:
            # removing the non-connected components, cc3d is compiled with C++  
            labeled = cc3d.connected_components(Shape_patch, connectivity=6)
            center_label = labeled[kx_radii, ky_radii, kz_radii]
            Shape_patch = (labeled == center_label)
            ind = np.where(Shape_patch.flatten())[0]
            B_patch_in_shape = B_patch.flatten()[ind] #vector
            F_adap = F[ind, :]
            C = pinv(F_adap) @ B_patch_in_shape# coefficients matrix
            

        return calculate_EPs_and_uncertainties(F_adap, C, B_patch_in_shape,j_mu0_omega, omega_eps0, J_c, J_biv)
            
    print("Processing...")
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_patch)(j)
        for j in tqdm(range(len(Indx)), desc="Processing Patches", ncols=100, position=0, leave=True))

    temp_sigma, temp_epsilon, temp_unc_sigma, temp_unc_epsilon = zip(*results)


    sigma = np.zeros_like(B,dtype=float)
    sigma[Indices] = temp_sigma
    epsilon = np.zeros_like(B,dtype=float)
    epsilon[Indices] = temp_epsilon

    unc_sigma = np.zeros_like(B,dtype=float)
    unc_sigma[Indices] = temp_unc_sigma
    unc_epsilon = np.zeros_like(B,dtype=float)
    unc_epsilon[Indices] = temp_unc_epsilon
    
    #add uncertainty if conductivity/permittivity is negative or too high which is not physically confident
    unc_sigma += np.abs(np.minimum(sigma,0)) + np.maximum(sigma-2.5,0)
    unc_epsilon += np.abs(np.minimum(epsilon-1,0)) + np.maximum(epsilon-100,0)
    unc_epsilon*=ROI
    
    sigma = sigma[kx_radii:-kx_radii, ky_radii:-ky_radii, kz_radii:-kz_radii]
    epsilon = epsilon[kx_radii:-kx_radii, ky_radii:-ky_radii, kz_radii:-kz_radii]
    unc_sigma = unc_sigma[kx_radii:-kx_radii, ky_radii:-ky_radii, kz_radii:-kz_radii]
    unc_epsilon = unc_epsilon[kx_radii:-kx_radii, ky_radii:-ky_radii, kz_radii:-kz_radii]

    print(f"Elapsed time: {time.time() - start_time:.2f} seconds")
    return sigma, epsilon, unc_sigma, unc_epsilon
    

def imrescale(image, new_min=0, new_max=1,ROI=None): # Intensity normalization to [0,1]
    if ROI is not None:
        ROI= ROI>0;
        old_min, old_max = np.min(image[ROI]), np.max(image[ROI])
    else:
        old_min, old_max = np.min(image.ravel()), np.max(image.ravel())
    return new_min + (image - old_min) * (new_max - new_min) / (old_max - old_min)



@njit(cache=True)
def calculate_EPs_and_uncertainties(F_adap, C, B_patch_in_shape,j_mu0_omega, omega_eps0, J_c, J_biv):

    Lap_B = 2 * (C[2] + C[5] + C[9])# Laplacian of B
    B1 = C[0]  # complex
    denom = j_mu0_omega * B1

    kappa = Lap_B / denom
    sigma = np.real(kappa)
    epsilon = np.imag(kappa) / omega_eps0


    n, rank = F_adap.shape
    dof =n-rank
    
    if dof <= 0:
        return sigma,epsilon, np.inf, np.inf
    else: 
        # --- Residual calculation ---
        B_patch_in_shape_fitted = F_adap.astype(np.complex128) @ C
        residual = B_patch_in_shape - B_patch_in_shape_fitted
        residual_real = np.real(residual)
        residual_imag = np.imag(residual)
        
        # --- Covariance matrix 
        residual_cov = np.array([
            [np.sum(residual_real**2), np.sum(residual_real * residual_imag)],
            [np.sum(residual_real * residual_imag), np.sum(residual_imag**2)]
        ], dtype=np.float64)  
        
        residual_cov /= dof
        
        M = pinv(F_adap.T @ F_adap)
        cov_C_bivariate = np.kron(residual_cov, M)  
        dkappa_dLaplacian = 2 / denom
        
        # Construct the bivariate Jacobian matrix (2 x 20) of kappa = Laplacian_B / (j mu0 omega B)
        # with respect to the real and imaginary parts of the complex coefficients C.
        # Bivariate Jacobian: rows = [Re(kappa), Im(kappa)]; columns = [Re(C0..C9), Im(C0..C9)]     
        
        J_c[0] = -Lap_B / (j_mu0_omega * (B1**2)) # dkappa_dB1 
        J_c[2] = dkappa_dLaplacian
        J_c[5] = dkappa_dLaplacian
        J_c[9] = dkappa_dLaplacian
     
        J_biv[0, :10] = J_c.real   # dRe(kappa)/dRe(C_i)
        J_biv[0, 10:] = -J_c.imag  # dRe(kappa)/dIm(C_i)
        J_biv[1, :10] = J_c.imag   # dIm(kappa)/dRe(C_i)
        J_biv[1, 10:] = J_c.real   # dIm(kappa)/dIm(C_i)
        
        # --- Uncertainty ---
        cov_kappa = J_biv @ cov_C_bivariate @ J_biv.T 
        unc_sigma = np.sqrt(np.abs(cov_kappa[0,0])) #real
        unc_epsilon = np.sqrt(np.abs(cov_kappa[1, 1]))/omega_eps0 #imaginairy
        return sigma,epsilon,unc_sigma, unc_epsilon




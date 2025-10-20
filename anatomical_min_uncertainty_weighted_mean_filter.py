# ///////////////////////////////////////////////////////////////////////////////////////////////
# // Zhongzheng He, PhD, ICube, Université de Strasbourg, Strasbourg, France
# // Contact: zhongzheng.he@unistra.fr
# ///////////////////////////////////////////////////////////////////////////////////////////////


import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm  # For progress bar
import time
import os
import cc3d

# last modified on 08/09/2025 by Zhongzheng He 

def anatomical_min_uncertainty_weighted_mean_filter(Im,Ref, uncertainty, kernel_size=[3,3,3], shape="cube", thresh=0.1, ROI=None, n_jobs=-1):
    """
    Applies a 3D filter using an anatomically-guided, uncertainty-based selection.

    This advanced filter is designed for edge-preserving denoising by leveraging
    both anatomical information and a voxel-wise uncertainty map. For each voxel,
    it first defines a local neighborhood that is structurally adapted to respect
    tissue boundaries (using `Ref` and `thresh`).

    Within this anatomically-constrained kernel, the function identifies the 25%
    of voxels that have the lowest uncertainty. It then calculates a weighted
    average of the values from *only this subset* of most reliable voxels. The
    weights are inversely proportional to the square of the uncertainty
    (1/uncertainty²), giving a strong preference to the most confident
    measurements. This dual approach of anatomical guidance and reliability-based
    selection makes the filter highly effective at reducing noise while preserving
    fine details.

    Parameters
    ----------
    Im : numpy.ndarray
        The 3D input image to be filtered.
    Ref : numpy.ndarray
        3D reference image for anatomical guidance, such as a magnitude image
        or a tissue segmentation map.
    uncertainty : numpy.ndarray
        A 3D map of the same shape as `Im`. Each voxel's value represents the
        uncertainty (e.g., standard deviation) of the corresponding voxel in `Im`.
    kernel_size : list of int, optional
        Dimensions [kx, ky, kz] of the filter kernel. All values must be odd.
        Default is [3, 3, 3].
    shape : {'cube', 'ellipse', 'cross'}, optional
        The base shape of the kernel before anatomical adaptation.
        Default is 'cube'.
    thresh : float, optional
        Threshold for anatomical adaptation (0 to 1). Only voxels in the
        `Ref` image with a relative intensity difference below this threshold
        (compared to the kernel's central voxel) are included in the
        filtering process. Default is 0.1.
    ROI : numpy.ndarray, optional
        3D binary mask defining the Region of Interest. Filtering is only
        applied within this region. If None, the ROI is automatically
        generated from non-zero voxels in the `Ref` image. Default is None.
    n_jobs : int, optional
        Number of CPU cores to use for parallel processing.
        -1 means using all available cores. Default is -1.

    Returns
    -------
    Im_min_unc : numpy.ndarray
        The 3D filtered image, with the same dimensions as the input `Im`.
    """
    cpu_cores = os.cpu_count()
    if n_jobs==-1:
        n_jobs=cpu_cores
        print(f"Number of CPU cores available and used: {cpu_cores}")
    else:
        print(f"Number of CPU cores available and used: {n_jobs}")
        
    # Initial setup
    kx, ky, kz = kernel_size
    if (kx%2 ==0) | (ky%2 ==0) | (kz%2 ==0):
        raise ValueError('the kernel size [kx,ky,kz] should be odd')
        
    kx_radii, ky_radii, kz_radii = (kx - 1) // 2, (ky - 1) // 2, (kz - 1) // 2

    start_time = time.time()

    Im = np.pad(Im, ((kx_radii, kx_radii), (ky_radii, ky_radii), (kz_radii, kz_radii)), mode='constant')
    Ref = np.pad(Ref, ((kx_radii, kx_radii), (ky_radii, ky_radii), (kz_radii, kz_radii)), mode='constant')
    uncertainty = np.pad(uncertainty, ((kx_radii, kx_radii), (ky_radii, ky_radii), (kz_radii, kz_radii)), mode='constant')

    if ROI is None: 
        ROI = Ref > 0
    else:
        ROI = np.pad(ROI, ((kx_radii, kx_radii), (ky_radii, ky_radii), (kz_radii, kz_radii)), mode='constant')
        
    if np.max(np.ravel(Ref)) > 1 :
        Ref= imrescale(Ref,ROI=ROI)

    Indices = np.where(ROI)
    Indx, Indy, Indz = Indices[0], Indices[1], Indices[2]

    if len(Indx) == 0:
        raise ValueError("No valid ROI indices found.")
    
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

    def process_patch(j):
        indx = slice(Indx[j] - kx_radii, Indx[j] + kx_radii + 1)
        indy = slice(Indy[j] - ky_radii, Indy[j] + ky_radii + 1)
        indz = slice(Indz[j] - kz_radii, Indz[j] + kz_radii + 1)
        
        Im_patch = Im[indx, indy, indz]
        Ref_patch = Ref[indx, indy, indz]
        uncertainty_patch = uncertainty[indx, indy, indz]
        
        Shape_patch = np.abs(Ref_patch - Ref_patch[kx_radii, ky_radii, kz_radii]) <= thresh
        Shape_patch &= Shape 
    
            # Check if all elements are in shape, i.e., Shape_patch == Shape
        if ~np.array_equal(Shape_patch, Shape): 
            labeled = cc3d.connected_components(Shape_patch, connectivity=6)
            center_label = labeled[kx_radii, ky_radii, kz_radii]
            Shape_patch = (labeled == center_label)
    
        ind = np.where(Shape_patch.flatten())[0]  
        Im_patch_in_shape = Im_patch.flatten()[ind]
        uncertainty_patch_in_shape =  uncertainty_patch.flatten()[ind] 

        # --- find Indices of the 1st-quartile-lowest uncertainties ---
        idx_sorted = np.argsort(uncertainty_patch_in_shape)[:np.maximum(np.int32(len(ind)/4), 1)]
        
        uncertainty_selected = uncertainty_patch_in_shape[idx_sorted]
        w = 1 / (uncertainty_selected**2)
        denominator = np.sum(w)
        
        if np.isnan(denominator) or np.isinf(denominator) or (denominator < 1e-10):
            return np.nanmedian(Im_patch_in_shape[idx_sorted])
        else:
            return np.sum(Im_patch_in_shape[idx_sorted] * w) / denominator
      
    print("Processing...")
    temp_min_unc = Parallel(n_jobs=n_jobs)(
        delayed(process_patch)(j)
        for j in tqdm(range(len(Indx)), desc="Processing Patches", ncols=100, position=0, leave=True))

    # Crop results
    Im_min_unc = np.zeros_like(Im)
    Im_min_unc[Indices] = np.ravel(temp_min_unc)
    Im_min_unc = Im_min_unc[kx_radii:-kx_radii, ky_radii:-ky_radii, kz_radii:-kz_radii]

    print(f"Elapsed time: {time.time() - start_time:.2f} seconds")
    return Im_min_unc

def imrescale(image, new_min=0, new_max=1,ROI=None): # Intensity normalization to [0,1]
    if ROI is not None:
        ROI= ROI>0;
        old_min, old_max = np.min(image[ROI]), np.max(image[ROI])
    else:
        old_min, old_max = np.min(image.ravel()), np.max(image.ravel())
    return new_min + (image - old_min) * (new_max - new_min) / (old_max - old_min)
    

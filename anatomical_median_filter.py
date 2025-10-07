# ///////////////////////////////////////////////////////////////////////////////////////////////
# // Zhongzheng He, PhD, ICube, Université de Strasbourg, Strasbourg, France
# // Contact: zhongzheng.he@unistra.fr
# ///////////////////////////////////////////////////////////////////////////////////////////////

import numpy as np
from scipy.ndimage import label, convolve
from joblib import Parallel, delayed
from tqdm import tqdm  # For progress bar
import time
import os
import cc3d

# last modified on 03/09/2025
def anatomical_median_filter(Im,Ref, kernel_size=[3,3,3], shape="cube", thresh=0.1, ROI=None, n_jobs=-1):
       """Applies a 3D anatomically-adaptive median filter.

    At each voxel, the filter considers a local neighborhood defined by
    `kernel_size` and a base `shape`. However, only the voxels within this
    neighborhood that have a similar intensity in the `Ref` image to the
    central voxel are included in the median calculation. The similarity is
    controlled by the `thresh` parameter. This ensures that the filtering is
    primarily performed within homogeneous regions, reducing blurring across
    anatomical boundaries.

    Parameters
    ----------
    Im : numpy.ndarray
        The 3D input image to be filtered.
    Ref : numpy.ndarray
        3D reference image for anatomical guidance, such as a magnitude image
        or a tissue segmentation map.
    kernel_size : list of int, optional
        Dimensions [kx, ky, kz] of the filter kernel. All values must be odd.
        Default is [3, 3, 3].
    shape : {'cube', 'ellipse', 'cross'}, optional
        The base shape of the kernel before anatomical adaptation.
        Default is 'cube'.
    thresh : float, optional
        Threshold for anatomical adaptation (0 to 1). Only voxels in the
        `Ref` image with a relative intensity difference below this threshold
        (compared to the kernel's central voxel) are included in the median
        calculation. Default is 0.1.
    ROI : numpy.ndarray, optional
        3D binary mask defining the Region of Interest. Filtering is only
        applied within this region. If None, the ROI is automatically
        generated from non-zero voxels in the `Ref` image. Default is None.
    n_jobs : int, optional
        Number of CPU cores to use for parallel processing.
        -1 means using all available cores. Default is -1.

    Returns
    -------
    numpy.ndarray
        The 3D filtered image with the same dimensions as the input `Im`.

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
    
        Shape_patch = np.abs(Ref_patch - Ref_patch[kx_radii, ky_radii, kz_radii]) <= thresh
        Shape_patch &= Shape
    
            # Check if all elements are in shape, i.e., Shape_patch == Shape
        if ~np.array_equal(Shape_patch, Shape): 
            labeled = cc3d.connected_components(Shape_patch, connectivity=6)
            center_label = labeled[kx_radii, ky_radii, kz_radii]
            Shape_patch = (labeled == center_label)
    
        ind = np.where(Shape_patch.flatten())[0] 
        Im_patch_in_shape = Im_patch.flatten()[ind]
    
        return np.nanmedian(Im_patch_in_shape)
  
    print("Processing...")
    temp_median = Parallel(n_jobs=n_jobs)(
        delayed(process_patch)(j)
        for j in tqdm(range(len(Indx)), desc="Processing Patches", ncols=100, position=0, leave=True))

    # Crop results
    Im_median = np.zeros_like(Im)
    Im_median[Indices] = np.ravel(temp_median)
    Im_median = Im_median[kx_radii:-kx_radii, ky_radii:-ky_radii, kz_radii:-kz_radii]

    print(f"Elapsed time: {time.time() - start_time:.2f} seconds")
    return Im_median


def imrescale(image, new_min=0, new_max=1,ROI=None): # Intensity normalization to [0,1]
    if ROI is not None:
        ROI= ROI>0;
        old_min, old_max = np.min(image[ROI]), np.max(image[ROI])
    else:
        old_min, old_max = np.min(image.ravel()), np.max(image.ravel())
    return new_min + (image - old_min) * (new_max - new_min) / (old_max - old_min)

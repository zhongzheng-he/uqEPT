<p align="center"><img src="https://github.com/zhongzheng-he/uqEPT/blob/main/figure.png" width=730 /></p>

# uqEPT
  
Uncertainty Quantification of Helmholtz-based Electrical Properties Tomography and its application to Uncertainty-Guided Post-processing

**Reference**:
Rigorous Uncertainty Quantification for Helmholtz-based EPT: Application to Uncertainty-Guided Post-processing, ISMRM 2026


## Requirements
You will need Python 3 and the following packages. You can install them all with a single `pip` or 'conda' command:

```bash
pip install numpy scipy joblib tqdm numba connected-components-3d
```

##



Description
-----------

This Python toolkit contains:

### Raw EPT Reconstcuion + Uncertainty Qantification 
* **Phase_based_EPT_With_Uncertainty_Quantification**: reconstructs conductivity from transceive phase via an anatomically-adaptive Savitzky-Golay filter + uncertainty map
  ```python
    sigma, unc_sigma = Phase_based_EPT_With_Uncertainty_Quantification(
    transceive_phase, # 3D array in [rad]
    Ref=magnitude_image_or_segmentation, # Ref will be normalized into [0,1]
    kernel_size=[11,11,11],
    thresh=0.05, # thresh value to distinguish homogenous region in Ref
    shape = 'cube', # available kernel shapes: cube/cross/ellipse
    omega=128e6*2*np.pi,# angular frequency in [rad/s]
    h=[0.001, 0.001, 0.001], # 1mm isotropic voxel size
    ROI= ROI)
  ```
* **B1_based_EPT_With_Uncertainty_Quantification**: reconstructs conductivity and permittivity from complex B field via an anatomically-adaptive Savitzky-Golay filter + uncertainty maps
  ```python
    sigma, epsilon, unc_sigma, unc_epsilon = B1_based_EPT_With_Uncertainty_Quantification(
    B, # complex B field: |B_1^+|*np.exp(transceive_phase/2) for standard Helmholtz-based EPT or np.sqrt(|M_UTE|)*np.exp(phase_UTE/2) for Image-based EPT
    Ref=magnitude_image_or_segmentation, # Ref will be normalized into [0,1]
    kernel_size=[11,11,11],
    thresh=0.05, # thresh value to distinguish homogenous region in Ref
    shape = 'cube', # available kernel shapes: cube/cross/ellipse
    omega=128e6*2*np.pi,# angular frequency in [rad/s]
    h=[0.001, 0.001, 0.001], # 1mm isotropic voxel size
    ROI= ROI)
  ```

### Post-processing filters
* **anatomical_median_filter**: performs 3D median filtering with a kernel that adapts to anatomical structures defined by a reference image. Conventional method for post-processing
  ```python
     sigma_f_med=anatomical_median_filter(
      sigma, # reconstructed conductivity map
      Ref=magnitude_image_or_segmentation, # Ref will be normalized into [0,1]
      kernel_size=[21,21,21],
      thresh=0.05, # thresh value to distinguish homogenous region in Ref
      shape = 'cube', # available kernel shapes: cube/cross/ellipse
      ROI = ROI)
  ```

* **anatomical_min_uncertainty_weighted_mean_filter**: an advanced adaptive filter that combines anatomical guidance with a reliability-based selection criterion. Within an anatomically-constrained kernel, it identifies the 25% of voxels with the lowest uncertainty and computes a weighted average from only this subset. The weights are inversely proportional to the square of the uncertainty (1/uncertainty²), making it highly effective at suppressing noise while trusting the most reliable data points.
  ```python
     sigma_f_min_unc=anatomical_median_filter(
      sigma, # reconstructed conductivity map
      Ref=magnitude_image_or_segmentation, # Ref will be normalized into [0,1]
      uncertainty = unc_sigma, # calculated uncertainty map
      kernel_size=[21,21,21],
      thresh=0.05, # thresh value to distinguish homogenous region in Ref
      shape = 'cube', # available kernel shapes: cube/cross/ellipse
      ROI = ROI)
  ```






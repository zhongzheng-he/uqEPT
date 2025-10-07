# UQEPT
Uncertainty Quantification of Helmholtz-based Electrical Properties Tomography and its application to Uncertainty-Guided Post-processing

Reference: Rigorous Uncertainty Quantification for Helmholtz-based EPT: Application to Uncertainty-Guided Post-processing, ISMRM 2026

Description
-----------

This toolkit includes the following functions:
* ** Phase_based_EPT_With_Uncertainty_Quantification.py** \\
  Reconstructs electrical conductivity from the transceive phase of an MRI scan using the Laplacian-based EPT method. The Laplacian is estimated using an anatomically-adaptive Savitzky-Golay filter. The function also calculates a voxel-wise map of the uncertainty in the conductivity reconstruction.

## Requirements

To use these functions, you will need Python 3 and the following packages. You can install them all with a single `pip` command:

```bash
pip install numpy scipy joblib tqdm numba connected-components-3d
```

##




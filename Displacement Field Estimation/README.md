# Displacement Field Estimation

This repository provides MATLAB-based code for 3D displacement field estimation from ultrasound volume matrices.
It reproduces the displacement-field estimation workflow and the two ROI visualizations at the `t1` time point shown in Fig. 4b of the manuscript, and also provides a batch-processing workflow for one complete cardiac cycle.
The pipeline uses a custom CUDA MEX implementation with automatic CPU fallback when GPU execution is unavailable.

The main scripts in this repository are:

- `DisplacementFieldEstimation.m`
- `DisplacementFieldEstimation_one_cardiac_cycle.m`

Running these scripts produces:

- dense 3D displacement field outputs
- ROI / displacement visualization for the two-time-point workflow

## Repository Structure

```text
.
├── DisplacementFieldEstimation.m
├── DisplacementFieldEstimation_one_cardiac_cycle.m
├── DisplacementFieldEstimation_data/
│   ├── volumeMatrices/
│   │   ├── volumeMatrix_ref.mat
│   │   └── volumeMatrix_t1.mat
│   └── ...
├── DisplacementFieldEstimation_one_cardiac_cycle_data/
│   ├── volumeMatrices/
│   │   └── volumeMatrices_one_cardiac_cycle.mat
│   └── ...
├── func/
│   ├── demons3d_cuda.cu
│   ├── demonsCPU.m
│   └── demonsGPU.m
├── plot/
│   └── roi_view.png
└── README.md
```

## File Description

| File | Description |
| :--- | :--- |
| **`DisplacementFieldEstimation.m`** | **Main script.** Estimates the displacement field between `ref` and `t1`, writes the binary displacement-field output, and generates ROI visualizations. |
| **`DisplacementFieldEstimation_one_cardiac_cycle.m`** | Performs sequential displacement-field estimation for data from one complete cardiac cycle. |
| **`func/demonsGPU.m`** | MATLAB wrapper for the GPU-based Demons registration workflow. |
| **`func/demons3d_cuda.cu`** | CUDA MEX kernel implementing the core 3D Demons update. |
| **`func/demonsCPU.m`** | CPU fallback implementation of the same registration workflow. |
| **`DisplacementFieldEstimation_data/`** | Data directory for the two-time-point workflow, including `volumeMatrix_ref.mat`, `volumeMatrix_t1.mat`, and the precomputed displacement field `D_ref2t1.bin`. |
| **`DisplacementFieldEstimation_one_cardiac_cycle_data/`** | Data directory for the one-cardiac-cycle workflow, including `volumeMatrices_one_cardiac_cycle.mat`. |
| **`plot/roi_view.png`** | Example ROI visualization output. |

## Requirements

- MATLAB R2021a or newer
- Image Processing Toolbox
- Parallel Computing Toolbox
- CUDA-capable GPU and compatible CUDA Toolkit for GPU acceleration

If CUDA MEX compilation fails or GPU execution is unavailable, the workflow automatically falls back to the CPU implementation.

## Usage

1. Download the dataset from [Figshare](https://doi.org/10.6084/m9.figshare.31986894). repare the input files in `DisplacementFieldEstimation_data/` for the two-time-point workflow, or in `DisplacementFieldEstimation_one_cardiac_cycle_data/` for the one-cardiac-cycle workflow.
2. Open MATLAB in the repository root, or run the scripts from the terminal with `matlab -batch`.
3. If displacement-field data already exist under `DisplacementFieldEstimation_data/DisplacementField/`, the program automatically skips the displacement-field computation step and proceeds with the downstream workflow using the existing results.
4. Run either of the main scripts below:

```bash
# Process DisplacementFieldEstimation_data
matlab -batch "DisplacementFieldEstimation"
```

```bash
# Process DisplacementFieldEstimation_one_cardiac_cycle_data
matlab -batch "DisplacementFieldEstimation_one_cardiac_cycle"
```

## Outputs

Running `DisplacementFieldEstimation.m` generates:

- a displacement-field binary file, such as `D_ref2t1.bin`
- ROI / displacement visualizations

Running `DisplacementFieldEstimation_one_cardiac_cycle.m` generates:

- pairwise displacement-field binary files across the cardiac cycle

## Citation

If you use this code in your research, please cite the associated manuscript and related references as appropriate.

## License

This repository is distributed under the Apache License 2.0. For further details, please refer to the `LICENSE` file.
Copyright (c) 2026 Youshen Xiao.

## Contact

For questions regarding this repository, please contact the corresponding author at shuai@pku.edu.cn.

## Appendix

### A. Demons Registration Algorithm

Whole-volume motion estimation is formulated as a dense 3D non-rigid registration problem between two ultrasound states (`moving -> fixed`). The objective is to recover a voxel-wise displacement field that minimizes local intensity mismatch while preserving the spatial smoothness of the deformation.

#### Mathematical Formulation

Let \(M\) denote the moving image, \(F\) denote the fixed image, and \(\mathbf{U}(\mathbf{x})\) denote the displacement field at voxel \(\mathbf{x}\). Under the additive Demons formulation, the update field is given by

\[
\Delta \mathbf{U}(\mathbf{x}) = \frac{\big(F(\mathbf{x}) - M(\mathbf{x}+\mathbf{U}(\mathbf{x}))\big)\,\nabla F(\mathbf{x})}{\|\nabla F(\mathbf{x})\|^2 + \big(F(\mathbf{x}) - M(\mathbf{x}+\mathbf{U}(\mathbf{x}))\big)^2},
\]

and the displacement field is iteratively updated according to

\[
\mathbf{U}^{k+1}(\mathbf{x}) = \mathbf{U}^{k}(\mathbf{x}) + \Delta \mathbf{U}(\mathbf{x}).
\]

This formulation couples the local intensity residual with the structural direction provided by the fixed-image gradient, while the denominator regularizes the update magnitude in low-gradient or high-mismatch regions.

#### Implementation Procedure

1. Intensity normalization and histogram matching are applied to reduce inter-frame contrast bias.
2. Registration is performed in a coarse-to-fine manner using a multi-resolution pyramid.
3. At each resolution level, the moving image is warped using the current estimate of \(\mathbf{U}\), after which the incremental update is computed and accumulated iteratively.
4. Gaussian smoothing is applied to each component of the displacement field after every iteration to enforce spatial coherence.
5. Updates in low-information or ill-conditioned regions are suppressed to improve numerical stability.
6. The final displacement field is represented as a dense 3D vector field \(\mathbf{U}\) and stored using MATLAB's `single` data type, corresponding to single-precision floating-point format.

### B. Data Organization

For `DisplacementFieldEstimation_data/`:

- number of files: 3
- included files:
  `volumeMatrices/volumeMatrix_ref.mat` (219 MB),
  `volumeMatrices/volumeMatrix_t1.mat` (219 MB),
  `DisplacementField/D_ref2t1.bin` (5.8 GB)

To mitigate potential failures in displacement-field computation caused by limited hardware resources, the precomputed displacement field for the `ref`-to-`t1` pair is also included in this submission.

For `DisplacementFieldEstimation_one_cardiac_cycle_data/`:

- number of files: 1
- included files:
  `volumeMatrices/volumeMatrices_one_cardiac_cycle.mat` (12.9 GB)

### C. Hardware Requirement and Test Record

The GPU-based workflow was tested on an NVIDIA GeForce RTX 4090 GPU with 48 GB of memory (`409048G`).

During one representative run, the recorded peak resource usage was:

- peak GPU memory usage: 44,247 MiB
- peak RAM usage: 18,458 MiB

Accordingly, the GPU workflow should be executed on hardware with sufficient GPU memory and system RAM. In practice, a 48 GB GPU is recommended for stable execution of the full-resolution workflow.

Because the implementation involves GPU acceleration, interpolation, and floating-point arithmetic, minor numerical differences may be observed across different hardware or software environments. However, the overall displacement-field estimation results and visual patterns should remain broadly consistent.

### D. References

1. Thirion, J.-P. (1998). Image matching as a diffusion process: an analogy with Maxwell's demons. *Medical Image Analysis*, 2(3), 243-260. https://doi.org/10.1016/S1361-8415(98)80022-4
2. Vercauteren, T., Pennec, X., Perchant, A., & Ayache, N. (2009). Diffeomorphic Demons: Efficient non-parametric image registration. *NeuroImage*, 45(1, Supplement), S61-S72. https://doi.org/10.1016/j.neuroimage.2008.10.040
3. MathWorks. *imregdemons*. Image Processing Toolbox Documentation. https://www.mathworks.com/help/images/ref/imregdemons.html

# Displacement Field Estimation

This repository provides MATLAB code for 3D displacement field estimation from ultrasound volume matrices. It supports two workflows:

1. a two-time-point workflow that estimates the displacement field from `ref` to `t1` and generates ROI visualizations corresponding to Fig. 4b of the manuscript;
2. a one-cardiac-cycle workflow that performs pairwise displacement-field estimation across a sequence of 60 individually stored volume files.

The implementation uses a custom CUDA MEX kernel when GPU execution is available and automatically falls back to a CPU implementation otherwise.

## 1. System requirements

### Operating system

- CPU execution can be used on operating systems supported by MATLAB R2021a or later.
- The GPU-enabled workflow was validated in a Linux server environment.

### Software dependencies

- MATLAB R2021a or later
- Image Processing Toolbox
- Parallel Computing Toolbox
- For GPU acceleration: a CUDA toolkit version compatible with the MATLAB release in use

### Tested environment

The GPU workflow was validated with the following software stacks:

- MATLAB 2023a + CUDA 11.8 (`nvcc`) + GCC 11.4.0
- MATLAB R2021a + CUDA 12.5 (`nvcc`) + GCC 9.4.0

The following incompatible combinations were observed in the validated environment:

- CUDA 12.x with the default `mexcuda` configuration in MATLAB R2021a, because `mexcuda` / `nvcc` rejected the deprecated `compute_35` target used by the default compilation path
- GCC 13.x with MATLAB R2021a CUDA MEX compilation, because `mexcuda` reported `unsupported GNU version`

### Non-standard hardware

To run the software itself, no frame grabber, acquisition card, ultrasound scanner, or photoacoustic hardware is required. The code operates offline on saved MATLAB volume files.

Optional non-standard hardware for acceleration:

- an NVIDIA CUDA-capable GPU for the GPU registration path

Validated GPU platforms:

- NVIDIA H100
- NVIDIA A800
- NVIDIA GeForce RTX 4090 48 GB

Observed peak resource usage during one representative full-resolution GPU run:

- peak GPU memory: 44,247 MiB
- peak system RAM: 18,458 MiB

In practice, a 48 GB GPU is recommended for stable full-resolution recomputation. If no compatible GPU is available, the code will run with the CPU fallback, but runtime may become substantially longer.

### Expected setup time on a standard computer

If MATLAB and the required toolboxes are already installed, repository setup typically takes under 1 minutes.

## 2. Installation guide

No additional package installation is required beyond MATLAB and its toolboxes.

To verify the environment, run:

```bash
matlab -batch "mex -setup C++; gpuDeviceCount('available'); mexcuda('func/demons3d_cuda.cu', '-output', 'func/demons3d_cuda')"
```

If this command completes without error, the environment is ready. The check typically takes under 1 minutes.

## 3. Demo

### Demo A: two-time-point workflow

This demo reproduces the displacement-field estimation workflow between `ref` and `t1` and generates ROI visualizations.

Expected input layout:

```text
DisplacementFieldEstimation_data/
├── volumeMatrices/
│   ├── volumeMatrix_ref.mat
│   └── volumeMatrix_t1.mat
└── DisplacementField/
    └── D_ref2t1.bin   (optional precomputed cache)
```

Run:

```bash
matlab -batch "DisplacementFieldEstimation"
```

Expected output:

- a dense 3D displacement-field binary file such as `D_ref2t1.bin`
- an ROI visualization image written to `results/roi_view.png`

If `DisplacementFieldEstimation_data/DisplacementField/D_ref2t1.bin` already exists, the script reuses it and skips the most expensive registration step.

Expected runtime:

- on an NVIDIA GeForce RTX 4090 48 GB GPU, Demo A takes approximately 2.5 minutes
- runtime may be longer on other hardware, especially when GPU acceleration is unavailable. Full-resolution execution on the CPU may require several hours.

### Demo B: one-cardiac-cycle workflow

This demo processes a cardiac cycle stored as 60 split volume files and computes pairwise displacement fields for consecutive frame pairs `(01->02), (03->04), ..., (59->60)`.

Expected input layout:

```text
DisplacementFieldEstimation_one_cardiac_cycle_data/
└── volumeMatrices/
    ├── volumeMatrix_01.mat
    ├── volumeMatrix_02.mat
    ├── ...
    └── volumeMatrix_60.mat
```

Run:

```bash
matlab -batch "DisplacementFieldEstimation_one_cardiac_cycle"
```

Expected output:

- pairwise displacement-field binary files in
  `DisplacementFieldEstimation_one_cardiac_cycle_data/DisplacementField/`
- output filenames follow the pattern `D_01to02.bin`, `D_03to04.bin`, ..., `D_59to60.bin`

Expected runtime:

- on an NVIDIA GeForce RTX 4090 48 GB GPU, Demo B takes approximately 90 minutes
- runtime may be longer on other hardware, especially when GPU acceleration is unavailable. Full-resolution execution on the CPU may require several hours.

## 4. Instructions for use

### 4.1 Running the two-time-point workflow on your own data

1. Convert your two volumetric datasets to MATLAB `.mat` files.
2. Place them under:

```text
DisplacementFieldEstimation_data/volumeMatrices/
```

3. Name the files:

- `volumeMatrix_ref.mat`
- `volumeMatrix_t1.mat`

4. Make sure the `.mat` files contain a 3D volume variable compatible with the script. In the current implementation, variables named `volumeMatrix_ref` and `volumeMatrix_t1` are supported.
5. Run:

```bash
matlab -batch "DisplacementFieldEstimation"
```

6. The displacement field will be written to:

```text
DisplacementFieldEstimation_data/DisplacementField/
```

and the visualization image will be written to:

```text
results/roi_view.png
```

### 4.2 Running the one-cardiac-cycle workflow on your own data

1. Convert each cardiac-cycle frame to a separate MATLAB `.mat` file.
2. Place the files under:

```text
DisplacementFieldEstimation_one_cardiac_cycle_data/volumeMatrices/
```

3. Name the files sequentially:

- `volumeMatrix_01.mat`
- `volumeMatrix_02.mat`
- `...`
- `volumeMatrix_60.mat`

4. Ensure that numbering is consecutive, starts from `01`, and matches the expected total frame count.
5. Each file should contain one 3D volume variable. In the current implementation, variables named `volumeMatrix`, `volume`, or `img` are accepted; if the file contains only one variable, that variable will also be used.
6. Run:

```bash
matlab -batch "DisplacementFieldEstimation_one_cardiac_cycle"
```

7. Output files will be written to:

```text
DisplacementFieldEstimation_one_cardiac_cycle_data/DisplacementField/
```

### 4.3 Input data format

- Input files must be MATLAB `.mat` files.
- Each input file must contain a 3D ultrasound volume.
- All frames within one workflow must have identical spatial dimensions.
- For best compatibility, follow the naming convention and directory structure used in the examples above.

### 4.4 Parameter settings

The main runtime parameters are defined in the entry scripts:

- `DisplacementFieldEstimation.m`
- `DisplacementFieldEstimation_one_cardiac_cycle.m`

Key parameters include:

- `cfg.scale`: preprocessing upsampling factor
- `cfg.pyrLevels`: number of multi-resolution pyramid levels
- `cfg.iterations`: iterations per pyramid level
- `cfg.smooth`: Gaussian regularization strength
- `cfg.useGPU`: whether to attempt GPU execution first

For manuscript reproduction, keep the default parameter settings unchanged.

### 4.5 Output files

Two-time-point workflow:

- displacement field: `DisplacementFieldEstimation_data/DisplacementField/D_ref2t1.bin`
- ROI figure: `results/roi_view.png`

One-cardiac-cycle workflow:

- pairwise displacement fields: `DisplacementFieldEstimation_one_cardiac_cycle_data/DisplacementField/D_XXtoYY.bin`

All displacement fields are written as raw single-precision binary files containing the three displacement components.

### 4.6 Reproducing the main results of the paper

To reproduce the main repository-level results associated with the manuscript:

1. keep the default parameters unchanged;
2. place the provided `ref` and `t1` example volumes in `DisplacementFieldEstimation_data/volumeMatrices/` and run `DisplacementFieldEstimation` to reproduce the two-ROI visualization workflow corresponding to Fig. 4b;
3. place the 60 split cardiac-cycle volumes in `DisplacementFieldEstimation_one_cardiac_cycle_data/volumeMatrices/` and run `DisplacementFieldEstimation_one_cardiac_cycle` to reproduce the one-cardiac-cycle pairwise displacement-field workflow.

## References

1. Thirion, J.-P. (1998). Image matching as a diffusion process: an analogy with Maxwell's demons. *Medical Image Analysis*, 2(3), 243-260. https://doi.org/10.1016/S1361-8415(98)80022-4
2. Vercauteren, T., Pennec, X., Perchant, A., & Ayache, N. (2009). Diffeomorphic Demons: Efficient non-parametric image registration. *NeuroImage*, 45(1, Supplement), S61-S72. https://doi.org/10.1016/j.neuroimage.2008.10.040
3. MathWorks. *imregdemons*. Image Processing Toolbox Documentation. https://www.mathworks.com/help/images/ref/imregdemons.html

## License

This repository is distributed under the Apache License 2.0. See `LICENSE.txt` for details.

## Contact

For correspondence regarding this repository, please contact the repository owner.


# Dynamic Functional Connectivity Analysis

This repository provides MATLAB-based code for Dynamic Functional Connectivity (dFC) analysis, to reproduce the long-term dynamic analysis as demonstrated in **Supplementary Video 8**.


The main script in this repository is:

- Dynamic_functional_connectivity_analysis.m

Running this script produces:

- Dynamic phase matrix
- Dynamic transitions of connectivity states




## Repository Structure

```text
.
├── dFC_data/                                  
├── Dynamic_functional_connectivity_analysis.m    
├── utils/                                    
│   ├── dfc_phase.m                           
│   ├── dfc_matrix_show.m                    
│   └── state_plot.m             
├── LICENSE.txt
└── README.md                                
```

## File Description

| File | Description |
| :--- | :--- |
| **`dFC_data/`** | Input data folder for dFC analysis.
| **`Dynamic_functional_connectivity_analysis.m`** | **Main script.** Executes the full dFC pipeline, generating dynamic phase matrix and dynamic transitions of connectivity states. |
| **`dfc_phase.m`** | Performs phase extraction from signals. |
| **`dfc_matrix_show.m`** | Visualizes dynamic transitions of phase matrix. |
| **`state_plot.m`** | Visualizes dynamic transitions of connectivity states. |





## Requirements

- MATLAB R2022b or later
- MATLAB Signal Processing Toolbox

The software has been tested on MATLAB R2023b under Windows 10 and Windows 11.

No Python, CUDA, external compiled libraries, GPU or non-standard hardware are required to run this offline analysis code. The photoacoustic/ultrasound imaging system is only required for data acquisition and is not required for running the demo or analysis scripts.

## Installation

Install MATLAB R2023b and the MATLAB Signal Processing Toolbox. Then download or clone this repository, open MATLAB, and set the repository root folder as the current MATLAB working directory.

The code does not require compilation. Installation typically takes less than 3 minutes on a standard desktop computer, excluding MATLAB installation and dataset download time.

## Demo and Usage
1. Download the dataset from [Figshare](https://doi.org/10.6084/m9.figshare.31986894) and place the `dFC_data.mat` file into the `dFC_data/` folder.
   
   **Expected directory structure:**
   ```text
   .
   ├── dFC_data/
   │   └── dFC_data.mat        # Input data for dFC analysis
   ├── Dynamic_functional_connectivity_analysis.m 
   └── ...
   ```

2. Open MATLAB.
3. Run the main script:
- Dynamic_functional_connectivity_analysis.m 

The demo dataset is provided as `dFC_data.mat`. Running `Dynamic_functional_connectivity_analysis.m` on this dataset displays the dynamic phase matrix and dynamic transitions of connectivity states in MATLAB.

The demo does not automatically save output files. The generated results are displayed as MATLAB figures during execution.

The expected run time for the demo is less than 2 minutes on a standard desktop computer running MATLAB R2023b.

## Outputs

Running Dynamic_functional_connectivity_analysis.m will generate results including:

- Dynamic phase matrix
- Dynamic transitions of connectivity states

To run the software on user data, prepare the input data in MATLAB `.mat` format with the same variable structure as `dFC_data.mat`, place the file into the `dFC_data/` folder, and modify the data-loading section of `Dynamic_functional_connectivity_analysis.m` if the file name or variable names are different from the demo dataset.

## Citation

If you use this code in your research, please cite the associated work if applicable.

## License

This project is licensed under the Apache License 2.0. See LICENSE for details.

## Contact

For questions regarding this repository, please contact the corresponding author at shuai@pku.edu.cn.

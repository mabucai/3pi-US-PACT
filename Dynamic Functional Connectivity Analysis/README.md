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
├── data/                                  
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
| **`data/`** | Input data folder for dynamic functional connectivity analysis.
| **`Dynamic_functional_connectivity_analysis.m`** | **Main script.** Executes the full dFC pipeline, generating dynamic phase matrix and dynamic transitions of connectivity states. |
| **`dfc_phase.m`** | Performs phase extraction from signals. |
| **`dfc_matrix_show.m`** | Visualizes dynamic transitions of phase matrix. |
| **`state_plot.m`** | Visualizes dynamic transitions of connectivity states. |





## Requirements

- MATLAB R2022b or later

## Usage
1. Download the dataset from [Figshare](https://doi.org/10.6084/m9.figshare.31986894) and place the `Dynamic_functional_connectivity_data.mat` file into the `data/` folder.
   
   **Expected directory structure:**
   ```text
   .
   ├── data/
   │   └── Dynamic_functional_connectivity_data.mat        # Input data
   ├── Dynamic_functional_connectivity_analysis.m 
   └── ...

2. Open MATLAB.
3. Run the main script:
- Dynamic_functional_connectivity_analysis.m 



## Outputs

Running Dynamic_functional_connectivity_analysis.m will generate results including:

- Dynamic phase matrix
- Dynamic transitions of connectivity states



## Citation

If you use this code in your research, please cite the associated work if applicable.

## License

This project is licensed under the Apache License 2.0. See LICENSE for details.
Copyright (c) 2026 Youshen Xiao.

## Contact

For questions regarding this repository, please contact the corresponding author at shuai@pku.edu.cn.



# COX_Statistical-scaling

This repository contains the analysis code associated with the Statistical Scaling project, led by Dr. Dezerae Cox. This manuscript has been submitted for publication under the title *"Statistical Scaling: A Nuanced Alternative to p-value Thresholding for Variable Biological Datasets"*

## Prerequisites

This analysis assumes a standard installation of Python 3 (=> 3.6). For specific package requirements, see the environment.yml file, or create a new conda environment containing all packages by running ```conda create -f environment.yml```.

## Raw data and databases

In the case of the published dataset comparisons, raw data in the form of supplementary datasets from each publication can be accessed using the [```raw_data```](src/omics/raw_data.py) script. Unfortunately, due to journal subscription requirements, in some instances journal access is required. In these cases datasets will need to be manually downloaded for which the URL information is available via the script and summarised in Supplementary Table 1.

Novel simulated and digitised datasets generated for this study have also been uploaded as an open-access Zenodo dataset available [here](https://doi.org/10.5281/zenodo.8127985). 

Finally, various public databases (e.g. UniProt) were queried as indicated in the accompanying manuscript for which access protocols are also provided in the respective analysis workflow where appropriate.

## Workflow

To reproduce analyses presented in the manuscript, where processing order is important for individual analyses scripts have been numbered and should be run in order before unnumbered counterparts. Otherwise, there is no interdependence between the analysis for different data types (simulated, omics, biomarkers).

In addition, each figure can be generated using the scripts provided under the ```src/figures``` folder. 

## A note on randomisation

By it's very nature, randomisation produces different datasets upon each run of the code. In this case, the numpy seed value has been set to ensure reproducibility between code runs in which random numbers are generated. By fixing the seed, the same sequence of random numbers can be generated each time the code is run, enabling consistent results and facilitating debugging, testing, and sharing of code and data across different environments.


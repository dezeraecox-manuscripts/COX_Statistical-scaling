# Complete initial cleanup as specified in individual scripts
# from src.omics.preprocessing import (
#     Bader_AD_CSF,
#     Bai_AD_CSF,
#     Bereman_ALS_CSF,
#     Collins_MND_CSF,
#     DAlessandro_COVID_serum,
#     Di_COVID_serum,
# )

import os, re
import pandas as pd
import numpy as np

from loguru import logger
logger.info('Import OK')

input_folder = 'data/omics/'
output_folder = 'results/omics/'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Get list of raw data input files
file_list = [filename for filename in os.listdir(input_folder) if '.csv' in filename]

# Compile cleaned data
compiled = pd.concat([pd.read_csv(f'{input_folder}{filename}') for filename in file_list])
compiled.drop([col for col in compiled.columns.tolist() if 'Unnamed: ' in col], axis=1, inplace=True)
compiled.dropna(inplace=True)

# Save compiled dataset to csv
compiled['dataset'] = [dataset.split('_')[0] if '_' in dataset else dataset for dataset in compiled['dataset']]
compiled.to_csv(f'{output_folder}compiled.csv')

# Check number of replicates for each dataset
compiled.groupby('dataset').max()['replicate'].reset_index()

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

from loguru import logger

logger.info('Import OK')

input_folder = 'data/biomarkers/'
output_folder = 'results/biomarkers/'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Read in raw datasets
file_list = [filename for filename in os.listdir(input_folder) if '.csv' in filename]

raw_data = []
for filename in file_list:
    if filename in ['sample_labels.csv', 'stats_datasets.csv']:
        continue
    dataset = filename.split('.')[0]
    raw_vals = pd.read_csv(f'{input_folder}{filename}', header=None)
    raw_vals.columns = ['sample_position', 'value']
    raw_vals['position'] = raw_vals['sample_position'].round(0)
    if dataset == 'Conti':
        raw_vals['value'] = 10**(raw_vals['value'])
        
    raw_vals['dataset'] = dataset
    raw_data.append(raw_vals)
raw_data = pd.concat(raw_data)

# Read in sample info
sample_labels = pd.read_csv(f'{input_folder}sample_labels.csv', sep=',')

# map sample info to raw data
cleaned_data = pd.merge(raw_data, sample_labels, on=['dataset', 'position'], how='outer')
# map neurological controls to new labels (some studies use e.g. Parkinson's as control patients)
sample_map = {
    'Control': 'Control',
    'ALS': 'ALS',
    'FTLD': 'FTLD',
    'AD': 'AD',
    # 'MN': 'MN',
    'FTLD-tau': 'FTLD',
    'Parkinsons': 'Control',
    'MS/NMO': 'Control',
    'GBS/MFS': 'Control',
    'GBS': 'Control',
    # 'C9ORF72': 'C9ORF72',
    # 'GRN': 'GRN'
}
cleaned_data['label'] = cleaned_data['label'].map(sample_map)

# save cleaned dataset
cleaned_data.to_csv(f'{output_folder}cleaned_datasets.csv')
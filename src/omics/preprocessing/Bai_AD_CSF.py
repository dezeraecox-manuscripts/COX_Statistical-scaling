import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils import randomised_control

from loguru import logger

logger.info('Import OK')

input_folder = 'data/omics/raw_data/'
output_folder = 'data/omics/'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Read in excel datasheet
raw_data = pd.read_excel(f'{input_folder}Bai_AD_CSF.xlsx', sheet_name=None)
raw_data.keys()
# Find datasheet of interest
raw_data = raw_data['S4C']

# Fix column headers
col_headers = raw_data.iloc[2].T.fillna('').tolist()
col_headers[0:4] = ['Unnamed:0', 'Protein_IDs', 'protein_name', 'gene_name', ]
col_headers[8:13] = [
    f'{sample.replace("LPC", "Control")}' for sample in col_headers[8:13]]

raw_data.drop([0, 1, 2], inplace=True)
raw_data.columns = col_headers
raw_data.drop([col for col in raw_data.columns.tolist() if 'Unnamed:' in col], axis=1, inplace=True)
raw_data.drop(['# PSMs', '# peptides', '# PSMs', '# peptides', ], axis=1, inplace=True)

# Clean Protein accessions
raw_data['Protein_IDs'] = raw_data['Protein_IDs'].str.split('|').str[1]
# Drop duplicates
raw_data = raw_data.drop_duplicates(subset='Protein_IDs', keep=False)
raw_data = raw_data[['Protein_IDs'] + [col for col in raw_data.columns if 'Control' in col] + [col for col in raw_data.columns if 'AD' in col] ].copy()

info_cols = ['Protein_IDs']

# Generate pseudo control data
pControl_data = pd.melt(
    raw_data,
    id_vars=info_cols,
    value_vars=[col for col in raw_data.columns if 'Control' in col],
    var_name=['sample'],
    value_name='abundance'
    )

pControl_data['replicate'] = pControl_data['sample'].str.replace(r'\D+', '')
pControl_data['sample'] = pControl_data['sample'].str.replace(r'\d+', '')

pControl_data = randomised_control(
    dataframe=pControl_data,
    col='abundance', 
    group_cols=info_cols + ['sample'],
    comparison_type='per_protein'
    )

pControl_data['sample'] = pControl_data['sample'] + pControl_data['replicate']
pControl_data = pd.pivot_table(
    data=pControl_data, 
    index=info_cols,
    columns=['sample'], 
    values='pseudo_abundance'
)
pControl_data.columns = [f'pseudo{col}' for col in pControl_data.columns]

prot_data = pd.merge(
    raw_data,
    pControl_data.reset_index(),
    how='outer',
    on=info_cols
)


# Calculate ratios
prot_data[f'Control_mean'] = prot_data[[col for col in prot_data.columns if (
    'Control' in col and 'pseudo' not in col)]].mean(axis=1)
prot_data[f'pseudoControl_mean'] = prot_data[[
    col for col in prot_data.columns if 'pseudo' in col]].mean(axis=1)

treatment_cols = [col for col in prot_data.columns if ('AD') in col]
for col in treatment_cols:
    prot_data[f'{col}_ratio'] = prot_data[col] / \
    prot_data[f'Control_mean']

control_cols = [col for col in prot_data.columns if 'Control' in col and 'pseudo' not in col and 'mean' not in col]
for col in control_cols:
    prot_data[f'pseudo{col}_ratio'] = prot_data[col] / \
    prot_data[f'pseudoControl_mean']

# Select ratio columns and melt to standard format
prot_ratios = pd.melt(
    prot_data,
    id_vars=info_cols,
    value_vars=[col for col in prot_data.columns if 'ratio' in col],
    var_name=['sample'],
    value_name='ratio'
)

prot_ratios['replicate'] = prot_ratios['sample'].str.replace(r'\D+', '')
prot_ratios['sample'] = prot_ratios['sample'].str.replace(r'\d+', '').str.replace('_ratio', '')

# Add log2 vals
prot_ratios['log2_ratio'] = np.log2(prot_ratios['ratio'].astype(float))
prot_ratios['dataset'] = "Bai"
prot_ratios['source'] = prot_ratios['sample'].str.split('_').str[-1].map({'Control': 'C', 'pseudoControl': 'C', 'AD': 'AD'})
prot_ratios['type'] = ['pseudo' if 'pseudo' in sample else 'experimental' for sample in prot_ratios['sample']]
prot_ratios = prot_ratios[['dataset', 'Protein_IDs', 'source', 'type', 'replicate', 'ratio', 'log2_ratio']].copy()

# Save to excel
prot_ratios.to_csv(f'{output_folder}Bai_AD_CSF.csv')

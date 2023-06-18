from src.utils import map_protein_accessions
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
raw_data = pd.read_excel(
    f'{input_folder}Collins_MND_CSF.xlsx', sheet_name=None)
raw_data.keys()
# Find datasheet of interest
raw_data = raw_data['Global Results']

# Fix column headers
raw_data.rename(columns={
    'Protein': 'protein_name',
    'Accession': 'Protein_IDs'
}, inplace=True)

# remove other disease columns
raw_data.drop(['OND 1 (MS)',
               'OND 2 (MS)',
               'OND 3 (LMND)',
               'OND 4 (UMND)',
               'OND 5 (AD)',
               'OND 6 (AD)'], axis=1, inplace=True)

# clean protein ID column, remove protein groups and unreviewed proteins
raw_data = map_protein_accessions(raw_data, from_type='ACC+ID', to_type='ACC+ID')
# 15 proteins represented by more than one isoform, drop these as impossible to compare
raw_data = raw_data.drop_duplicates(subset='Protein_IDs', keep=False)
raw_data = raw_data[['Protein_IDs'] + [col for col in raw_data.columns if ' ' in col]].copy()


info_cols = ['Protein_IDs']

# Generate pseudo control data
pControl_data = pd.melt(
    raw_data,
    id_vars=info_cols,
    value_vars=[col for col in raw_data.columns if 'HC' in col],
    var_name=['sample'],
    value_name='abundance'
    )

pControl_data[['sample', 'replicate']] = pControl_data['sample'].str.split(' ', expand=True)

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
prot_data[f'HC_mean'] = prot_data[[col for col in prot_data.columns if (
    'HC' in col and 'pseudo' not in col)]].mean(axis=1)
prot_data[f'pseudoHC_mean'] = prot_data[[
    col for col in prot_data.columns if 'pseudo' in col]].mean(axis=1)

treatment_cols = [col for col in prot_data.columns if ('sALS') in col]
for col in treatment_cols:
    prot_data[f'{col}_ratio'] = prot_data[col] / \
    prot_data[f'HC_mean']

control_cols = [col for col in prot_data.columns if 'HC' in col and 'pseudo' not in col and 'mean' not in col]
for col in control_cols:
    prot_data[f'pseudo{col}_ratio'] = prot_data[col] / \
    prot_data[f'pseudoHC_mean']

# Select ratio columns and melt to standard format
prot_ratios = pd.melt(
    prot_data,
    id_vars=info_cols,
    value_vars=[col for col in prot_data.columns if 'ratio' in col],
    var_name=['sample'],
    value_name='ratio'
)


prot_ratios[['sample', 'replicate']] = prot_ratios['sample'].str.replace('_ratio', '').str.split(' ', expand=True)

# Add log2 vals
prot_ratios['log2_ratio'] = np.log2(prot_ratios['ratio'].astype(float))
prot_ratios['dataset'] = "Collins"
prot_ratios['source'] = prot_ratios['sample'].str.split('_').str[-1].map({'HC': 'C', 'pseudoHC': 'C',  'sALS': 'ALS'})
prot_ratios['type'] = ['pseudo' if 'pseudo' in sample else 'experimental' for sample in prot_ratios['sample']]
prot_ratios = prot_ratios[['dataset', 'Protein_IDs', 'source', 'type', 'replicate', 'ratio', 'log2_ratio']].copy().replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna()

# Save to excel
prot_ratios.to_csv(f'{output_folder}Collins_MND_CSF.csv')

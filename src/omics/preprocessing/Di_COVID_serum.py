import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.core.reshape.merge import merge
import seaborn as sns

from src.utils import randomised_control
from src.utils import map_protein_accessions

from loguru import logger

logger.info('Import OK')

input_folder = 'data/omics/raw_data/'
output_folder = 'data/omics/'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Read in excel datasheet
raw_data = pd.read_excel(
    f'{input_folder}Di_COVID_serum.xls', sheet_name='b. Protein matrix')
raw_data.rename(columns={'Unnamed: 0': 'sample'}, inplace=True)
raw_data.drop([col for col in raw_data.columns.tolist() if 'Unnamed: ' in col], axis=1, inplace=True)


# Extract accession information
raw_data['Protein_IDs'] = [entry.split('|')[1] if 'HUMAN' in entry else np.nan for entry in raw_data['Protein']]

# Remove unmapped entry (iRT-Kit?)
raw_data.dropna(subset=['Protein_IDs'], inplace=True)

# Remove proteins not mapped
raw_data = raw_data.dropna(subset=['Protein_IDs']) # 472/493 rows left

# relabel columns in sample_replicate format
cols = ['Protein_IDs'] + [col for col in raw_data.columns if 'Protein' not in col]
new_cols = ['Protein_IDs'] + [col.replace('Normal', 'HC').replace('Patients', 'COV')
                              for col in cols if 'Protein' not in col]
raw_data = raw_data[cols].copy()
raw_data.rename(columns=dict(zip(cols, new_cols)), inplace=True)

raw_data = map_protein_accessions(raw_data, from_type='ACC+ID', to_type='ACC+ID')
raw_data.drop(['genes'], axis=1, inplace=True)
# Drop duplicates
raw_data = raw_data.drop_duplicates(subset='Protein_IDs', keep=False)

info_cols = ['Protein_IDs']

# Generate pseudo control data
pControl_data = pd.melt(
    raw_data,
    id_vars=['Protein_IDs'],
    value_vars=[col for col in raw_data.columns if 'HC' in col],
    var_name=['sample'],
    value_name='abundance'
    )

pControl_data = randomised_control(
    dataframe=pControl_data,
    col='abundance', 
    group_cols=['Protein_IDs'],
    comparison_type='per_protein'
    )

pControl_data = pd.pivot_table(
    data=pControl_data, 
    index=['Protein_IDs'],
    columns=['sample'], 
    values='pseudo_abundance'
)
pControl_data.columns = [f'pseudo{col}' for col in pControl_data.columns]

prot_data = pd.merge(
    raw_data,
    pControl_data.reset_index(),
    how='outer',
    on=['Protein_IDs']
)


# Calculate ratios
prot_data['HC_mean'] = prot_data[
    [col for col in prot_data.columns if ('HC' in col and 'pseudo' not in col)]
].mean(axis=1)

prot_data['pseudoHC_mean'] = prot_data[
    [col for col in prot_data.columns if 'pseudo' in col]
].mean(axis=1)

treatment_cols = [col for col in prot_data.columns if 'COV' in col]
for col in treatment_cols:
    prot_data[f'{col}_ratio'] = prot_data[col] / prot_data['HC_mean']

control_cols = [col for col in prot_data.columns if 'HC' in col and 'pseudo' not in col and 'mean' not in col]
for col in control_cols:
    prot_data[f'pseudo{col}_ratio'] = prot_data[col] / \
        prot_data['pseudoHC_mean']

# Select ratio columns and melt to standard format
prot_ratios = pd.melt(
    prot_data.reset_index(),
    id_vars=['Protein_IDs'],
    value_vars=[col for col in prot_data.columns if 'ratio' in col],
    var_name=['sample'],
    value_name='ratio'
)
prot_ratios[['sample', 'replicate', 'discard']] = prot_ratios['sample'].str.split('_', expand=True)
prot_ratios.drop('discard', axis=1, inplace=True)

# Add log2 vals
prot_ratios['log2_ratio'] = np.log2(prot_ratios['ratio'].astype(float))
prot_ratios['dataset'] = "Di"
prot_ratios['source'] = prot_ratios['sample'].str.split('_').str[-1].map({'HC': 'C', 'pseudoHC': 'C', 'COV': 'COVID'})
prot_ratios['type'] = ['pseudo' if 'pseudo' in sample else 'experimental' for sample in prot_ratios['sample']]
prot_ratios = prot_ratios[['dataset', 'Protein_IDs', 'source', 'type', 'replicate', 'ratio', 'log2_ratio']].copy().dropna()


# Save to excel
prot_ratios.to_csv(f'{output_folder}Di_COVID_serum.csv')

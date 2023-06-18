import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils import randomised_control
from src.utils import map_protein_accessions, uniprot_map

from loguru import logger

logger.info('Import OK')

input_folder = 'data/omics/raw_data/'
output_folder = 'data/omics/'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Read in excel datasheet
raw_data = pd.read_excel(f'{input_folder}Bereman_ALS_CSF.xlsx', sheet_name=None)
# Find datasheet of interest
raw_data = raw_data['csfprot']


raw_data.columns = [f'LFQ_{x}_{sample_type}'for x, sample_type in enumerate(raw_data.T[0].tolist()[:-8])] + raw_data.T.index.tolist()[-8:]
raw_data.drop(0, inplace=True)

# Relabel columns with consistent names
col_names = {
    "Gene names": 'Protein_IDs'
}
raw_data.rename(columns=col_names, inplace=True) 


# remove contaminant proteins
raw_data = raw_data[~raw_data['Protein IDs'].str.contains('CON_')] # remove contaminants - 11

# clean protein ID column, remove protein groups and unreviewed proteins
raw_data = map_protein_accessions(raw_data.dropna(subset=['Protein_IDs']), from_type='GENENAME', to_type='ACC', species='9606')

# Drop duplicates
raw_data = raw_data.drop_duplicates(subset='Protein_IDs', keep=False)
raw_data = raw_data[['Protein_IDs'] + [col for col in raw_data.columns if 'LFQ' in col]].copy()

info_cols = ['Protein_IDs']

# Generate pseudo control data
pControl_data = pd.melt(
    raw_data,
    id_vars=info_cols,
    value_vars=[col for col in raw_data.columns if '_H' in col],
    var_name=['sample'],
    value_name='abundance'
    )

pControl_data[['_', 'replicate', 'sample']] = pControl_data['sample'].str.split('_', expand=True)

pControl_data = randomised_control(
    dataframe=pControl_data,
    col='abundance', 
    group_cols=info_cols + ['sample'],
    comparison_type='per_protein'
    )

pControl_data['sample'] = pControl_data['_'] + '_' + pControl_data['replicate'] + '_' + pControl_data['sample']
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
prot_data['H_mean'] = prot_data[[col for col in prot_data.columns if ('_H' in col and 'pseudo' not in col)]].mean(axis=1)

prot_data['pseudoH_mean'] = prot_data[[col for col in prot_data.columns if 'pseudo' in col]].mean(axis=1)


treatment_cols = [col for col in prot_data.columns if ('_A' in col and 'A/H' not in col)]
for col in treatment_cols:
    prot_data[f'{col}_ratio'] = prot_data[col] / prot_data['H_mean']

control_cols = [col for col in prot_data.columns if '_H' in col and 'pseudo' not in col and 'mean' not in col]
for col in control_cols:
    prot_data[f'pseudo{col}_ratio'] = (prot_data[col] / prot_data['pseudoH_mean'])

# Select ratio columns and melt to standard format
prot_ratios = pd.melt(
    prot_data,
    id_vars=info_cols,
    value_vars=[col for col in prot_data.columns if 'ratio' in col],
    var_name=['sample'],
    value_name='ratio'
)
prot_ratios[['LFQ', 'replicate', 'sample', 'discard']] = prot_ratios['sample'].str.split('_', expand=True)
prot_ratios['sample'] = prot_ratios['LFQ'] + '_' + prot_ratios['sample']
prot_ratios.drop('discard', axis=1, inplace=True)
prot_ratios.drop('LFQ', axis=1, inplace=True)

# Add log2 vals
prot_ratios['log2_ratio'] = np.log2(prot_ratios['ratio'].astype(float))
prot_ratios['dataset'] = "Bereman"
prot_ratios['source'] = prot_ratios['sample'].str.split('_').str[-1].map({'H': 'C', 'A': 'ALS'})
prot_ratios['type'] = ['pseudo' if 'pseudo' in sample else 'experimental' for sample in prot_ratios['sample']]
prot_ratios = prot_ratios[['dataset', 'Protein_IDs', 'source', 'type', 'replicate', 'ratio', 'log2_ratio']].copy()


# Save to excel
prot_ratios.to_csv(f'{output_folder}Bereman_ALS_CSF.csv')

import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
raw_data = pd.read_excel(f'{input_folder}Bader_AD_CSF.xlsx', sheet_name=None)
# Find datasheet of interest
raw_data = raw_data['MSB-19-9356_Dataset EV3']
raw_data.columns = raw_data.T.index.tolist()[:3] + raw_data.T[0].tolist()[3:]
raw_data.drop(0, inplace=True)

# Read in sample map
sample_map = pd.read_excel(f'{input_folder}Bader_AD_samplemap.xlsx', sheet_name=None)
sample_map = sample_map['MSB-19-9356_Dataset EV2']
# map Kiel and Magdeburg as one location (as per sample_map data)
sample_map['collection site'] = ['Magdeburg' if location == 'Kiel' else location for location in sample_map['collection site']]
# classify AD based on biochemical classification (opposed to clinical)
sample_map['sample_type'] = sample_map['primary biochemical AD classification'].str.split(' ').str[-1]
sample_map['sample_shortname'] = [f'{location}_{sample}_{x}' for x, (location, sample) in enumerate(sample_map[['collection site', 'sample_type']].values)]
col_names = dict(sample_map[['sample name', 'sample_shortname']].fillna('noid').values)
col_names['PG.ProteinAccessions (uniprot protein accessions)'] = 'Protein_IDs'
col_names['PG.Genes (gene names)'] = 'gene_names'
col_names['Protein names'] = 'protein_names'

# Relabel columns according to sample shortnames
raw_data.columns = [col_names[col] for col in raw_data.columns]

# clean protein ID column, remove protein groups and unreviewed proteins
raw_data = map_protein_accessions(raw_data, from_type='ACC+ID', to_type='ACC+ID')
# drop proteins represented by more than one isoform, as impossible to compare
raw_data = raw_data.drop_duplicates(subset='Protein_IDs', keep=False)
raw_data = raw_data[['Protein_IDs'] + [col for col in raw_data.columns if 'control' in col] + [col for col in raw_data.columns if 'AD' in col] ].copy()

info_cols = ['Protein_IDs']
# for col in [col for col in raw_data.columns if col not in info_cols]:
#     raw_data[col] = 10**raw_data[col].astype(float)


# Generate pseudo control data
pControl_data = pd.melt(
    raw_data,
    id_vars=info_cols,
    value_vars=[col for col in raw_data.columns if 'control' in col],
    var_name=['sample'],
    value_name='abundance'
    )

pControl_data[['location', 'sample', 'replicate']] = pControl_data['sample'].str.split('_', expand=True)

pControl_data = randomised_control(
    dataframe=pControl_data,
    col='abundance', 
    group_cols=info_cols + ['location', 'sample'],
    comparison_type='per_protein'
    )

pControl_data['sample'] = pControl_data['location'] + '_' + pControl_data['sample'] + '_' + pControl_data['replicate']
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
for location in ['Berlin', 'Sweden', 'Magdeburg']:
    prot_data[f'{location}_mean'] = prot_data[[col for col in prot_data.columns if ('control' in col and 'pseudo' not in col)]].mean(axis=1)
    prot_data[f'pseudo{location}_mean'] = prot_data[[
        col for col in prot_data.columns if 'pseudo' in col]].mean(axis=1)

treatment_cols = [col for col in prot_data.columns if 'AD' in col]
for col in treatment_cols:
    location, _, replicate = col.split('_')
    prot_data[f'{col}_ratio'] = prot_data[col] / prot_data[f'{location}_mean']

control_cols = [col for col in prot_data.columns if 'control' in col and 'pseudo' not in col and 'mean' not in col]
for col in control_cols:
    location, _, replicate = col.split('_')
    prot_data[f'pseudo{col}_ratio'] = prot_data[col] / \
        prot_data[f'pseudo{location}_mean']

# Select ratio columns and melt to standard format
prot_ratios = pd.melt(
    prot_data,
    id_vars=info_cols,
    value_vars=[col for col in prot_data.columns if 'ratio' in col],
    var_name=['sample'],
    value_name='ratio'
)
prot_ratios[['location', 'sample', 'replicate', 'discard']] = prot_ratios['sample'].str.split('_', expand=True)
prot_ratios['sample'] = prot_ratios['location'] + '_' + prot_ratios['sample']
prot_ratios.drop('discard', axis=1, inplace=True)

# Add log2 vals
prot_ratios['log2_ratio'] = np.log2(prot_ratios['ratio'].astype(float))
prot_ratios['dataset'] = [f"Bader_{location.replace('pseudo', '')}" for location in prot_ratios['location']]
prot_ratios['source'] = prot_ratios['sample'].str.split('_').str[-1].map({'control': 'C', 'AD': 'AD'})
prot_ratios['type'] = ['pseudo' if 'pseudo' in sample else 'experimental' for sample in prot_ratios['sample']]
prot_ratios = prot_ratios[['dataset', 'Protein_IDs', 'source', 'type', 'replicate', 'ratio', 'log2_ratio']].copy()

# Save to excel
prot_ratios.to_csv(f'{output_folder}Bader_AD_CSF.csv')

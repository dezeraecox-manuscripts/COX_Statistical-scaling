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
    f'{input_folder}DAlessandro_COVID_serum.xlsx', sheet_name='Proteomics Report')
raw_data.rename(columns={'Unnamed: 0': 'sample'}, inplace=True)
raw_data.drop([col for col in raw_data.columns.tolist() if 'Unnamed: ' in col], axis=1, inplace=True)

# transpose to have proteins x samples
raw_data = raw_data.set_index('sample').T
raw_data.drop('IL-6_CLIN', axis=0, inplace=True) # remove clinical IL6 measure

# Map protein names to human accessions, retrieved from https://www.uniprot.org/uniprot/?query=yourlist:M20211025F248CABF64506F29A91F8037F07B67D124F7AA4&sort=yourlist:M20211025F248CABF64506F29A91F8037F07B67D124F7AA4&columns=yourlist(M20211025F248CABF64506F29A91F8037F07B67D124F7AA4),id,entry%20name,reviewed,protein%20names,genes,organism,length
# 486 out of 493 Gene name identifiers were successfully mapped to 499 UniProtKB IDs in the table.
accession_map = pd.read_excel(
    f'{input_folder}DAlessandro_COVID_serum_map.xlsx')
accession_map.rename(
    columns={'yourlist:M20211025F248CABF64506F29A91F8037F07B67D124F7AA4': 'Query'}, inplace=True)
# check for proteins assigned to more than 1 ID
accession_map[accession_map.duplicated(subset='Query', keep=False)]
# remove these, such that they cannot be mapped (as the ID is non-unique)
accession_map = accession_map.drop_duplicates(subset='Query', keep=False)
# map back to original data
raw_data['Protein_IDs'] = raw_data.index.map(dict(accession_map[['Query', 'Entry']].values))

# Remove proteins not mapped
raw_data = raw_data.dropna(subset=['Protein_IDs']) # 472/493 rows left

# relabel columns in sample_replicate format
cols = ['Protein_IDs'] + [col for col in raw_data.columns if 'Protein_IDs' not in col]
new_cols = ['Protein_IDs'] + [str('_'.join([col[0], re.findall(r"\d+", col)[0]])).replace('C', 'HC').replace('A', 'COV') for col in cols if 'Protein_IDs' not in col]
compiled = raw_data.reset_index()[cols].copy()
compiled.rename(columns=dict(zip(cols, new_cols)), inplace=True)

# Generate pseudo control data
pControl_data = pd.melt(
    compiled,
    id_vars=['Protein_IDs'],
    value_vars=[col for col in compiled.columns if 'HC' in col],
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
    compiled,
    pControl_data.reset_index(),
    how='outer',
    on=['Protein_IDs']
).set_index('Protein_IDs')


# Calculate ratios
prot_data['HC_mean'] = prot_data[
    [col for col in prot_data.columns if ('HC' in col and 'pseudo' not in col)]
].mean(axis=1)

prot_data['pseudoHC_mean'] = prot_data[
    [col for col in prot_data.columns if 'pseudo' in col]
].mean(axis=1)

treatment_cols = [col for col in prot_data.columns if 'HC' not in col]
for col in treatment_cols:
    prot_data[f'{col}_ratio'] = prot_data[col] / prot_data[f'HC_mean']

control_cols = [col for col in prot_data.columns if 'HC' in col and 'pseudo' not in col and 'mean' not in col]
for col in control_cols:
    prot_data[f'pseudo{col}_ratio'] = prot_data[col] / \
        prot_data[f'pseudoHC_mean']

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
prot_ratios['dataset'] = "DAlessandro"
prot_ratios['source'] = prot_ratios['sample'].str.split('_').str[-1].map({'HC': 'C', 'pseudoHC': 'C', 'COV': 'COVID'})
prot_ratios['type'] = ['pseudo' if 'pseudo' in sample else 'experimental' for sample in prot_ratios['sample']]
prot_ratios = prot_ratios[['dataset', 'Protein_IDs', 'source', 'type', 'replicate', 'ratio', 'log2_ratio']].copy()

# Save to excel
prot_ratios.to_csv(f'{output_folder}DAlessandro_COVID_serum.csv')

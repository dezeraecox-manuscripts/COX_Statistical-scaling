import os
import pandas as pd
import numpy as np
from src.utils import pval_smoothing

from loguru import logger

logger.info('Import OK')

input_path = 'results/biomarkers/cleaned_datasets.csv'
output_folder = 'results/biomarkers/'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Read in summary
clean_data = pd.read_csv(input_path)
clean_data.drop([col for col in clean_data.columns.tolist() if 'Unnamed: ' in col], axis=1, inplace=True)
clean_data.dropna(inplace=True)
clean_data = clean_data[clean_data['source'] == 'CSF'].copy()
clean_data['value'] = [round(val, 2) for val in clean_data['value']]

# ----Add normalised data to mean control----
normalised = []
for dataset, df in clean_data.groupby('dataset'):
    if dataset in ['Gaian', 'Junttila']:  # NFL not TDP43, no control
        continue

    df['control_mean'] = df[df['label'] == 'Control']['value'].mean()
    df['norm_value'] = df['value'] / df['control_mean']
    normalised.append(df)
normalised = pd.concat(normalised)

# pivot normalised raw data
normalised_mean = pd.pivot_table(
    normalised,
    index=['dataset', 'source'],
    columns='label',
    values='norm_value'
)
normalised_mean = normalised.groupby(['dataset', 'source', 'label']).mean().reset_index()

normalised.to_csv(f'{output_folder}normalised_summary.csv')

# --------Smooth raw data, then pivot--------

smoothed = []
for dataset, df in normalised.groupby('dataset'):
    num_replicates = df.groupby(['label']).count()['source'].max()
    logger.info(num_replicates)
    if df.dropna().shape[0] < 2:
        continue
    smooth = pval_smoothing(
        df, 
        sample_cols=['norm_value'], 
        group_cols=['dataset', 'label', 'source'], 
        popmean=1, 
        penalty_factor=num_replicates*10, 
        complete=False).reset_index()
    smoothed.append(smooth)
smoothed = pd.concat(smoothed)

smoothed_mean = pd.pivot_table(
    smoothed,
    index=['dataset', 'source'],
    columns='label',
    values='norm_value'
)


# -----------Generate comparison------------

comparison = pd.merge(normalised_mean.rename(columns={'norm_value': 'raw_norm_mean'}), smoothed.rename(columns={'norm_value': 'smooth_norm_mean'}), on=['dataset', 'label', 'source'])

comparison['difference'] = comparison['smooth_norm_mean'] - comparison['raw_norm_mean']
comparison['hue'] = [f'{dataset}_{label}' for dataset, label in comparison[['dataset', 'label']].values]

comparison.to_csv(f'{output_folder}comparison.csv')

# -----------Smoothing mean values as 'sample'-----------

# Smooth means of individual datasets
smoothed_means = pval_smoothing(
    df=comparison[['label', 'smooth_norm_mean']], 
    sample_cols=['smooth_norm_mean'], 
    group_cols=['label'], 
    popmean=1, 
    penalty_factor=20, 
    complete=True).reset_index()
smoothed_means= pd.melt(
    smoothed_means, 
    id_vars=['label'], 
    value_vars=['mean_smooth_norm_mean', 'scaled_smooth_norm_mean'], 
    value_name='norm_value',
    var_name='data_type')
smoothed_means['data_type'] = smoothed_means['data_type'].str.split('_').str[0]
smoothed_means.to_csv(f'{output_folder}smoothed_means.csv')

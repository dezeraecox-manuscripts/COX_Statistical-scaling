import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from src.utils import correlation_df
from sklearn import preprocessing
from scipy.stats import pearsonr

logger.info('Import OK')

input_path = 'results/omics/scale_thresholded.csv'
output_folder = 'results/omics/'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

pd.options.display.float_format = '{:.5f}'.format

treatments = {
    'Bader': 'AD',
    'Bereman': 'ALS',
    "Bai": 'AD',
    'Collins': 'ALS',
    'DAlessandro': 'COVID',
    'Di': 'COVID',
}
# Read in cleaned data
compiled = pd.read_csv(input_path)
compiled.drop([col for col in compiled.columns.tolist() if 'Unnamed: ' in col], axis=1, inplace=True)

# -----------------------Scaling-----------------------
# before measuring the correlation, scale datasets to the common bounds of -1, 1 for log2 data and 0 centre for raw data
scaled = []
for (dataset, source, ratio_type), df in compiled.groupby(['dataset', 'source', 'ratio_type']):
    if source == 'C':
        continue
    if dataset not in treatments:
        continue
    if ratio_type == 'log2':
        absmax_val = df['ratio'].abs().max() # Use this to make sure scaling in symmetrical
        df['scaled_ratio'] = preprocessing.maxabs_scale(df['ratio'].tolist()+[absmax_val, -absmax_val])[:-2]
        fig, axes = plt.subplots(2, 1)
        sns.distplot(df['ratio'], color='grey', ax=axes[0])
        sns.kdeplot(df['ratio'], color='grey', ax=axes[0])
        sns.distplot(df['scaled_ratio'], color='firebrick', ax=axes[1])
        sns.kdeplot(df['scaled_ratio'], color='firebrick', ax=axes[1])
        for ax in axes:
            ax.set_xlim(-1, 1)
        plt.show()
    if ratio_type == 'raw':
        df['scaled_ratio'] = preprocessing.robust_scale(df['ratio'])
        fig, axes = plt.subplots(2, 1)
        sns.distplot(df['ratio'], color='darkorange', ax=axes[0])
        sns.kdeplot(df['ratio'], color='darkorange', ax=axes[0])
        sns.distplot(df['scaled_ratio'], color='rebeccapurple', ax=axes[1])
        sns.kdeplot(df['scaled_ratio'], color='rebeccapurple', ax=axes[1])
        for ax in axes:
            ax.set_xlim(-10, 10)
        plt.show()
    scaled.append(df)
scaled = pd.concat(scaled)
scaled.to_csv(f'{output_folder}scaled.csv')

# --------------------Correlation--------------------
# calculate correlation matrix before and after smoothing, with and without filtering
correlations = []
for (processing, ratio_type), dataframe in scaled.groupby(['processing', 'ratio_type']):
    for filter_style, df in zip(['none', 'minvalue', 'thresholded'], [dataframe, dataframe[dataframe['ratio'].abs() > 0.8], dataframe.dropna(subset=['thresholded'])]):
        pivot_df = pd.pivot(
            df[df['source'] != 'C'],
            index=['Protein_IDs'],
            columns=['dataset'],
            values='ratio'
        )
        for dataset_a in pivot_df.columns.tolist():
            for dataset_b in pivot_df.columns.tolist():
                if dataset_a == dataset_b:
                    continue
                correlation = correlation_df(
                    pivot_df[dataset_a],
                    pivot_df[dataset_b],).T
                
                correlation['processing'] = processing
                correlation['ratio_type'] = ratio_type
                correlation['filter_type'] = filter_style
                
                correlation['dataset_a'] = dataset_a
                correlation['dataset_b'] = dataset_b
                correlation['key'] = "-".join(sorted([dataset_a, dataset_b]))
                
                correlations.append(correlation)
correlations = pd.concat(correlations)

# combine before and after correlations
correlations['treatment_a'] = correlations['dataset_a'].map(treatments)
correlations['treatment_b'] = correlations['dataset_b'].map(treatments)
correlations['comparison_type'] = [a if a == b else 0 for a,
                                             b in correlations[['treatment_a', 'treatment_b']].values]

correlations.to_csv(f'{output_folder}correlations.csv')
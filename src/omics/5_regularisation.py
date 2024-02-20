import functools
import os
import random
import re
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from scipy.stats import wilcoxon
from sklearn.linear_model import Lasso, ElasticNet, Ridge

from src.utils import correlation_df, one_sample_ttest, pval_smoothing

logger.info('Import OK')

input_path = 'results/omics/compiled.csv'
output_folder = 'results/omics/'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Set seed for reproducibility
num_combinations = 20

# Read in compiled data for subsampling
compiled = pd.read_csv(input_path)
compiled = compiled[compiled['type'] == 'experimental'].copy()
compiled.drop([col for col in compiled.columns.tolist() if 'Unnamed: ' in col], axis=1, inplace=True)
compiled = compiled[compiled['dataset'].isin(['Bader'])].copy()


# =======================Functions=======================

def plotter(comb, df, reg_type='lasso'):
    
    palette = {
        'nothing': 'lightgrey',
        'top': 'firebrick',
        'bottom': 'firebrick',
    }
    
    df['color'] = ['Selected' if val != 0 else 'Not selected' for val in df[f'{reg_type}']]
    df = df.sort_values(f'{reg_type}').copy()
    df['rank'] = ['top' if (pos < 25) else ('bottom' if (pos > len(df)-25) else 'nothing') for pos in range(len(df))]
    df['rank'] = [rank if val != 0 else 'nothing' for rank, val in df[['rank', f'{reg_type}']].values]
    df['scale_size'] = ['big' if val < 50**0.05 else 'small' for val in df['scalefactor_log2_ratio']]
    df['inverse_scale'] = 1/df['scalefactor_log2_ratio']
    
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    axes = ax.ravel()
    
    for data, ax in zip(['raw', 'smooth', f'{reg_type}'], axes):
        for dtype in ['nothing', 'bottom', 'top']:
            sns.scatterplot(
                data=df[df['rank'] == dtype],
                x='total',
                y=data,
                ax=ax,
                color=palette[dtype],
                size='scale_size',
                sizes={'big':100, 'small':20},
                alpha=0.75
            )
            ax.set_title(f'{comb} {data.capitalize()}')
            ax.set_ylim(-0.2, 0.2)
            ax.legend('')
            

    plt.show()


def apply_smoothing(compiled, penalty, combination):
    smooth_sampled = pval_smoothing(
            df=compiled[compiled['replicate'].isin(combination)].copy(),
            sample_cols=['log2_ratio'],
            group_cols=['dataset', 'Protein_IDs', 'source', 'type'],
            popmean=0,
            penalty_factor=penalty,
            complete=True
        )
    
    return smooth_sampled


def apply_regularisation(compiled, combination, reg_type='lasso'):
    for_reg = compiled[compiled['replicate'].isin(combination)].copy()
    for_reg = pd.pivot_table(for_reg, index=['dataset',	'Protein_IDs'], columns=['replicate'], values=['log2_ratio']).dropna()
    for_reg['crl'] = 0
    for_reg = for_reg.T
        # Calculate the mean log2(fold-change) for each protein
    fold_change_columns = [col for col in for_reg.columns]
        # Prepare the feature matrix X and target vector y
    y = for_reg.reset_index()[['replicate']].values
    y = [0,0,0,0,0, 1]
    X = for_reg[fold_change_columns].values
    
    if reg_type == 'lasso':
        model = Lasso(alpha=1/10**7)
        
    elif reg_type == 'ridge':    # Initialize and fit the  model
        model = Ridge(alpha=1/10**7)
    model.fit(X, y)

    for_reg = for_reg.T
    for_reg.columns = [f'{i}' if j == '' else f'{j}' for i,j in for_reg.columns]
    for_reg[f'{reg_type}'] = model.coef_


    return for_reg



#------------------------Sampling------------------------
# COllect subsample for each dataset and calculate smoothed and raw means per protein
sampled = []
for num_reps in [5]:

    penalty = 10*num_reps # penalty for pVal smoothing
    
    logger.info(f'{num_reps}: {len(compiled["replicate"].unique().tolist())}')
    # Generate all possible combination subsets
    replicates = list(combinations(compiled['replicate'].unique().tolist(), num_reps))
    logger.info(f'{num_reps} replicate combinations: {len(replicates)}')
    
    if len(replicates) < num_combinations:
        logger.info(f'{num_reps} not processed due to insufficient replicates')
        continue
    
    random.seed(10301)
    replicates = [replicates[random.randint(0, len(replicates))] for x in range(num_combinations)]
    for x, combination in enumerate(replicates):
        combination
        
        # Calculate raw ground truth
        raw_total = compiled.copy().groupby(['dataset', 'Protein_IDs', 'source', 'type']).mean()['log2_ratio']
        
        # Complete smoothing
        smooth_sampled = apply_smoothing(compiled, penalty, combination)
        
        # test LASSO
        lasso_proteins = apply_regularisation(compiled, combination, reg_type='lasso')
        # Test Ridge
        ridge_proteins = apply_regularisation(compiled, combination, reg_type='ridge')


        merged_df = functools.reduce(
            lambda left, right: pd.merge(
                left, right, on=['dataset', 'Protein_IDs', 'source', 'type'], how='outer'), [
            smooth_sampled.reset_index().rename(columns={'scaled_log2_ratio': 'smooth', 'mean_log2_ratio': 'raw'}), 
            raw_total.reset_index().rename(columns={'log2_ratio': 'total'})],)
        merged_df['combination'] = x
        merged_df['num_reps'] = num_reps
        
        merged_df = pd.merge(merged_df, ridge_proteins.reset_index(), on=['dataset', 'Protein_IDs'], how='outer')
        merged_df = pd.merge(merged_df, lasso_proteins.reset_index(), on=['dataset', 'Protein_IDs'], how='outer')
        plotter(combination, df=merged_df.dropna(), reg_type='lasso')
        plotter(combination, merged_df.dropna(), reg_type='ridge')
        
        sampled.append(merged_df.dropna())
        
sampled = pd.concat(sampled)
sampled.to_csv(f'{output_folder}regularisation.csv')


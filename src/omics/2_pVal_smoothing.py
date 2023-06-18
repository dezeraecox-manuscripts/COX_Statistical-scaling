import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import ttest_1samp

from src.utils import calculate_intercepts, pval_smoothing, fit_cdf_threshold, sigmoid, fit_sigmoid

from loguru import logger
logger.info('Import OK')

input_path = 'results/omics/compiled.csv'
output_folder = 'results/omics/'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Read in prepared data
compiled = pd.read_csv(f'{input_path}')
compiled.drop([col for col in compiled.columns.tolist()
               if 'Unnamed: ' in col], axis=1, inplace=True)

# ----------------Standard calculations----------------
# Complete t-test
mean_per_protein = []
for (dataset, protein, dtype), df in compiled.groupby(['dataset', 'Protein_IDs', 'type']):
    results_df = df.groupby(['dataset', 'Protein_IDs', 'source', 'type']).mean()[['ratio', 'log2_ratio']].reset_index()
    results_df[['tval_rawratio', 'pval_rawratio']] = list(
        ttest_1samp(df['ratio'], popmean=1))
    mean_per_protein.append(results_df)
mean_per_protein = pd.concat(mean_per_protein)

# Add -log10(pval), significance markers
mean_per_protein['-log10(pval)'] = - \
    np.log10(mean_per_protein['pval_rawratio'])
mean_per_protein['log2_meanratio'] = np.log2(mean_per_protein['ratio'])

mean_per_protein.to_csv(f'{output_folder}volcano.csv')

# --------------------Smoothing--------------------
smoothed = []
for dataset, df in compiled.groupby(['dataset']):
    penalty = df['replicate'].max() * 10
    smooth_ratios = pval_smoothing(
        df,
        sample_cols=['ratio', 'log2_ratio'],
        group_cols=['dataset', 'Protein_IDs', 'source', 'type'],
        popmean=1,
        penalty_factor=penalty,
    )
    smoothed.append(smooth_ratios.reset_index())
smoothed = pd.concat(smoothed)
smoothed['processing'] = 'smooth'

# Calculate raw values as mean of original (no smoothing)
raw = compiled.groupby(['dataset', 'Protein_IDs', 'source', 'type']).mean()[['ratio', 'log2_ratio']].reset_index()
raw['processing'] = 'raw'

# combine raw and smooth to single long-form dataframe
compiled = pd.concat([raw, smoothed])

# -------------------Thresholding-------------------
# Fit distributions to determine thresholds
thresholded = []
for (dataset, processing), df in compiled.groupby(['dataset', 'processing']):
    fig, axes = plt.subplots(2)
    thresholds = fit_cdf_threshold(
        df[df['source'] == 'C'],
        sample_col='log2_ratio',
        z_val=3,
        verbose=False,
        curve_func=sigmoid, 
        fit_func=fit_sigmoid, 
        threshold_func=calculate_intercepts, 
        val_type='log'
        # xrange_vals=(-0.5, 0.5)
    )
    df[['log2_lower_thresh', 'log2_upper_thresh', 'log2_max_thresh']] = thresholds
    sns.histplot(df['log2_ratio'], color='grey', ax=axes[0])
    axes[0].axvline(thresholds[0], color='red')
    axes[0].axvline(thresholds[1], color='red')
    
    thresholds = fit_cdf_threshold(
        df[df['source'] == 'C'],
        sample_col='ratio',
        z_val=3,
        verbose=False,
        curve_func=sigmoid, 
        fit_func=fit_sigmoid, 
        threshold_func=calculate_intercepts, 
        val_type='raw'
        # xrange_vals=(-0.5, 0.5)
    )
    df[['lower_thresh', 'upper_thresh', 'max_thresh']] = thresholds
    sns.histplot(df['ratio'], color='grey', ax=axes[1])    
    axes[1].axvline(thresholds[0], color='red')
    axes[1].axvline(thresholds[1], color='red')
    axes[1].set
    plt.suptitle(f'{dataset}: {processing}')
    plt.show()

    thresholded.append(df)
thresholded = pd.concat(thresholded)

# Threshold experimental data according to pseudo controls
thresholded['log_thresholded'] = [1 if ((val > upper_thresh) or (val < lower_thresh)) else np.nan for val, upper_thresh, lower_thresh in thresholded[['log2_ratio', 'log2_upper_thresh', 'log2_lower_thresh']].values]
thresholded['raw_thresholded'] = [1 if ((val > upper_thresh) or (val < lower_thresh)) else np.nan for val, upper_thresh, lower_thresh in thresholded[['ratio', 'upper_thresh', 'lower_thresh']].values]

thresholded_raw = thresholded[['dataset', 'Protein_IDs', 'source', 'type', 'ratio', 'processing', 'lower_thresh', 'upper_thresh', 'max_thresh', 'raw_thresholded']].copy().rename(columns={'raw_thresholded': 'thresholded'})
thresholded_raw['ratio_type'] = 'raw'
thresholded_log = thresholded[['dataset', 'Protein_IDs', 'source', 'type', 'log2_ratio',
       'processing', 'log2_lower_thresh', 'log2_upper_thresh',
       'log2_max_thresh', 'log_thresholded']].copy().rename(columns={'log2_ratio': 'ratio', 'log2_lower_thresh': 'lower_thresh', 'log2_upper_thresh': 'upper_thresh',
       'log2_max_thresh': 'max_thresh', 'log_thresholded': 'thresholded'})
thresholded_log['ratio_type'] = 'log2'

thresholded = pd.concat([thresholded_raw, thresholded_log])

# Save to excel
thresholded.to_csv(f'{output_folder}scale_thresholded.csv')

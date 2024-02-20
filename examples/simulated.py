import os
import pandas as pd
import numpy as np
from scipy.stats import ttest_1samp
import matplotlib.pyplot as plt
from loguru import logger
import seaborn as sns
import matplotlib.patches as mpatches

logger.info('Import OK')

# optional: for reproducible numbers, set numpy seed
np.random.seed(567)

# Generate two datasets with different variability
data_1 = [(val + np.random.randint(200, 400)).round(2) for val in np.random.rand(1, 5)[0]]
data_2 = [(val + np.random.randint(100, 1000)).round(2) for val in np.random.rand(1, 5)[0]]
dataset = pd.DataFrame([data_1, data_2], index=['data_1', 'data_2'])
dataset.columns = np.arange(1, 6) # Set column names to replicate number
dataset

# PERCEPT as defined in Eq. 1
def percept(m0, m1, F, p):
    return m0 + ((m0 - m1) / -(F**p))
    

# define a simple function to apply the PERCEPT scaling
def apply_percept(data, hypothethical_mean, penalty):
    
    # 1. Calculate p-value    
    tval, pval = ttest_1samp(
        data,
        popmean=hypothethical_mean,
        nan_policy='omit'
    )

    # 2. Calculate sample mean
    sample_mean = np.mean(data)
    
    # 3. Apply percept, returning scaled mean value
    return percept(
                m0=hypothethical_mean,
                m1=sample_mean,
                F=penalty,
                p=pval
                )


# Apply PERCEPT to the data
data_1_scaled_mean = apply_percept(
    data=data_1,
    hypothethical_mean=500,
    penalty=50
    )
data_1_scaled_mean

data_2_scaled_mean = apply_percept(
    data=data_2,
    hypothethical_mean=500,
    penalty=50
    )
data_2_scaled_mean

# Summarise data by adding to original dataframe
dataset['raw_mean'] = dataset.mean(axis=1).values
dataset['scaled_mean'] = [data_1_scaled_mean, data_2_scaled_mean]

# Save example to csv
dataset.to_csv('examples/simulated-dataset.csv')

# Prepare data for visualisation
summary = pd.melt(
    dataset.reset_index().rename(columns={'index': 'dataset'}),
    id_vars=['dataset'], # Keep the dataset labels
    value_vars=np.arange(1, 6), # Collect the raw data columns
    var_name='replicate' # Label the new column created 
    )
summary

# Visualise the data before and after scaling
fig, ax = plt.subplots(figsize=(5,5))
sns.stripplot(
    data=summary,
    x='dataset',
    y='value',
    color='black',
    dodge=False,
    ax=ax,
    s=15,
    label='Raw points'
)
sns.boxplot(
    data=summary.groupby('dataset').mean().reset_index(),
    x='dataset',
    y='value',
    ax=ax,
    medianprops=dict(color='#E3B504', linewidth=4),
)
  
sns.boxplot(
    data=dataset.reset_index(),
    x='index',
    y='scaled_mean',
    ax=ax,
    medianprops=dict(color='#B0185E', linewidth=4, linestyle='--'),
)

# Add line at hypothetical mean
ax.axhline(500, color='#420264', linestyle='--', linewidth=2)
ax.annotate('Hypothetical mean: 500', xy=(0.005, 0.467), xycoords="axes fraction", ha='left', va='top', color='#420264')
# Set axes labels
ax.set(ylabel='Value', xlabel='Dataset', xticklabels=['Consistent', 'Variable'])

# Customise legend
handles, labels = plt.gca().get_legend_handles_labels()
raw_patch = mpatches.Patch(color='#E3B504', label='Raw mean')
scaled_patch = mpatches.Patch(color='#B0185E', label='Scaled mean')
plt.legend(handles=[handles[0], raw_patch, scaled_patch], labels=['Raw points', 'Raw', 'Scaled'], loc='upper left')

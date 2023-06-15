import os
import pandas as pd
import numpy as np
from scipy.stats import ttest_1samp
from loguru import logger
from src.utils import pval_smoothing

logger.info('Import OK')

data_folder = 'data/simulated-data/optimisation/'

# optional: for reproducible numbers, set numpy seed
np.random.seed(1234)

if not os.path.exists(data_folder):
    os.makedirs(data_folder)

def calc_scaling_factor(penalty_factor, pval):
    return 1 / penalty_factor**pval

# -----------generate sample pvals scaled by factors-----------

high = [1] + [val for val in np.random.rand(1, 10)[0]]
medium = [val / 10 for val in np.random.rand(1, 10)[0]]
low = [val / 100 for val in np.random.rand(1, 10)[0]] + [0]

pvals = sorted(high + medium + low)

# set penalty options to test
penalty_factors = [1, 2, 5, 10, 20, 50, 100, 200, 500]
scaled_data = pd.DataFrame(penalty_factors, columns=['penalty_factor'])

# Apply pval scaling
for pval in pvals:
    scaled_data[pval] = [calc_scaling_factor(penalty, pval) for penalty in penalty_factors]

# melt df to longform
scaled_data = pd.melt(
    scaled_data, 
    id_vars=['penalty_factor'], 
    value_vars=[col for col in scaled_data.columns.tolist() if 'penalty' not in str(col)], 
    var_name='pvalue', 
    value_name='scaling_factor'
)

# Save generated datasets to csv
scaled_data.to_csv(f'{data_folder}scaled_pvals.csv')

# -------Generate example datapoints scaled by different factors-------
np.random.seed(0)

data_1 = [(val + np.random.randint(200, 400)).round(2) for val in np.random.rand(1, 5)[0]]
data_2 = [(val + np.random.randint(100, 1000)).round(2) for val in np.random.rand(1, 5)[0]]

dataset = pd.DataFrame([np.arange(1, 6), data_1, data_2], index=['replicate', 'data_1', 'data_2']).T
dataset= pd.melt(
            dataset,
            id_vars=['replicate'], 
            value_vars=['data_1', 'data_2'], 
            var_name='dataset', 
            value_name='value'
            )
factors = [0, 2, 20, 200]

smoothed_data = []
for penalty in factors:
    df = pval_smoothing(
        dataset, 
        sample_cols=['value'], 
        group_cols=['dataset'], 
        popmean=500, 
        penalty_factor=penalty, 
        )
    df['penalty_factor'] = penalty
    smoothed_data.append(df)
smoothed_data = pd.concat(smoothed_data).reset_index()


dataset_params = pd.DataFrame(
    [
        [np.mean(data_1), ttest_1samp(data_1, popmean=500)[1]],
        [np.mean(data_2), ttest_1samp(data_2, popmean=500)[1]],
    ],
    index=['data_1', 'data_2'],
    columns=['mean', 'pvalue']
).reset_index().rename(columns={'index': 'dataset'})

# Save to csv
dataset.to_csv(f'{data_folder}dataset.csv')
dataset_params.to_csv(f'{data_folder}dataset_params.csv')
smoothed_data.to_csv(f'{data_folder}smoothed_data.csv')
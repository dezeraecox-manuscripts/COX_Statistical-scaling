import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp

from loguru import logger
logger.info('Import OK')


def one_sample_ttest(compiled, sample_cols, group_cols, popmean=1):
    df = compiled.copy()
    ttest_results = []
    for group_key, df in df.groupby(group_cols):
        results = []
        for col in sample_cols:
            test_vals = df[col].values
            if len(test_vals) > 1:
                results.append(tuple(ttest_1samp(test_vals, popmean=popmean, nan_policy='omit')))
            else:
                results.append(tuple([np.nan, np.nan]))
        results = pd.DataFrame(results)
        results.columns = ['t-stat', 'p-val']
        results['conc'] = sample_cols
        for x, col in enumerate(group_cols):
            if len(group_cols) > 1:
                results[col] = group_key[x]
            else:
                results[col] = group_key
        ttest_results.append(results)
    ttest_results = pd.concat(ttest_results)
    ttest_results[['t-stat', 'p-val']] = ttest_results[['t-stat', 'p-val']].astype(float)

    return ttest_results # 5% of the points detected as significantly different


def pval_smoothing(compiled, sample_cols, group_cols, popmean, penalty_factor=20, zero_norm=False):
    """Sclae mean value proportional to pvalue, imposing penalty for variability

    Parameters
    ----------
    compiled : DataFrame
        Longoform pandas df containing descriptive columns (group_cols) and data columns (sample_cols),
        where replicates of each datapoint are stored in columns.
    sample_cols : list[str]
        List of column names where quantitative data can be found. Replicate data points should 
        be contained wholly within single columns
    group_cols : list[str]
        List of column names to group ```compiled``` of, such that grouped df for each group is 
        length of replicates 
    popmean : int
        Hypothesised population mean. Typically, for ratiometric analyses this may be 1 or 0, however 
        can be any value to which samples will be compared
    penalty_factor : int, optional
        Weight to which p-value will be scaled, by default 20. Larger value imposes more severe 
        scaling of the mean value with increased p-value.
    zero_norm : bool, optional
        When smoothing values which have been normalised to a given sample e.g. time 0 normalisation, 
        all entries for the sample used as a normalisation factor will be the same. In the case of 
        samples normalised to 0, then the normalisation sample will return NaN for the pvalue which 
        is then propogated to the final value dataframe. Setting this option to True restores the 0 
        values for samples which are all 0. By default False.

    Returns
    -------
    DataFrame
        Smoothed dataframe where replicates have been reduced to the mean value, 
        scaled by p-value smoothing.
    """    
    # Apply t-test to sample
    ttest_results = one_sample_ttest(compiled, sample_cols, group_cols=group_cols, popmean=popmean)
    # Generate scaling factors
    ttest_results['exp_p-val'] = penalty_factor**ttest_results['p-val']
    p_vals = pd.pivot_table(ttest_results, values='exp_p-val', index=group_cols, columns='conc')

    # Calculate mean of the input df
    proportional_pval = compiled.groupby(group_cols).mean().copy().sort_values(group_cols)
    for col in sample_cols:
        if col in p_vals.columns.tolist():
            # apply scaling factor to means
            proportional_pval[col] = popmean + (proportional_pval[col] - popmean) * (1 / p_vals.sort_values(group_cols)[col])
    if zero_norm == True:
        proportional_pval[0.0] = 0

    return proportional_pval


def randomised_control(df, col, group_cols):
    # General idea is to use control values for a set of proteins to define a mean and variance (SD?) that is then translated to a normal distribution from which corresponding 'faux-control' samples are taken.
    # This then enables the creation of pseudo-control vs control sample to define threshold of biological interest
    dfs = []
    for group, df in df.groupby(group_cols):
        group_vals = df[col].tolist()
        group_mean, group_variation = np.mean(group_vals), np.std(group_vals)
        distribution = np.random.normal(loc=group_mean, scale=group_variation, size=len(group_vals))
        df[f'pseudo_{col}'] = distribution

    return pd.concat(dfs)


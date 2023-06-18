import os
import functools
import seaborn as sns
import pandas as pd
import numpy as np
from itertools import combinations
import random

from scipy.stats import wilcoxon
from src.utils import pval_smoothing, one_sample_ttest, correlation_df

from loguru import logger
logger.info('Import OK')

input_path = 'results/omics/compiled.csv'
output_folder = 'results/omics/'

num_reps = 4 # number of replicates to collect when subsampling
penalty = 10*num_reps # penalty for pVal smoothing
num_combinations = 100
# Set seed for reproducibility
np.random.seed(10301)

if not os.path.exists(output_folder):
    os.makedirs(output_folder)



def fit_ecdf(x):
    x = np.sort(x)
    def result(v):
        return np.searchsorted(x, v, side = 'right') / x.size
    return result


def sample_ecdf(df, value_cols, num_points=100, method='nearest', order=False):
    
    test_vals = pd.DataFrame(np.arange(0, 1.01, (1/num_points)), columns=['ecdf'])
    test_vals['type'] = 'interpolated'

    interpolated = test_vals.copy()
    for col in value_cols:
        test_df = df.dropna().drop_duplicates(subset=[col])
        ecdf = fit_ecdf(test_df[col])
        test_df['ecdf'] = ecdf(test_df.dropna().drop_duplicates(subset=[col])[col])
        combined = pd.concat([test_df.sort_values('ecdf').dropna(), test_vals])
        combined = combined.set_index('ecdf').interpolate(method=method, order=order).reset_index()
        interpolated[col] = combined[combined['type'] == 'interpolated'].copy()[col].tolist()
        
    return interpolated


# Read in compiled data for subsampling
compiled = pd.read_csv(input_path)
compiled = compiled[compiled['type'] == 'experimental'].copy()
compiled.drop([col for col in compiled.columns.tolist() if 'Unnamed: ' in col], axis=1, inplace=True)
compiled = compiled[compiled['dataset'].isin(['Bader'])].copy()

#------------------------Sampling------------------------
# COllect subsample for each dataset and calculate smoothed and raw means per protein
if not os.path.exists(f'{output_folder}sampled_compiled.csv'):
    sampled = []
    for num_reps in [3, 5]:
        
        logger.info(f'{num_reps}: {len(compiled["replicate"].unique().tolist())}')
        # Generate all possible combination subsets
        replicates = list(combinations(compiled['replicate'].unique().tolist(), num_reps))
        logger.info(f'{num_reps} replicate combinations: {len(replicates)}')
        
        if len(replicates) < num_combinations:
            logger.info(f'{num_reps} not processed due to insufficient replicates')
            continue
        replicates = random.sample(replicates, num_combinations)
        for x, combination in enumerate(replicates):
            # Calculate raw ground truth
            raw_total = compiled.copy().groupby(['dataset', 'Protein_IDs', 'source', 'type']).mean()['log2_ratio']
                    
            # Calculate raw sampled mean with ttest
            group_cols = ['dataset', 'Protein_IDs', 'source', 'type']
            sample_cols = ['log2_ratio']
            ttest_results = one_sample_ttest(compiled[compiled['replicate'].isin(combination)], sample_cols, group_cols=group_cols, popmean=0)
            # Generate scaling factors
            ttest_results = pd.pivot_table(ttest_results, values='p-val', index=group_cols, columns='sample_col')
            ttest_results.columns = [f'pval_{col}' for col in ttest_results.columns]

            # Calculate mean of the input df
            raw_sampled = compiled[compiled['replicate'].isin(combination)].groupby(group_cols).mean()[sample_cols].copy().sort_values(group_cols)

            raw_sampled = pd.merge(raw_sampled, ttest_results, on=group_cols, how='outer')
            
            # Complete smoothing
            smooth_sampled = pval_smoothing(
                df=compiled[compiled['replicate'].isin(combination)].copy(),
                sample_cols=['log2_ratio'],
                group_cols=['dataset', 'Protein_IDs', 'source', 'type'],
                popmean=0,
                penalty_factor=penalty,
                complete=True
            )

            merged_df = functools.reduce(lambda left, right: pd.merge(left, right, on=['dataset', 'Protein_IDs', 'source', 'type'], how='outer'), [smooth_sampled.reset_index().rename(columns={'scaled_log2_ratio': 'smooth'}), raw_sampled.reset_index().rename(columns={'log2_ratio': 'raw', 'pval_log2_ratio': 'pvalue'}), raw_total.reset_index().rename(columns={'log2_ratio': 'total'})])
            merged_df['combination'] = x
            merged_df['num_reps'] = num_reps
            sampled.append(merged_df)
    sampled = pd.concat(sampled)
    sampled.to_csv(f'{output_folder}sampled_compiled.csv')
else:
    sampled = pd.read_csv(f'{output_folder}sampled_compiled.csv')
    sampled = sampled[sampled['num_reps'] == 5].copy()
    sampled.drop([col for col in sampled.columns.tolist() if 'Unnamed: ' in col], axis=1, inplace=True)

# ------------------Comparison: Correlation------------------
# Measure correlation between smooth/total and raw/total for each subsample
correlations = []
for (dataset, combination), df in sampled.groupby(['dataset', 'combination']):
    correlation_raw = correlation_df(
                    df['raw'],
                    df['total'],).T
    correlation_raw['comparison'] = 'raw'
    correlation_smooth = correlation_df(
                    df['smooth'],
                    df['total'],).T
    correlation_smooth['comparison'] = 'smooth'
    correlation = pd.concat([correlation_raw, correlation_smooth])
    correlation['dataset'] = dataset
    correlation['combination'] = combination
    correlations.append(correlation)
correlations = pd.concat(correlations)


# ------------------Comparison: KS Distribution Distance------------------
from scipy.stats import epps_singleton_2samp
ks_distances = []
for (dataset, combination), df in sampled.groupby(['dataset', 'combination']):
    test_df = df.dropna().copy()
    distance_raw = pd.DataFrame(epps_singleton_2samp(
                    test_df['raw'], test_df['total']), index=['ks_stat', 'pvalue']).T
    distance_raw['comparison'] = 'raw'
    distance_smooth = pd.DataFrame(epps_singleton_2samp(
                    test_df['smooth'], test_df['total']), index=['ks_stat', 'pvalue']).T
    distance_smooth['comparison'] = 'smooth'
    distance = pd.concat([distance_raw, distance_smooth])
    distance['dataset'] = dataset
    distance['combination'] = combination
    ks_distances.append(distance)
ks_distances = pd.concat(ks_distances)
 
# ------------------Comparison: Distance------------------
differences = []
for (dataset, combination), df in sampled.groupby(['dataset', 'combination']):
    difference_raw = pd.DataFrame([(df['total'] - df['raw']).abs().mean() ], columns=['difference'])
    difference_raw['comparison'] = 'raw'
    difference_smooth = pd.DataFrame([(df['total'] - df['smooth']).abs().mean() ], columns=['difference'])
    difference_smooth['comparison'] = 'smooth'
    difference = pd.concat([difference_raw, difference_smooth])
    difference['dataset'] = dataset
    difference['combination'] = combination
    differences.append(difference)
differences = pd.concat(differences)

# ------------------Comparison: Wilcoxon------------------
wilxocons = []
for (dataset, combination), df in sampled.groupby(['dataset', 'combination']):
    wilxocon_raw = pd.DataFrame(wilcoxon(
                    df['raw'], df['total']), index=['wilcoxon', 'pvalue']).T
    wilxocon_raw['comparison'] = 'raw'
    wilxocon_smooth = pd.DataFrame(wilcoxon(
                    df['smooth'], df['total']), index=['wilcoxon', 'pvalue']).T
    wilxocon_smooth['comparison'] = 'smooth'
    wilxocon = pd.concat([wilxocon_raw, wilxocon_smooth])
    wilxocon['dataset'] = dataset
    wilxocon['combination'] = combination
    wilxocons.append(wilxocon)
wilxocons = pd.concat(wilxocons)

# -------------------Combine distance measures-------------------
distances = functools.reduce(lambda left, right: pd.merge(left, right, on=['comparison', 'dataset', 'combination'], how='outer'), [ks_distances, differences, correlations, wilxocons])
distances = distances[['comparison', 'dataset', 'combination', 'difference', 'ks_stat', 'pearsons_r', 'spearmans_r', 'wilcoxon']].copy()
distances[['difference', 'ks_stat', 'wilcoxon']] = distances[['difference', 'ks_stat', 'wilcoxon']] / distances[distances['comparison'] == 'raw'][['difference', 'ks_stat', 'wilcoxon']].median()
distances = pd.melt(distances, id_vars=['comparison', 'dataset', 'combination'], value_vars=['difference', 'ks_stat', 'wilcoxon'], value_name='distance', var_name='measure')
distances.to_csv(f'{output_folder}distances.csv')


# ------------------Fit ECDFs------------------
# interpolate ecdf for  combined visualisation
fitted_ecdfs = []
for (dataset, combination), df in sampled.groupby(['dataset', 'combination']):
    
    fitted_ecdf = sample_ecdf(df, value_cols=['raw', 'smooth', 'total'], method='polynomial', order=3)
    fitted_ecdf['dataset'] = dataset
    fitted_ecdf['combination'] = combination
    fitted_ecdfs.append(fitted_ecdf)
fitted_ecdfs = pd.concat(fitted_ecdfs)
fitted_ecdfs.to_csv(f'{output_folder}fitted_ecdfs.csv')

## CREATING EXAMPLES FOR BADER
sampled = sampled[sampled['dataset'] == 'Bader'].copy()
sampled['-log10(pval)'] = - np.log10(sampled['pvalue'])

# Assign thresholded proteins
thresholded = []
for combination, df in sampled.groupby('combination'):
    lower, upper = np.percentile(df['raw'].dropna().values, [5, 95])
    x_range = df['raw'].dropna().abs().max() + np.max([abs(lower), abs(upper)])

    df['raw_category'] = ['significant_effected' if ((pval > 1.3) & ((val > upper) | (val < lower) )) else ('nonsignificant_effected' if ((pval < 1.3) & ((val > upper) | (val < lower) )) else 'nonsignificant_noneffected') for val, pval in df[['raw', '-log10(pval)']].values]
    df[['lower', 'upper']] = lower, upper
    thresholded.append(df)
thresholded = pd.concat(thresholded)
    
# Number of NS/A Proteins per sampled version
threshold_distribution = thresholded.groupby(['combination', 'raw_category']).count()['total'].reset_index()
threshold_distribution = pd.pivot(threshold_distribution, index=['combination'], columns='raw_category', values='total')
threshold_distribution['total'] = threshold_distribution.sum(axis=1)
threshold_distribution['%_NSA'] = threshold_distribution['nonsignificant_effected'] / threshold_distribution['total'] * 100
threshold_distribution['%_SA'] = threshold_distribution['significant_effected'] / threshold_distribution['total'] * 100

thresholded.to_csv(f'{output_folder}thresholded.csv')
threshold_distribution.to_csv(f'{output_folder}threshold_distribution.csv')

# effet size of NSA vs SA
effect_size = thresholded.copy()
effect_size['raw'] = effect_size['raw'].abs()
effect_size['smooth'] = effect_size['smooth'].abs()
effect_size['threshold'] = effect_size[['lower', 'upper']].abs().max(axis=1)
effect_size['smooth_category'] = ['smooth_effected' if ((val > upper) | (val < lower)) else 'smooth_noneffected' for val, lower, upper in effect_size[['smooth', 'lower', 'upper']].values]
    
effect_size['raw_label'] = effect_size['raw_category'].map({'nonsignificant_effected': 'NS/A', 'significant_effected': 'S/A', 'nonsignificant_noneffected': 'NS/NA'})
effect_size['smooth_label'] = effect_size['smooth_category'].map({'smooth_effected': 'A', 'smooth_noneffected': 'NA'})
effect_size['label'] = effect_size['raw_label'] + '\n' + effect_size['smooth_label']

effect_proportion = effect_size.groupby(['combination', 'raw_category', 'smooth_category']).mean()[['raw', 'smooth', 'threshold']].reset_index()
effect_proportion = pd.melt(effect_proportion, id_vars=['combination', 'raw_category', 'smooth_category', 'threshold'], value_vars=['raw', 'smooth'], value_name='effect', var_name=['processing'])
effect_proportion['raw_label'] = effect_proportion['raw_category'].map({'nonsignificant_effected': 'NS/A', 'significant_effected': 'S/A', 'nonsignificant_noneffected': 'NS/NA'})
effect_proportion['smooth_label'] = effect_proportion['smooth_category'].map({'smooth_effected': 'A', 'smooth_noneffected': 'NA'})
effect_proportion['label'] = effect_proportion['raw_label'] + '\n' + effect_proportion['smooth_label']

effect_size.to_csv(f'{output_folder}effect_size.csv')
effect_proportion.to_csv(f'{output_folder}effect_proportion.csv')

# What proportion of smoothed values are then moved back within the NSNA boundary?

proportional_change = effect_size.groupby(['combination', 'raw_category', 'smooth_category']).count()['total'].reset_index()
proportional_change = pd.pivot(proportional_change, index=['combination', 'raw_category'], columns='smooth_category', values='total').fillna(0).reset_index()
proportional_change['proportion_effected'] = proportional_change['smooth_effected'] / (proportional_change[['smooth_effected', 'smooth_noneffected']].sum(axis=1)) * 100
proportional_change['proportion_noneffected'] = proportional_change['smooth_noneffected'] / (proportional_change[['smooth_effected', 'smooth_noneffected']].sum(axis=1)) * 100
proportional_change = pd.melt(proportional_change, id_vars=['combination', 'raw_category'], value_vars=['proportion_effected', 'proportion_noneffected'], value_name='proportion', var_name='smooth_category').reset_index()
proportional_change['smooth_category'] = proportional_change['smooth_category'].str.split('_').str[-1]
proportional_change['raw_label'] = effect_proportion['raw_category'].map({'nonsignificant_effected': 'NS/A', 'significant_effected': 'S/A', 'nonsignificant_noneffected': 'NS/NA'})
proportional_change['smooth_label'] = proportional_change['smooth_category'].map({'effected': 'A', 'noneffected': 'NA'})

proportional_change.to_csv(f'{output_folder}proportional_change.csv')

# Proportion of points fall into the NS/A category with increasing numbers of repicates per sample? 
sampled = pd.read_csv(f'{output_folder}sampled_compiled.csv')
# repeat calculations for all-replicate sample
group_cols = ['dataset', 'Protein_IDs', 'source', 'type']
sample_cols = ['log2_ratio']
ttest_results = one_sample_ttest(compiled, sample_cols, group_cols=group_cols, popmean=0)
# Generate scaling factors
ttest_results = pd.pivot_table(ttest_results, values='p-val', index=group_cols, columns='sample_col')
ttest_results.columns = [f'pvalue' for col in ttest_results.columns]
# Complete smoothing
smooth_all = pval_smoothing(
    df=compiled.copy(),
    sample_cols=['log2_ratio'],
    group_cols=['dataset', 'Protein_IDs', 'source', 'type'],
    popmean=0,
    penalty_factor=penalty,
    complete=True
)
raw_all = compiled.groupby(['dataset', 'Protein_IDs', 'source', 'type']).mean()[['log2_ratio']].copy().sort_values(['dataset', 'Protein_IDs', 'source', 'type'])

merged_df = functools.reduce(lambda left, right: pd.merge(left, right, on=['dataset', 'Protein_IDs', 'source', 'type'], how='outer'), [smooth_all.reset_index().rename(columns={'scaled_log2_ratio': 'smooth'}), raw_all.reset_index().rename(columns={'log2_ratio': 'raw', 'pval_log2_ratio': 'pvalue'}), ttest_results.reset_index(), raw_all.reset_index().rename(columns={'log2_ratio': 'total'})])
merged_df['combination'] = 1
merged_df['num_reps'] = 88

rep_comparison = pd.concat([sampled[['Protein_IDs', 'raw', 'smooth', 'total', 'pvalue', 'combination', 'num_reps']], merged_df[['Protein_IDs', 'raw', 'smooth', 'total', 'pvalue', 'combination', 'num_reps']]])
rep_comparison['-log10(pval)'] = -np.log10(rep_comparison['pvalue'])

rep_thresholded = []
for (num_reps, combination), df in rep_comparison.groupby(['num_reps', 'combination']):
    lower, upper = np.percentile(df['raw'].dropna().values, [5, 95])
    x_range = df['raw'].dropna().abs().max() + np.max([abs(lower), abs(upper)])

    df['raw_category'] = ['significant_effected' if ((pval > 1.3) & ((val > upper) | (val < lower) )) else ('nonsignificant_effected' if ((pval < 1.3) & ((val > upper) | (val < lower) )) else 'nonsignificant_noneffected') for val, pval in df[['raw', '-log10(pval)']].values]
    df['smooth_category'] = ['smooth_effected' if ((val > upper) | (val < lower)) else 'smooth_noneffected' for val in df['smooth']]
    df[['lower', 'upper']] = lower, upper
    rep_thresholded.append(df)
rep_comparison = pd.concat(rep_thresholded)

# calculate proportion of proteins in each bucket per combination
proportions = (rep_comparison.groupby(['combination', 'num_reps', 'raw_category', 'smooth_category']).count()['smooth'] / rep_comparison.groupby(['combination', 'num_reps']).count()['smooth'] * 100).reset_index()
proportions['raw_label'] = proportions['raw_category'].map({'nonsignificant_effected': 'NS/A', 'significant_effected': 'S/A', 'nonsignificant_noneffected': 'NS/NA'})
proportions['smooth_label'] = proportions['smooth_category'].map({'smooth_effected': 'A', 'smooth_noneffected': 'NA'})
proportions['label'] = proportions['raw_label'] + '\n' + proportions['smooth_label']
proportions['num_reps'] = proportions['num_reps'].astype(str)


# Collect subset of proteins for heatmap
rand_combinations = list(np.random.randint(0, 100, 5))
interesting = thresholded[thresholded['raw_category'] == 'nonsignificant_effected']['Protein_IDs'].unique().tolist()

heatmap = thresholded[(thresholded['combination'].isin(rand_combinations)) & (thresholded['Protein_IDs'].isin(interesting))].copy()

heatmap = heatmap[['combination', 'Protein_IDs',
                   'total', 'raw', 'smooth']].copy()
heatmap = pd.pivot(heatmap, columns='combination', index=[
                   'Protein_IDs', 'total'], values=['raw', 'smooth']).dropna().reset_index()
heatmap.columns = [f'{i}_{j}' if i not in [
    'Protein_IDs', 'total'] else i for i, j in heatmap.columns]

rand_proteins = [heatmap['Protein_IDs'].unique().tolist()[x] for x in np.random.randint(
    0, len(heatmap['Protein_IDs'].unique().tolist())+1, 100)]
heatmap = heatmap[heatmap['Protein_IDs'].isin(rand_proteins)]
heatmap['buffer-1'] = np.nan
heatmap['buffer-2'] = np.nan
heatmap = heatmap[
    ['Protein_IDs'] +
    [col for col in heatmap.columns if 'raw' in col] +
    ['buffer-1', 'total', 'buffer-2'] +
    [col for col in heatmap.columns if 'smooth' in col]
].copy()

heatmap.to_csv(f'{output_folder}heatmap.csv')
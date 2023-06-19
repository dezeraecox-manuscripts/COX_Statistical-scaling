import urllib.parse
from io import StringIO
import urllib.request
import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp, pearsonr, spearmanr
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import seaborn as sns

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
        results['sample_col'] = sample_cols
        for x, col in enumerate(group_cols):
            if len(group_cols) > 1:
                results[col] = group_key[x]
            else:
                results[col] = group_key
        ttest_results.append(results)
    ttest_results = pd.concat(ttest_results)
    ttest_results[['t-stat', 'p-val']] = ttest_results[['t-stat', 'p-val']].astype(float)

    return ttest_results # 5% of the points detected as significantly different


def pval_smoothing(df, sample_cols, group_cols, popmean, penalty_factor=20, complete=False):
    """Scale mean value proportional to pvalue, imposing penalty for variability

    Parameters
    ----------
    df : DataFrame
        Longoform pandas df containing descriptive columns (group_cols) and data columns (sample_cols),
        where replicates of each datapoint are stored in columns.
    sample_cols : list[str]
        List of column names where quantitative data can be found. Replicate data points should 
        be contained wholly within single columns
    group_cols : list[str]
        List of column names to group ```df``` of, such that grouped df for each group is length of replicates 
    popmean : int
        Hypothesised population mean. Typically, for ratiometric analyses this may be 1 or 0, however 
        can be any value to which samples will be compared
    penalty_factor : int, optional
        Weight to which p-value will be scaled, by default 20. Larger value imposes more severe 
        scaling of the mean value with increased p-value.

    Returns
    -------
    DataFrame
        Smoothed dataframe where replicates have been reduced to the mean value, 
        scaled by p-value smoothing.
    """
    # Apply t-test to sample
    ttest_results = one_sample_ttest(
        df, sample_cols, group_cols=group_cols, popmean=popmean)
    # Generate scaling factors
    ttest_results['exp_p-val'] = penalty_factor**ttest_results['p-val']
    p_vals = pd.pivot_table(ttest_results, values='exp_p-val',
                            index=group_cols, columns='sample_col')
    p_vals.columns = [f'scalefactor_{col}' for col in p_vals.columns]

    # Calculate mean of the input df
    proportional_pval = df.groupby(group_cols).mean(
    )[sample_cols].copy().sort_values(group_cols)
    proportional_pval.columns = [
        f'mean_{col}' for col in proportional_pval.columns]

    proportional_pval = pd.merge(
        proportional_pval, p_vals, on=group_cols, how='outer')
    for col in sample_cols:
        proportional_pval[f'scaled_{col}'] = popmean + (
            proportional_pval[f'mean_{col}'] - popmean) * (1 / proportional_pval[f'scalefactor_{col}'])

    # Restoring legacy function to return only scaled values matching input df
    smoothed_vals = proportional_pval[[
        col for col in proportional_pval.columns if 'scaled' in col]].copy()
    smoothed_vals.columns = [col.replace(
        'scaled_', '') for col in smoothed_vals.columns]

    if complete:
        return proportional_pval
    else:
        return smoothed_vals


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


def uniprot_map(genes, columns=['id', 'entry_name', 'reviewed'], from_type='ACC+ID', to_type='ACC+ID', species=False):
    logger.info(f'Mapping {len(genes)} IDs from {from_type} to {to_type}')
    url = 'https://www.uniprot.org/uploadlists/'

    params = {
        'from': from_type,
        'to': to_type,
        'format': 'tab',
        'query': ' '.join(genes),
        'columns': ','.join(columns),
    }
    if species:
        params['taxon'] = species
    data = urllib.parse.urlencode(params)
    data = data.encode('utf-8')
    req = urllib.request.Request(url, data)
    with urllib.request.urlopen(req) as f:
        response = f.read()

    result = pd.read_csv(StringIO(response.decode('utf-8')), sep='\t')
    result.rename(columns={
                  [col for col in result.columns if 'yourlist' in col][0]: 'query_id'}, inplace=True)

    return result


def map_protein_accessions(dataframe, from_type='ACC+ID', to_type='ACC+ID', species=False):
    all_genes = [item for sublist in dataframe['Protein_IDs'].str.split(
        ';') for item in sublist]
    # all_genes = list({gene.split('-')[0] for gene in all_genes}) # To clean isoforms - not needed?
    mapped_genes = uniprot_map(all_genes, columns=[
                               'id', 'entry name', 'reviewed'], from_type=from_type, to_type=to_type, species=species)
    # len({item for sublist in mapped_genes['query_id'].str.split(',') for item in sublist}.symmetric_difference(set(all_genes)))
    mapped_genes['query_id'] = mapped_genes['query_id'].str.split(',')
    gene_map = dict(mapped_genes[mapped_genes['Status'] == 'reviewed'].explode(
        'query_id')[['query_id', 'Entry']].values)
    dataframe['genes'] = dataframe['Protein_IDs'].str.split(';')
    clean_proteins = []
    for entry, df in dataframe.groupby('Protein_IDs'):
        if len(df) != 1:
            logger.info(df)
            continue
        genes = list(
            {
                gene_map[gene]
                for gene in df['genes'].tolist()[0]
                if gene in gene_map
            }
        )

        if len(genes) != 1:  # only collect proteins with one unique id'd accession
            logger.info(f'Entry with {entry} not processed')
        else:
            df['Protein_IDs'] = genes[0]
            clean_proteins.append(df)
    return pd.concat(clean_proteins)


def gauss(x, H, A, mean, sigma):
    return H + A * np.exp(-(x - mean) ** 2 / (2 * sigma ** 2))


def fit_gauss(x, y):
    mean = sum(x * y) / sum(y)
    sigma = np.sqrt(sum(y * (x - mean) ** 2) / sum(y))
    popt, pcov = curve_fit(gauss, x, y, p0=[min(y), max(y), mean, sigma])
    return popt


def distribution_z_threshold(data, z_val=1.96):
    """
    Calculate the lower and upper values encompassing a given proportion of the population).

    Common vals:
    ============
    0.5: 38.2%
    1.0: 68.2%
    1.5: 86.6%
    1.96: 95%
    2.0: 95.4%
    2.5: 98.8%
    2.58: 99%
    3.0: 99.8%

    For more explanation: https://upload.wikimedia.org/wikipedia/commons/thumb/2/25/The_Normal_Distribution.svg/1024px-The_Normal_Distribution.svg.png
    """
    range_val = z_val * np.std(data)
    return np.mean(data) - range_val, np.mean(data) + range_val


def calculate_ecdf(df, col):
    """Calculates the empirical Cumulative Distribution for the dataset contained in col. The eCDF value at any specified value of the measured variable is the fraction of observations of the measured variable that are less than or equal to the specified value. More [here](https://en.wikipedia.org/wiki/Empirical_distribution_function).

    Parameters
    ----------
    df : DataFrame
        Contains data of interest, new column will be appened to df before it is returned
    col : str
        column name in which quantitative data is contained for cumulative calculation

    Returns
    -------
    DataFrame
        Input df is returned with additional 'ecdf_{col}' column containing cumulative distribution.
    """    
    
    df[f'ecdf_{col}']=[sum( df[col] <= x)/float(df[col].size) for x in df[col]]

    return df


def logistic_cdf(x, mu, sigma):
    return 1 / (1 + np.exp(-(x - mu)/sigma))


def fit_logistic_cdf(x, y):
    mu = np.mean(x)
    sigma = np.std(x)
    popt, pcov = curve_fit(logistic_cdf, x, y, p0=[mu, sigma])
    return popt


def sigmoid(x, x50, k):
    return (1) / (1 + np.exp((x50 - x)/k))


def fit_sigmoid(xdata, ydata):
    # generate parameter guesses
    x50 = np.mean(xdata)
    k = 0.5
    popt, pcov = curve_fit(sigmoid, xdata, ydata,
                           p0=[x50, k])
    return popt


def calculate_intercepts(x50, k):
    upper = ((4 * k)) + x50 - (2*k)
    lower = x50 - (2*k)
    return upper, lower


def bolt_sigmoid(x, top, bottom, x50, k):
    return bottom + (top - bottom) / (1 + np.exp((x50 - x)/k))


def fit_bolt_sigmoid(xdata, ydata):
    # generate parameter guesses
    top = np.max(ydata)
    bottom = np.min(ydata)
    x50 = np.mean(xdata)
    k = 0.5
    popt, pcov = curve_fit(bolt_sigmoid, xdata, ydata,
                           p0=[top, bottom, x50, k])
    return popt


def calculate_tlag(top, bottom, x50, k):
    upper = ((4 * k * top) / (top - bottom)) + x50 - (2*k)
    lower = ((4 * k * bottom) / (top - bottom)) + x50 - (2*k)
    return upper, lower


def plot_bolt_sigmoid(xdata, ydata):

    popt = fit_bolt_sigmoid(xdata, ydata)

    xfit = np.arange(np.min(xdata), np.max(xdata),
                     (np.max(xdata) - np.min(xdata))/1000)

    fig, ax = plt.subplots()
    plt.plot(xdata, ydata, 'ko', label='data')
    plt.plot(xfit, bolt_sigmoid(xfit, *popt), '--r', label='fit')
    plt.show()

    return popt


def calculate_curvature(xdata, ydata, xfit, yfit, visualise=False):
    ### test curvature calculation
    dx = np.gradient(xdata)  # first derivatives
    dy = np.gradient(ydata)

    d2x = np.gradient(dx)  # second derivatives
    d2y = np.gradient(dy)

    cur = np.abs(d2y) / (np.sqrt(1 + dy ** 2)) ** 1.5  # curvature

    fitted = pd.DataFrame([xfit, yfit], index=['xfit', 'yfit']).T.reset_index()
    dx_fit = np.gradient(xfit)  # first derivatives
    dy_fit = np.gradient(yfit)

    d2x_fit = np.gradient(dx_fit)  # second derivatives
    d2y_fit = np.gradient(dy_fit)

    cur_fit = np.abs(d2y_fit) / \
        (np.sqrt(1 + dy_fit ** 2)) ** 1.5  # curvature

    if visualise:
        plot_curvature(xdata, ydata, xfit, yfit, cur,
                       cur_fit, dy, dy_fit, d2y, d2y_fit)

    return cur_fit, dy, dy_fit, d2y, d2y_fit


def plot_curvature(xdata, ydata, xfit, yfit, cur, cur_fit, dy, dy_fit, d2y, d2y_fit):
    # looking at curvature calculation
    fig, ax = plt.subplots(figsize=(10, 8))

    plt.subplot(221)
    plt.plot(xdata, ydata, 'b', xfit, yfit, 'r')
    plt.title('y=f(x)')
    plt.xlim(-0.5, 0.5)
    plt.subplot(222)
    plt.plot(xdata, cur, 'b', xfit, cur_fit, 'r')
    plt.title('curvature')
    plt.xlim(-0.5, 0.5)
    plt.legend(['From Values', 'From Fit'], bbox_to_anchor=(1.5, 1.0))
    plt.subplot(223)
    plt.plot(xdata, dy, 'b', xfit, dy_fit, 'r')
    plt.title('dy/dx')
    plt.xlim(-0.5, 0.5)
    plt.subplot(224)
    plt.plot(xdata, d2y, 'b', xfit, d2y_fit, 'r')
    plt.title('d2y/dx2')
    plt.xlim(-0.5, 0.5)
    plt.tight_layout()
    plt.show()


def apply_threshold(compiled, sample_col, threshold):
    """Produces upper and lower thresholds for all samples based on their relative control sample, as specified in the thresholding_dict, and according the z-valself.

    Parameters
    ----------
    compiled : DataFrame
        contains smoothed data to be thresholded, based on control vs control or control vs pseudocontrol thresholds.
    sample_col: str
        column containing values to be thresholded
    threshold: float
        value which determines the upper and lower bounds outside which sample_col vals will be considered signficant
    Returns
    ----------
    DataFrame
        As compiled input, but with additional column containing 1 (outside threshold) or np.nan (inside threshold)
    """
    compiled['thresholded'] = [1 if abs(val) > threshold else np.nan for val in compiled[sample_col]]

    return compiled


def fit_cdf_threshold(df, sample_col, z_val, val_type='log', curve_func=bolt_sigmoid, fit_func=fit_bolt_sigmoid, threshold_func=calculate_tlag, verbose=False, xrange_vals=False):
    
    """Applies the cumulative distribution thresholding to  sample cols, generating a range of upper and lower thresholds based on linear extrpolation of the sigmoid midpoint. Similar to derivation of Tlag presented [here](https://dx.doi.org/10.1074/jbc.M116.739250)

    Parameters
    ----------
    compiled : DataFrame
        Contains quantitative sample columns which will be fitted. If grouping columns are needed,they should be the index e.g. the result of a groupby operation
    sample_col : str
        column name to be fitted with the threshold
    z_val : float
        Modified for the calculated upper and lower cdf intercepts, roughly equivalent to the standard deviation Z-score such that 2.58 should account for ~99% of the data within the computed threshold
    verbose : bool, optional
        Determines whether to generate the plot and print visualisations for each calculation, by default False

    Returns
    -------
    tuple(floats)
        contains lower, upper and summary (absolute max) thresholds based on fit val and z_val
    """
    # must not contain nans, and sigmoid fitting works better with log vals
    og_data = df[sample_col].copy()
    if val_type == 'raw':
        df[sample_col] = np.log(df[sample_col])
    df = df.dropna(subset=[sample_col])
    
    # calculate eCDF
    df = calculate_ecdf(df, sample_col)

    # sort based on original values for fitting
    data = df[[sample_col, f'ecdf_{sample_col}']].copy().sort_values(sample_col)
    # extract x and y vals
    xdata = data[f'{sample_col}'].tolist()
    ydata = data[f'ecdf_{sample_col}'].tolist()

    popt = fit_func(xdata, ydata)
    xfit = np.arange(np.min(xdata), np.max(xdata),
                        (np.max(xdata) - np.min(xdata))/10000)
    yfit = curve_func(xfit, *popt)

    # calculate threshold
    upper_lim, lower_lim = threshold_func(*popt)
    max_thresh = np.max([abs(upper_lim), abs(lower_lim)])

    if val_type == 'raw':
        # convert thresholds back to real space if provided vals were originally real
        upper, lower, max_thresh = 10**(lower_lim*z_val), 10**(upper_lim*z_val), round(10**(max_thresh*z_val), 2)
    else:
        upper, lower, max_thresh = lower_lim*z_val, upper_lim*z_val, round(max_thresh*z_val, 2)
    
    if verbose:
        fig, ax = plt.subplots()
        plt.plot(xdata, ydata, 'o', color='grey', label='data')
        plt.plot(xfit, yfit, '--', color='black', label='fit')
        ax.axvline(upper_lim, color='red')
        ax.axvline(lower_lim, color='red')
        plt.title(sample_col)
        if xrange_vals:
            plt.xlim(*xrange_vals)
        plt.show()

    
        # on the original data
        fig, ax = plt.subplots()
        sns.histplot(og_data, color='grey')
        ax.axvline(upper, color='red')
        ax.axvline(lower, color='red')
        plt.title(sample_col)
        if xrange_vals:
            plt.xlim(*xrange_vals)
        plt.show()
        
    return upper, lower, max_thresh


def correlation_df(x, y, corr_type=None):
    data = pd.DataFrame()
    data['y'] = y
    data['x'] = x

    data.replace(np.inf, np.nan, inplace=True)
    data.replace(-np.inf, np.nan, inplace=True)
    data.dropna(inplace=True)
    if len(data) < 5:
        return pd.DataFrame([np.nan, np.nan, len(data)], index=['pearsons_r', 'pearsons_pval', 'count'])
    if corr_type == 'pearsons':
        (r, p) = pearsonr(data['x'], data['y'])
        count = len(data)
        return pd.DataFrame([r, p, count], index=['pearsons_r', 'pearsons_pval', 'count'])
    elif corr_type == 'spearmans':
        (r, p) = spearmanr(data['x'], data['y'])
        count = len(data)
        return pd.DataFrame([r, p, count], index=['spearmans_r', 'spearmans_pval', 'count'])
    else:
        (pearsons_r, pearsons_p) = pearsonr(data['x'], data['y'])
        (spearmans_r, spearmans_p) = spearmanr(data['x'], data['y'])
        count = len(data)
        return pd.DataFrame([pearsons_r, pearsons_p, spearmans_r, spearmans_p, count], index=['pearsons_r', 'pearsons_pval', 'spearmans_r', 'spearmans_pval', 'count'])

# ==============Plotting==============

def volcano(df, cat_col, palette, ax=None, x_range=None, upper=None, lower=None, size=80):
    
    kws = {"s": size, "linewidth": 0.5}
    
    if not ax:
        fig, ax = plt.subplots(figsize=(10, 10))
    df[['face', 'edge', 'width']] = [[color[0], color[1], color[2]]
                            for color in df[cat_col].map(palette)]

    for dtype, config in palette.items():
        dataframe = df[df[cat_col] == dtype].copy()
        sns.scatterplot(
            data=dataframe,
            x='log2_meanratio', y='-log10(pval)',
            facecolor=config[0],
            edgecolor=config[1],
            linewidth=config[2],
            ax=ax,
            s=size
        )
    handles, labels = zip(*[
        (ax.scatter([], [], ec=edge if face == '#ffffff' else face, facecolor=face, **kws), key) for key, (face, edge, _) in palette.items()
    ])
    ax.legend(handles, labels, title='', frameon=False, loc='upper left', borderaxespad=-0.1, handletextpad=-0.5)
    
    if x_range:
        ax.set_xlim(-x_range, x_range)
    if upper and lower:
        ax.axhline(1.3, linestyle='--', color='black', linewidth=0.3)
        ax.axvline(upper, linestyle='--', color='black', linewidth=0.3)
        ax.axvline(lower, linestyle='--', color='black', linewidth=0.3)
    ax.set_ylabel('- $Log_{10}$ (p-value)')
    ax.set_xlabel('$Log_2$(Ratio)', labelpad=0.1)
    
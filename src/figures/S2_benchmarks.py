import matplotlib.patches as patches
import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import matplotlib.transforms as mtransforms
from seaborn import blend_palette

from src.utils import volcano, plot_interpolated_ecdf
from loguru import logger

logger.info('Import OK')

input_folder = 'results/omics/'
output_folder = 'results/figures/'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Subsample ecdf
distribution_all = pd.read_csv(f'{input_folder}fitted_ecdfs.csv')
distribution_all.drop([col for col in distribution_all.columns.tolist() if 'Unnamed: ' in col], axis=1, inplace=True)

# Perform Bland-Altman calculations
def mean_diff_plot(m1, m2, sd_limit=1.96, ax=None, scatter_kwds=None, mean_line_kwds=None, limit_lines_kwds=None):
    """
    Construct a Tukey/Bland-Altman Mean Difference Plot.

    Tukey's Mean Difference Plot (also known as a Bland-Altman plot) is a
    graphical method to analyze the differences between two methods of
    measurement. The mean of the measures is plotted against their difference.

    For more information see
    https://en.wikipedia.org/wiki/Bland-Altman_plot

    Parameters
    ----------
    m1 : array_like
        A 1-d array.
    m2 : array_like
        A 1-d array.
    sd_limit : float
        The limit of agreements expressed in terms of the standard deviation of
        the differences. If `md` is the mean of the differences, and `sd` is
        the standard deviation of those differences, then the limits of
        agreement that will be plotted are md +/- sd_limit * sd.
        The default of 1.96 will produce 95% confidence intervals for the means
        of the differences. If sd_limit = 0, no limits will be plotted, and
        the ylimit of the plot defaults to 3 standard deviations on either
        side of the mean.
    ax : AxesSubplot
        If `ax` is None, then a figure is created. If an axis instance is
        given, the mean difference plot is drawn on the axis.
    scatter_kwds : dict
        Options to to style the scatter plot. Accepts any keywords for the
        matplotlib Axes.scatter plotting method
    mean_line_kwds : dict
        Options to to style the scatter plot. Accepts any keywords for the
        matplotlib Axes.axhline plotting method
    limit_lines_kwds : dict
        Options to to style the scatter plot. Accepts any keywords for the
        matplotlib Axes.axhline plotting method

    Returns
    -------
    Figure
        If `ax` is None, the created figure.  Otherwise the figure to which
        `ax` is connected.

    References
    ----------
    Bland JM, Altman DG (1986). "Statistical methods for assessing agreement
    between two methods of clinical measurement"

    Examples
    --------

    Load relevant libraries.

    >>> import statsmodels.api as sm
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt

    Making a mean difference plot.

    >>> # Seed the random number generator.
    >>> # This ensures that the results below are reproducible.
    >>> np.random.seed(9999)
    >>> m1 = np.random.random(20)
    >>> m2 = np.random.random(20)
    >>> f, ax = plt.subplots(1, figsize = (8,5))
    >>> sm.graphics.mean_diff_plot(m1, m2, ax = ax)
    >>> plt.show()

    .. plot:: plots/graphics-mean_diff_plot.py
    """

    if len(m1) != len(m2):
        raise ValueError('m1 does not have the same length as m2.')
    if sd_limit < 0:
        raise ValueError(f'sd_limit ({sd_limit}) is less than 0.')

    means = np.mean([m1, m2], axis=0)
    diffs = m1 - m2
    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs, axis=0)

    scatter_kwds = scatter_kwds or {}
    if 's' not in scatter_kwds:
        scatter_kwds['s'] = 20
    mean_line_kwds = mean_line_kwds or {}
    limit_lines_kwds = limit_lines_kwds or {}
    for kwds in [mean_line_kwds, limit_lines_kwds]:
        if 'color' not in kwds:
            kwds['color'] = 'gray'
        if 'linewidth' not in kwds:
            kwds['linewidth'] = 1
    if 'linestyle' not in mean_line_kwds:
        kwds['linestyle'] = '--'
    if 'linestyle' not in limit_lines_kwds:
        kwds['linestyle'] = ':'

    ax.scatter(means, diffs, **scatter_kwds) # Plot the means against the diffs.
    ax.axhline(mean_diff, **mean_line_kwds)  # draw mean line.

    # Annotate mean line with mean difference.
    ax.annotate(f'Mean Diff.\n{np.round(mean_diff, 4)}',
                xy=(0.99, 0.5),
                horizontalalignment='right',
                verticalalignment='center',
                xycoords='axes fraction')

    if sd_limit > 0:
        half_ylim = (1.5 * sd_limit) * std_diff
        ax.set_ylim(mean_diff - half_ylim,
                    mean_diff + half_ylim)
        limit_of_agreement = sd_limit * std_diff
        lower = mean_diff - limit_of_agreement
        upper = mean_diff + limit_of_agreement
        for j, lim in enumerate([lower, upper]):
            ax.axhline(lim, **limit_lines_kwds)
        ax.annotate(f'CI: {lower:0.2g}',
                    xy=(0.99, 0.07),
                    horizontalalignment='right',
                    verticalalignment='bottom',
                    xycoords='axes fraction')
        ax.annotate(f'CI: {upper:0.2g}',
                    xy=(0.99, 0.92),
                    horizontalalignment='right',
                    xycoords='axes fraction')

    elif sd_limit == 0:
        half_ylim = 3 * std_diff
        ax.set_ylim(mean_diff - half_ylim,
                    mean_diff + half_ylim)

    ax.set_ylabel('Difference')
    ax.set_xlabel('Means')
    return ax


def plotter(df, ycol, reg_type='lasso', ax=None, palette=None):
    
    if not palette:
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

    for dtype in ['nothing', 'bottom', 'top']:
        sns.scatterplot(
            data=df[df['rank'] == dtype],
            x='total',
            y=ycol,
            color=palette[dtype],
            size='scale_size',
            sizes={'big':100, 'small':20},
            alpha=0.75,
            ax=ax,
        )



distribution_all['mean_r-t'] = distribution_all[['raw', 'total']].mean(axis=1)
distribution_all['mean_s-t'] = distribution_all[['smooth', 'total']].mean(axis=1)
distribution_all['mean_r-s'] = distribution_all[['raw', 'smooth']].mean(axis=1)

distribution_all['diff_r-t'] = distribution_all['raw'] - distribution_all['total']
distribution_all['diff_s-t'] = distribution_all['smooth'] - distribution_all['total']
distribution_all['diff_r-s'] = distribution_all['raw'] - distribution_all['smooth']


distribution_all['diff%_r-t'] = distribution_all['diff_r-t'] / distribution_all['mean_r-t'] * 100
distribution_all['diff%_s-t'] = distribution_all['diff_s-t'] / distribution_all['mean_s-t'] * 100
distribution_all['diff%_r-s'] = distribution_all['diff_r-s'] / distribution_all['mean_r-s'] * 100

regularisation = pd.read_csv(f'{input_folder}regularisation.csv')
regularisation.drop([col for col in regularisation.columns.tolist() if 'Unnamed: ' in col], axis=1, inplace=True)


# ===========Set plotting defaults===========
font = {'family': 'arial',
        'weight': 'normal',
        'size': 8}
matplotlib.rc('font', **font)
plt.rcParams['svg.fonttype'] = 'none'
cm = 1/2.54  # centimeters in inches


palette = {
    'raw': '#E3B504',
    'smooth': '#B0185E',
    'total': '#420264',
}



# Figure defaults
fig = plt.figure(figsize=(12*cm, 18*cm))
gs = GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.35)

axA = fig.add_subplot(gs[0, 0])
axB = fig.add_subplot(gs[0, 1])
axC = fig.add_subplot(gs[1, 0])
axD = fig.add_subplot(gs[1, 1])
axE = fig.add_subplot(gs[2, 0])
axF = fig.add_subplot(gs[2, 1])

axes = {
    'A': [axA,  (-0.55, -0.15)], 
    'B': [axB,  (-0.55, -0.15)], 
    'C': [axC,  (-0.55, -0.15)],
    # 'D': [axD,  (-0.55, -0.15)],
    'D': [axE,  (-0.55, -0.15)],
    # 'F': [axF,  (-0.55, -0.15)],
}

# Panel labels
for label, (ax, xy) in axes.items():
    # label physical distance to the left and up:
    trans = mtransforms.ScaledTranslation(xy[0], xy[1], fig.dpi_scale_trans)
    ax.text(0.0, 1.1, label, transform=ax.transAxes + trans,
            fontsize=12, va='bottom', fontweight='bold')

# ------------Panel A------------
mean_diff_plot(
    distribution_all['raw'], 
    distribution_all['total'], 
    sd_limit=1.96, 
    ax=axA, 
    scatter_kwds={'color': palette['raw'], 's': 5, 'alpha': 0.2}, 
    mean_line_kwds=None, 
    limit_lines_kwds=None
    )
axA.set_title('Population vs Raw', fontsize=8)

# ------------Panel B------------
mean_diff_plot(
    distribution_all['smooth'], 
    distribution_all['total'], 
    sd_limit=1.96, 
    ax=axB, 
    scatter_kwds={'color': palette['smooth'], 's': 5, 'alpha': 0.2}, 
    mean_line_kwds=None, 
    limit_lines_kwds=None
    )
axB.set_title('Population vs Scaled', fontsize=8)

for ax in [axA, axB]:
    ax.set_ylim(-0.07, 0.07)
    
# ------------Panel C/E------------
regularisation = regularisation[regularisation['combination'] == 18].copy()
for ax, reg_type in zip([axC, axE], ['lasso', 'ridge']):
    plotter(
        regularisation,
        ycol='raw',
        reg_type=reg_type,
        ax=ax,
        palette={'nothing': 'lightgrey',
                'top': palette['raw'],
                'bottom': palette['raw'],}
        )
            
    ax.set_title(f'Raw', fontsize=8)
    ax.set_ylim(-0.2, 0.2)
    ax.axhline(0, linestyle='--', color='black', linewidth=0.25)
    ax.axvline(0, linestyle='--', color='black', linewidth=0.25)
    ax.set(ylabel='Raw', xlabel='Population')
    
# ------------Panel D/F------------

for ax, reg_type in zip([axD, axF], ['lasso', 'ridge']):
    plotter(
        regularisation,
        ycol='smooth',
        reg_type=reg_type,
        ax=ax,
        palette={'nothing': 'lightgrey',
                'top': palette['smooth'],
                'bottom': palette['smooth'],}
        )
            
    ax.set_title(f'Scaled', fontsize=8)
    ax.set_ylim(-0.2, 0.2)

    ax.axhline(0, linestyle='--', color='black', linewidth=0.25)
    ax.axvline(0, linestyle='--', color='black', linewidth=0.25)
    ax.set(ylabel='Smooth', xlabel='Population')

# Figure admin
for ax in [axA, axB, axC, axD, axE, axF]:
    ax.spines[['right', 'top']].set_visible(False)
for ax in [axC, axD, axE, axF]:
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    new_labels = {'big': '$p$ < 0.05', 'small': '$p$ > 0.05',}
    ax.legend(by_label.values(), [new_labels[key] for key in by_label.keys()], labelspacing=0.2, borderaxespad=-0.3, handletextpad=0.15, frameon=False, loc=(-0.03, 0.75))
    
plt.savefig(f'{output_folder}S2_benchmarks.svg', bbox_inches='tight')
plt.show()

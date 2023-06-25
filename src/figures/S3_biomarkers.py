
import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import matplotlib.transforms as mtransforms

from loguru import logger

logger.info('Import OK')

input_folder = 'results/biomarkers/'
output_folder = 'results/figures/'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# ===========Set plotting defaults===========
font = {'family': 'arial',
        'weight': 'normal',
        'size': 8}
matplotlib.rc('font', **font)
plt.rcParams['svg.fonttype'] = 'none'
cm = 1/2.54  # centimeters in inches

# Define colours
palette = {
    'Control': 'darkgrey',
    'ALS': '#00c49a',
    'FTLD': '#b1cf5f',
    'Control_dark': 'black',
    'ALS_dark': '#009272',
    'FTLD_dark': '#738d2a',
    'raw': '#E3B504',
    'smooth': '#B0185E',
    'total': '#420264',
}

# ==============Read in datasets==============

# Normalised deltaTDP values
normalised = pd.read_csv(f'{input_folder}normalised_summary.csv')
normalised.drop([col for col in normalised.columns.tolist() if 'Unnamed: ' in col], axis=1, inplace=True)
normalised['dataset'] = normalised['dataset'].str.split('_').str[0]

# Smoothed comparisons
comparison = pd.read_csv(f'{input_folder}comparison.csv')
comparison.drop([col for col in comparison.columns.tolist() if 'Unnamed: ' in col], axis=1, inplace=True)

smoothed_means = pd.read_csv(f'{input_folder}smoothed_means.csv')
smoothed_means.drop([col for col in smoothed_means.columns.tolist() if 'Unnamed: ' in col], axis=1, inplace=True)

# ==============Generate figure==============

# Figure defaults
fig = plt.figure(figsize=(18*cm, 40*cm))

# make outer gridspec
gs = GridSpec(3, 4, figure=fig, wspace=0.5)
# make nested gridspecs
gs0 = GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[0], hspace=0.5)
gs1 = GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[1], hspace=0.5)
gs2 = GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[2], hspace=0.5)
gs3 = GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[3], hspace=0.5)
# add axes
axA = fig.add_subplot(gs0[0, 0])
axB = fig.add_subplot(gs0[1, 0])
axC = fig.add_subplot(gs0[2, 0])
axD = fig.add_subplot(gs1[0, 0])
axE = fig.add_subplot(gs1[1, 0])
axF = fig.add_subplot(gs1[2, 0])
axG1 = fig.add_subplot(gs2[0, 0])
axG2 = fig.add_subplot(gs2[1, 0])
axG3 = fig.add_subplot(gs2[2, 0])
axH1 = fig.add_subplot(gs3[0, 0])
axH2 = fig.add_subplot(gs3[1, 0])
axH3 = fig.add_subplot(gs3[2, 0])

axes = {
    'A': [axA,  (-0.5, -0.15)],
    'B': [axB,  (-0.5, -0.15)],
    'C': [axC,  (-0.5, -0.15)],
    'D': [axD,  (-0.5, -0.15)],
    'E': [axE,  (-0.5, -0.15)],
    'F': [axF,  (-0.5, -0.15)],
    'G': [axG1,  (-0.5, -0.15)],
    'H': [axG2,  (-0.5, -0.15)],
    'I': [axG3,  (-0.5, -0.15)],
    'J': [axH1,  (-0.5, -0.15)],
    'K': [axH2,  (-0.5, -0.15)],
    'L': [axH3,  (-0.5, -0.15)],
}

# Panel labels
for label, (ax, xy) in axes.items():
    # label physical distance to the left and up:
    trans = mtransforms.ScaledTranslation(xy[0], xy[1], fig.dpi_scale_trans)
    ax.text(0.0, 1.1, label, transform=ax.transAxes + trans,
            fontsize=12, va='bottom', fontweight='bold')

# ------------Panel A-F------------
locs = {
    'Noto': axA,
    'Kasai': axB,
    'Hosokawa': axC,
    'Bourbouli': axD,
    'Kuiperij': axE,
    'Suarez-Calvet': axF,
}

for dataset, ax in locs.items():
    df = normalised[normalised['dataset'] == dataset].copy()
    sns.boxplot(
        data=df,
        x='label',
        y='norm_value',
        order=['Control', 'ALS', 'FTLD'],
        color='white',
        zorder=100000000,
        boxprops=dict(facecolor=(0,0,0,0)),
        ax=ax,
        fliersize=0,
        linewidth=0.5
    )
    sns.stripplot(
        data=df,
        x='label',
        y='norm_value',
        color='label',
        order=['Control', 'ALS', 'FTLD'],
        palette=palette,
        dodge=True,
        alpha=0.7,
        s=5,
        ax=ax,
        zorder=1,
    )
    ax.set_title(f'{dataset}', fontsize=8)
    ax.set_ylabel('ΔTDP-43 (ng/ml)')
    ax.set_xlabel('')
    ax.set_ylim(-0.2, 4.7)
    ax.legend('', frameon=False)
    ax.axhline(1, color='lightgrey', linewidth=0.5, linestyle='--')
    


# ------------Panel G------------
locs = {
    'Control': axG1,
    'ALS': axG2,
    'FTLD': axG3,
}
for source, ax in locs.items():
    df = normalised[normalised['label'] == source].copy()
    sns.histplot(
        data=df,
        x='norm_value',
        kde=True,
        bins=np.arange(0, 4, 0.25),
        ax=ax,
        color=palette[source],
        line_kws={'color': palette[f'{source}_dark']}
    ) 
    ax.axvline(df.groupby('label').mean().reset_index()['norm_value'].tolist()[0], color='#4B1D3F', linewidth=1, linestyle='--')
    ax.set_xlim(0,4)
    if source == 'FTLD':
        ax.set_xlabel('ΔTDP-43 (ng/ml)')
    else:
        ax.set_xlabel('')
    ax.annotate(source, xy=(0, 1.1), xycoords='axes fraction', color=palette[f'{source}_dark'], )

# ------------Panel H------------
locs = {
    'Control': axH1,
    'ALS': axH2,
    'FTLD': axH3,
}
for source, ax in locs.items():
    df = comparison[comparison['label'] == source].copy()
    means = normalised[normalised['label'] == source].copy().mean()['norm_value']
    smooth = dict(smoothed_means[smoothed_means['label'] == source][['data_type', 'norm_value']].values)
    sns.stripplot(
        data=df,
        x='label',
        y='raw_norm_mean',
        color='label',
        palette=palette,
        dodge=True,
        alpha=0.7,
        s=5,
        ax=ax,
        zorder=100000,
    )
    sns.barplot(
        data=df,
        x='label',
        y='raw_norm_mean',
        color='white',
        ci='sd',
        errcolor=palette['raw'],
        errwidth=0.5,
        capsize=0.1,
        ax=ax,
        zorder=0
    )

    ax.plot([-0.1, 0.1], [means, means], color=palette['total'], linestyle='--', linewidth=1, label='Pop.')
    ax.plot([-0.1, 0.1], [df['raw_norm_mean'].mean(), df['raw_norm_mean'].mean()], color=palette['raw'], linestyle='-', linewidth=1, label='Raw')
    ax.plot([-0.1, 0.1], [smooth['scaled'], smooth['scaled']], color=palette['smooth'], linestyle='-.', linewidth=1, label='Scaled')

    ax.set_ylim(-0.1, 2.1)
    ax.set_xlim(-0.25, 0.25)
    ax.axhline(1, linestyle='--', color='grey')
    ax.set_ylabel('ΔTDP-43 (ng/ml)')
    ax.set_xlabel('')
    if source == 'FTLD':
        y = 0.65
    else:
        y = -0.02
    ax.legend(handlelength=1.5, labelspacing=0.1, borderaxespad=-0.3, handletextpad=0.15, frameon=False, loc=(0.03, y), fontsize=8)


# Figure admin
for ax in [axA, axB, axC, axD, axE, axF, axG1, axG2, axG3, axH1, axH2, axH3,]:
    ax.spines[['right', 'top']].set_visible(False)

plt.savefig(f'{output_folder}S3_Biomarkers.svg', bbox_inches='tight')
plt.show()

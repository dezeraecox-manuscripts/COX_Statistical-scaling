
import matplotlib.patches as patches
import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import matplotlib.transforms as mtransforms
import ptitprince as pt

from src.utils import volcano

from loguru import logger

logger.info('Import OK')

input_folder = 'results/omics/'
output_folder = 'results/figures/'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# ===========Set plotting defaults===========
font = {'family': 'arial',
        'weight': 'normal',
        'size': 8}
matplotlib.rc('font', **font)
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['figure.dpi'] = 300
cm = 1/2.54  # centimeters in inches

# Define colours
palette = {
    'AD_raw': '#fdcd8d',
    'AD_smooth': '#e08b1b',
    'ALS_raw': '#76e2cb',
    'ALS_smooth': '#00c49a',
    'COVID_raw': '#8aa5ba',
    'COVID_smooth': '#33658a',
    'NS/NA': '#b3b3b3'
}

# ==============Read in datasets==============
# Volcano data
complete_volcano = pd.read_csv(f'{input_folder}volcano.csv')
complete_volcano.drop([col for col in complete_volcano.columns.tolist() if 'Unnamed: ' in col], axis=1, inplace=True)
complete_volcano = complete_volcano[complete_volcano['type'] == 'experimental'].copy().dropna()

compiled_volcs = []
for dataset, df in complete_volcano.groupby('dataset'):
    lower, upper = np.percentile(
        df['log2_meanratio'].values, [5, 95])
    x_range = df['log2_meanratio'].abs().max() + np.max([abs(lower), abs(upper)])
    df['upper'] = upper
    df['lower'] = lower
    df['category'] = ['S/A' if ((pval > 1.3) & ((val > upper) | (val < lower))) else ('NS/A' if ((pval < 1.3) & (
        (val > upper) | (val < lower))) else 'NS/NA') for val, pval in df[['log2_meanratio', '-log10(pval)']].values]
    compiled_volcs.append(df)
compiled_volcs = pd.concat(compiled_volcs)


# Distributions
distribution = pd.read_csv(f'{input_folder}scaled.csv')
distribution.drop([col for col in distribution.columns.tolist() if 'Unnamed: ' in col], axis=1, inplace=True)

comparison_set = ['Bader', 'Bai', 'Bereman', 'Collins', 'Di', 'DAlessandro']
distribution = distribution[
    (distribution['dataset'].isin(comparison_set)) &
    (distribution['ratio_type'] == 'log2')
    ].copy()

distribution['color'] = [f'{source}_{process}' for process, source in distribution[['processing', 'source']].values]

# Correlations
correlations = pd.read_csv(f'{input_folder}correlations.csv')
correlations.drop([col for col in correlations.columns.tolist() if 'Unnamed: ' in col], axis=1, inplace=True)

correlations = correlations[(correlations['ratio_type'] == 'log2') & (correlations['filter_type'] == 'none')].copy()

correlations['comparison_type'] = [0 if comp == '0' else 1 for comp in correlations['comparison_type']]
correlations = correlations.groupby(['dataset_a', 'comparison_type', 'processing']).mean()[['pearsons_r', 'spearmans_r']].reset_index()
correlations['key'] = [f'{process}_{comp}' for process, comp in correlations[['processing', 'comparison_type']].values]

treatments = dict(compiled_volcs[['dataset', 'source']].values)
correlations['source'] = correlations['dataset_a'].map(treatments)
correlations['color'] = [f'{source}_{process}' for process, source in correlations[['processing', 'source']].values]

# ==============Generate figure==============

# Figure defaults
fig = plt.figure(figsize=(18*cm, 40*cm))

# make outer gridspec
gs = GridSpec(3, 4, figure=fig, wspace=0.5)

# make nested gridspecs
gs0 = GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[0], hspace=0.62, )
gs1 = GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[1], hspace=0.62, )
gs2 = GridSpecFromSubplotSpec(18, 1, subplot_spec=gs[2], hspace=-0.2, )
gs3 = GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[3], hspace=0.62, )
# add axes
axA = fig.add_subplot(gs0[0, 0])
axC = fig.add_subplot(gs0[1, 0])
axE = fig.add_subplot(gs0[2, 0])
axB = fig.add_subplot(gs1[0, 0])
axD = fig.add_subplot(gs1[1, 0])
axF = fig.add_subplot(gs1[2, 0])
axG = fig.add_subplot(gs2[:, :])
axG1 = fig.add_subplot(gs2[0:3, 0])
axH = fig.add_subplot(gs3[0, 0])
axI = fig.add_subplot(gs3[1, 0])
axJ = fig.add_subplot(gs3[2, 0])

axes = {
    'A': [axA,  (-0.35, -0.025)],
    'B': [axB,  (-0.35, -0.025)],
    'C': [axC,  (-0.35, -0.025)],
    'D': [axD,  (-0.35, -0.025)],
    'E': [axE,  (-0.35, -0.025)],
    'F': [axF,  (-0.35, -0.025)],
    'G': [axG1,  (-0.2, -0.015)],
    'H': [axH,  (-0.47, -0.025)],
    'I': [axI,  (-0.47, -0.025)],
    'J': [axJ,  (-0.47, -0.025)],
}

# Panel labels
for label, (ax, xy) in axes.items():
    # label physical distance to the left and up:
    trans = mtransforms.ScaledTranslation(xy[0], xy[1], fig.dpi_scale_trans)
    ax.text(0.0, 1.1, label, transform=ax.transAxes + trans,
            fontsize=12, va='bottom', fontweight='bold')

# ------------Panel A-F------------
compiled_volcs
order = {'Bader': axA, 'Bai': axB, 'Bereman': axC,
         'Collins': axD, 'DAlessandro': axE, 'Di': axF, }

for (dataset, source), df in compiled_volcs.groupby(['dataset', 'source']):
    color = palette[f'{source}_smooth']
    ax = order[dataset]
    upper, lower = df[['upper', 'lower']].drop_duplicates().values[0]
    x_range = df['log2_meanratio'].abs().max() + 2*np.max([abs(lower), abs(upper)])
    
    volc_config = {
        'NS/NA': [palette['NS/NA'], '#ffffff', 0.1],
        'NS/A': ['#ffffff', color, 0.5],
        'S/A': [color, '#ffffff', 0.1],
    }
    if dataset in ['DAlessandro', 'Bai']:
        legend_loc = 'upper right'
    else:
        legend_loc = 'upper left'
        
    volcano(
        df=df,
        cat_col='category',
        palette=volc_config,
        ax=ax,
        x_range=x_range,
        upper=upper,
        lower=lower,
        size=10,
        legend_loc=legend_loc,
    )
    if dataset == 'Collins':
        ax.set_ylim(-2, 12)
    ax.set_title(dataset)

# ------------Panel G------------
axG1.axis('off')

y_pos = {
    'Bader' : 2,
    'Bai' : 4,
    'Bereman' : 6,
    'Collins': 8,
    'DAlessandro' : 10,
    'Di' : 12,
}

reps = {
    'Bereman' : 32,
    'Collins' : 9,
    'Bader' : 88,
    'Bai' : 8,
    'Di' : 23,
    'DAlessandro' : 33
}

distribution['box_y'] = [y_pos[f'{dataset}'] + 0.2 if dtype == 'smooth' else y_pos[f'{dataset}'] - 0.2 for dtype, dataset in distribution[['processing', 'dataset']].values]
distribution['ridge_y'] = [y_pos[f'{dataset}'] - 0.4 if dtype == 'smooth' else y_pos[f'{dataset}'] - 0.8 for dtype, dataset in distribution[['processing', 'dataset']].values]


# fig, ax = plt.subplots(1, 1, figsize=(5, 20))
sns.stripplot(
    data=distribution,
    x='scaled_ratio',
    y='box_y',
    hue='color',
    palette=palette,
    alpha=0.3,
    orient='h',
    hue_order=palette,
    dodge=False,
    order=[round(x, 1) for x in np.arange(0, 12.5, 0.2)],
    zorder=0,
    ax=axG,
    s=2
    )
sns.boxplot(
    data=distribution,
    x='scaled_ratio',
    y='box_y',
    # hue='color',
    # palette=palette,
    dodge=False,
    orient='h',
    fliersize=0,
    linewidth=0.5,
    order=[round(x, 1) for x in np.arange(0, 12.5, 0.2)],
    width=1.2,
    color='white',
    zorder=1000000,
    boxprops=dict(facecolor=(0,0,0,0)),
    ax=axG
)
for (dataset, group, dtype), df in distribution.groupby(['dataset', 'color', 'processing']):
    sns.violinplot(
        data=df,
        x='scaled_ratio',
        y='ridge_y',
        orient='h',
        inner=None,
        order=[round(x, 1) for x in np.arange(0, 12.5, 0.2)],
        width=5,
        hue=True,
        hue_order=[True, False], 
        split=True,
        palette={True: palette[group], False: palette[group]},
        facecolor='white',
        linewidth=0,
        ax=axG
    )
    y = 5.5 if dtype == 'raw' else 7.5
    name = 'Raw' if dtype == 'raw' else 'Scaled'
    axG.annotate(name, xy=(-1.15, y+(dict(zip(y_pos.keys(), np.arange(0, len(y_pos))))[dataset]*10)), color=palette[group], fontsize=5)
    axG.annotate(f'$n$ = {reps[dataset]}', xy=(1.15, 4.5+(dict(zip(y_pos.keys(),
                 np.arange(0, len(y_pos))))[dataset]*10)), color='black', fontsize=5, ha='right')
axG.legend('', frameon=False)
axG.set_yticks([8.5, 18.5, 28.5, 38.5, 48.5, 58.5, ])
axG.set_yticklabels(list(y_pos.keys()), rotation=90, ha='center', rotation_mode='anchor')
axG.set_ylabel('')
axG.set_xlabel('Normalised Log$_{2}$ Ratio', labelpad=0.2)
axG.set_ylim(63, 3.5)
axG.axvline(0, color='lightgrey', linestyle='--', linewidth=0.5)
axG.tick_params(axis='y',  pad=5)
# ------------Panel H-J------------

markers = {
    'Bader': '',
    'Bai': (1, 1),
    'Bereman': '',
    'Collins': (1, 1),
    'DAlessandro': '',
    'Di': (1, 1),
}

ax_order = {'AD': axH, 'ALS': axI, 'COVID': axJ, }

palette.update({
    'Bader': palette['AD_smooth'], 
    'Bai': palette['AD_smooth'], 
    'Bereman': palette['ALS_smooth'], 
    'Collins': palette['ALS_smooth'], 
    'Di': palette['COVID_smooth'],
    'DAlessandro': palette['COVID_smooth'], 
                })

correlations['x_pos'] = correlations['key'].map({'raw_0':0, 'smooth_0':1, 'raw_1':2, 'smooth_1':3})

for source, data in correlations.groupby('source'):
    ax = ax_order[source]
    sns.stripplot(
        data=data,
        x='key',
        y='pearsons_r',
        hue='color',
        order=['raw_0', 'smooth_0', 'raw_1', 'smooth_1'],
        palette=palette,
        s=5,
        jitter=False,
        ax=ax
    )
    for comp, df in data.groupby('comparison_type'):
        sns.lineplot(
            data=df,
            x='x_pos',
            y='pearsons_r',
            hue='dataset_a',
            palette=palette,
            style='dataset_a',
            style_order=markers.keys(),
            dashes=markers,
            linewidth=2,
            ax=ax
        )

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    by_label = {key: value for key, value in by_label.items() if key in data['dataset_a'].unique().tolist()}
    
    ax.legend(list(by_label.values()), list(by_label.keys()),  handlelength=1.5, labelspacing=0.2, borderaxespad=-0.3, handletextpad=0.15, frameon=False, loc=(0.03, 0.75))
    ax.set_xlabel('')
    ax.set_ylabel("Correlation ($R$)")
    ax.set_ylim(-0.24, 0.48)
    ax.set_yticks([round(val, 1) for val in np.arange(-0.2, 0.47, 0.2)], labels=[round(val, 1) for val in np.arange(-0.2, 0.47, 0.2)])
    ax.set_xticklabels(['Raw', 'Scaled', 'Raw', 'Scaled'])
    
    ax.annotate('Outside', xy=(0.5, -0.50), annotation_clip=False, ha='center')
    ax.annotate('Inside', xy=(2.5, -0.50), annotation_clip=False, ha='center')

# Figure admin
for ax in [axA, axB, axC, axD, axE, axF, axG, axH, axI, axJ]:
    ax.spines[['right', 'top']].set_visible(False)
    
plt.savefig(f'{output_folder}S2_Omics.svg', bbox_inches='tight')
plt.show()

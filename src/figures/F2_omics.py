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
cm = 1/2.54  # centimeters in inches

# ==============Read in datasets==============

# Complete volcano
complete_volcano = pd.read_csv(f'{input_folder}volcano.csv')
complete_volcano = complete_volcano[complete_volcano['dataset'] == 'Bader'].copy().dropna()

# Subsampled volcano
sampled_volc = pd.read_csv(f'{input_folder}thresholded.csv')
sampled_volc_rep = sampled_volc[sampled_volc['combination'] == 84].copy()

# Effect size
effect = pd.read_csv(f'{input_folder}effect_size.csv')
effect = effect[effect['combination'] == 84].copy()
effect['log2_meanratio'] = effect['raw']

# Subsample ecdf
distribution_all = pd.read_csv(f'{input_folder}fitted_ecdfs.csv')

# Replicate ecdf
distribution_rep = distribution_all[distribution_all['combination'] == 84].copy()

# Distance
distances = pd.read_csv(f'{input_folder}distances.csv')
distances = distances[distances['measure'] == 'difference'].copy()
# Heatmap
heatmap = pd.read_csv(f'{input_folder}heatmap.csv')


# Clean dfs
for df in [complete_volcano, sampled_volc, effect, distribution_all, distribution_rep, distances, heatmap]:
    df.drop([col for col in df.columns.tolist() if 'Unnamed: ' in col], axis=1, inplace=True)
    
# ==============Define functions==============

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
    


# ==============Generate figure==============

# Figure defaults
fig = plt.figure(figsize=(18*cm, 18*cm))

# make outer gridspec
gs = GridSpec(2, 1, figure=fig, height_ratios=[0.55, 0.45], hspace=0.2)
# make nested gridspecs
gs0 = GridSpecFromSubplotSpec(2, 3, subplot_spec=gs[0], wspace=0.5, hspace=0.4)
gs1 = GridSpecFromSubplotSpec(1, 12, subplot_spec=gs[1], hspace=-1)
# add axes
axA = fig.add_subplot(gs0[0, 0])
axB = fig.add_subplot(gs0[0, 1])
axC = fig.add_subplot(gs0[0, 2])
axD = fig.add_subplot(gs0[1, 0])
axE = fig.add_subplot(gs0[1, 1])
axF = fig.add_subplot(gs0[1, 2])
axG2 = fig.add_subplot(gs1[0, :11])
axG1 = fig.add_subplot(gs1[0, :11])

axes = {
    'A': [axA,  (-0.5, -0.15)],
    'B': [axB,  (-0.53, -0.15)],
    'C': [axC,  (-0.53, -0.15)],
    'D': [axD,  (-0.5, -0.15)],
    'E': [axE,  (-0.53, -0.15)],
    'F': [axF,  (-0.53, -0.15)],
    'G': [axG1,  (-0.5, -0.25)],
}

# Panel labels
for label, (ax, xy) in axes.items():
    # label physical distance to the left and up:
    trans = mtransforms.ScaledTranslation(xy[0], xy[1], fig.dpi_scale_trans)
    ax.text(0.0, 1.1, label, transform=ax.transAxes + trans,
            fontsize=12, va='bottom', fontweight='bold')

# ------------Panel A------------
df = complete_volcano[complete_volcano['type'] == 'experimental'].copy()
lower, upper = np.percentile(df['log2_meanratio'].values, [5, 95])

x_range = df['log2_meanratio'].abs().max() + np.max([abs(lower), abs(upper) ])
df['category'] = ['S/A' if ((pval > 1.3) & ((val > upper) | (val < lower) )) else ('NS/A' if ((pval < 1.3) & ((val > upper) | (val < lower) )) else 'NS/NA') for val, pval in df[['log2_meanratio', '-log10(pval)']].values]

volc_palette = {
    'NS/NA': ['#b0b4b4', '#ffffff', 0.1],
    'NS/A': ['#ffffff', '#000000', 0.5],
    'S/A': ['#000000', '#ffffff', 0.1],
}

volcano(df=df, cat_col='category', palette=volc_palette, ax=axA, x_range=x_range, upper=upper, lower=lower, size=10)

# ------------Panel B------------
sampled_volc_rep['log2_meanratio'] = sampled_volc_rep['mean_log2_ratio']
sampled_volc_rep.dropna(inplace=True)
lower, upper = sampled_volc_rep[['lower', 'upper']].values[0]
x_range = sampled_volc_rep['log2_meanratio'].abs().max() + np.max([abs(lower), abs(upper)])

sampled_volc_rep['category'] = ['S/A' if ((pval > 1.3) & ((val > upper) | (val < lower))) else ('NS/A' if ((pval < 1.3) & (
    (val > upper) | (val < lower))) else 'NS/NA') for val, pval in sampled_volc_rep[['log2_meanratio', '-log10(pval)']].values]

volc_palette = {
    'NS/NA': ['#b0b4b4', '#ffffff', 0.1],
    'NS/A': ['#ffffff', '#000000', 0.5],
    'S/A': ['#000000', '#ffffff', 0.1],
}

volcano(df=sampled_volc_rep, cat_col='category', palette=volc_palette, ax=axB, x_range=x_range, upper=upper, lower=lower, size=10)

# ------------Panel C------------
lower, upper = effect[['lower', 'upper']].values[0]

effect['category'] = ['S/A' if ((pval > 1.3) & ((val > upper) | (val < lower))) else ('NS/A' if ((pval < 1.3) & (
    (val > upper) | (val < lower))) else 'NS/NA') for val, pval in effect[['log2_meanratio', '-log10(pval)']].values]

for dtype, df in effect.dropna(subset=['log2_meanratio', '-log10(pval)']).groupby('raw_label'):
    sns.stripplot(
        data=df,
        x='raw_label', y='log2_meanratio',
        color=volc_palette[dtype][0],
        edgecolor=volc_palette[dtype][1],
        linewidth=volc_palette[dtype][2],
        order=['S/A', 'NS/A', 'NS/NA'],
        ax=axC,
        jitter=True,
        zorder=0
    )
    sns.boxplot(
        data=df,
        x='raw_label', y='log2_meanratio',
        color='white',
        order=['S/A', 'NS/A', 'NS/NA'],
        ax=axC,
        zorder=100000,
        fliersize=0,
        boxprops=dict(facecolor=(0,0,0,0))
    )

axC.axhline(np.max([abs(lower), abs(upper)]), linestyle='--', color='black', linewidth=0.3)
axC.set_ylabel('Effect size')
axC.set_xlabel('Category', labelpad=0.2)

# ------------Panel D------------

palette = {
    'raw': '#E3B504',
    'smooth': '#B0185E',
    'total': '#420264',
}

for dtype, col in zip(['Population', 'Raw', 'Scaled'], ['total', 'raw', 'smooth']):
    if col == 'total':
        linestyle='--'
    else:
        linestyle=None
    sns.lineplot(
        data=distribution_rep.dropna(),
        x=col,
        y='ecdf',
        color=palette[col],
        linewidth=0.5,
        label=dtype,
        ax=axD,
        linestyle=linestyle
    )

axD.set_xlabel('Log$_2$(Ratio)')
axD.set_ylabel('Proportion')
axD.legend(loc = 'lower right', handletextpad=0.2, frameon=False, handlelength=1)

# ------------Panel E------------

for dtype, col in zip(['Population', 'Raw', 'Scaled'], ['total', 'raw', 'smooth']):
    if col == 'total':
        linestyle='--'
    else:
        linestyle=None
    sns.lineplot(
        data=distribution_all.dropna(),
        x='ecdf',
        y=col,
        color=palette[col],
        linewidth=0.5,
        label=dtype,
        ax=axE,
        linestyle=linestyle
    )
axE.set_ylim(-0.12, 0.12)
axE.set_xlim(0, 1)
axE.set_ylabel('Log$_2$(Ratio)', labelpad=1)
axE.set_xlabel('Proportion')
axE.legend(loc = 'upper left', handletextpad=0.2, frameon=False, handlelength=1)


# ------------Panel F------------
distances

sns.boxplot(
    data=distances,
    x='comparison',
    y='distance',
    color='white',
    ax=axF,        
    zorder=100000,
    fliersize=0,
    boxprops=dict(facecolor=(0,0,0,0)),
    order=['raw', 'smooth']
)
sns.stripplot(
    data=distances,
    x='comparison',
    y='distance',
    color='comparison',
    palette=palette,
    ax=axF,
    alpha=0.4,
    zorder=0,
    order=['raw', 'smooth']
)

axF.set_xticklabels(['Raw', 'Scaled'])
axF.set_ylabel('Relative mean distance to pop.')
axF.set_xlabel('')


# ------------Panel G------------

coolors = {'#4f3130': 1, '#644a4a': 2, '#796263': 3, '#d6d0d4': 4, '#f4f4f9': 5, '#e9aa95': 6, '#e38563': 7, '#e0734a': 8, '#dd6031': 9, }

def color_order(color):
    return coolors[color]

palette = sorted([item for sublist in [list(coolors)[:3]*4] + [list(coolors)[3:6]] + [list(coolors)[6:]*4] for item in sublist], key=color_order)

palette = blend_palette(palette, n_colors=100, as_cmap=True)

hm = sns.heatmap(
    heatmap.drop('Protein_IDs', axis=1).sort_values('total', ascending=False).T,
    center=0,
    cmap=palette,
    vmin=-0.2, vmax=0.2,
    cbar=False,
    ax=axG1
)

cax = axG2.inset_axes([1.06, 0, 0.02, 1])
cbar = fig.colorbar(axG1.collections[0], ax=axG2, cax=cax)
cbar.set_ticklabels(['-0.2', '', '-0.1', '',  '0','', '0.1', '', '0.2'], rotation=-90)
cax.set_ylabel('Log$_2$ Ratio', rotation=-90, labelpad=13)
cax.yaxis.set_ticks_position('left')

axG1.axis('off')
axG2.axis('off')


boxes = [
    (0, 0, 91, 5, '#E3B504', 'Raw'), #raw
    (0, 6, 91, 1, '#420264', 'Pop.'), #total
    (0, 8, 91, 5, '#B0185E', 'Scaled'), #smooth
]

for x, y, width, height, color, label in boxes:
    # Create a Rectangle patch
    rect = patches.Rectangle((x, y), width, height, linewidth=2, edgecolor=color, facecolor='none', clip_on=False)
    # Add the patch to the Axes
    axG1.add_patch(rect)
    
    axG1.annotate(label, xy=(-3, y+(height/2)+0.5), rotation=90, annotation_clip=False, ha='center')
    
# Figure admin
for ax in [axA, axB, axC, axD, axE, axF]:
    ax.spines[['right', 'top']].set_visible(False)
    
# gs.tight_layout(fig)
plt.savefig(f'{output_folder}F2_Omics.svg')
plt.show()
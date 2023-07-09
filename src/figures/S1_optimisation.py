import os, re
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import matplotlib.transforms as mtransforms

from loguru import logger

logger.info('Import OK')

input_folder = 'data/simulated-data/'
output_folder = 'results/figures/'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# ===========Set plotting defaults===========
font = {'family' : 'arial',
'weight' : 'normal',
'size'   : 8 }
matplotlib.rc('font', **font)
plt.rcParams['svg.fonttype'] = 'none'
cm = 1/2.54  # centimeters in inches

# Define colours
palette = {
    0: 'black',
    1: 'lightgrey', 
    2: 'skyblue', 
    5: 'darkorange', 
    10: 'darkgrey', 
    20: 'royalblue', 
    50: 'orangered', 
    100: 'dimgray', 
    200: 'darkblue', 
    500: 'firebrick'
}

# ==============Read in datasets==============
# Read in scaled penalty factors data
scaled_pvals = pd.read_csv(f'{input_folder}optimisation/scaled_pvals.csv')
scaled_pvals.drop([col for col in scaled_pvals.columns.tolist() if 'Unnamed: ' in col], axis=1, inplace=True)

dataset = pd.read_csv(f'{input_folder}optimisation/dataset.csv')
dataset.drop([col for col in dataset.columns.tolist() if 'Unnamed: ' in col], axis=1, inplace=True)

params = pd.read_csv(f'{input_folder}optimisation/dataset_params.csv')
params.drop([col for col in params.columns.tolist() if 'Unnamed: ' in col], axis=1, inplace=True)

smoothed_data = pd.read_csv(f'{input_folder}optimisation/smoothed_data.csv')
smoothed_data.drop([col for col in smoothed_data.columns.tolist() if 'Unnamed: ' in col], axis=1, inplace=True)
factors = sorted(smoothed_data['penalty_factor'].unique().tolist())

# ===========Read in image examples===========

summary = pd.read_csv(f'{input_folder}images/smoothed_summary.csv')
summary.drop([col for col in summary.columns.tolist() if 'Unnamed: ' in col], axis=1, inplace=True)

# ==============Generate figure==============
# Figure defaults
fig = plt.figure(figsize=(18*cm, 11*cm))
                            
# make outer gridspec
gs = GridSpec(2, 1, figure=fig, height_ratios = [1, 1], hspace=0.1) 
# make nested gridspecs
gs1 = GridSpecFromSubplotSpec(1, 2, subplot_spec = gs[0], wspace = 0.5)
gs2 = GridSpecFromSubplotSpec(1, 5, subplot_spec = gs[1], wspace = 0.1)
# add axes
axA = fig.add_subplot(gs1[0, 0])
axB = fig.add_subplot(gs1[0, 1])
axC1 = fig.add_subplot(gs2[0, 0])
axC2 = fig.add_subplot(gs2[0, 1])
axC3 = fig.add_subplot(gs2[0, 2])
axC4 = fig.add_subplot(gs2[0, 3])
axC5 = fig.add_subplot(gs2[0, 4])

axes = {
    'A': [axA,  (-0.4, -0.15)], 
    'B': [axB,  (-0.5, -0.15)], 
    'C': [axC1,  (-0.4, -0.15)],
}

# Panel labels
for label, (ax, xy) in axes.items():
    # label physical distance to the left and up:
    trans = mtransforms.ScaledTranslation(xy[0], xy[1], fig.dpi_scale_trans)
    ax.text(0.0, 1.1, label, transform=ax.transAxes + trans,
            fontsize=12, va='bottom', fontweight='bold')

# ------------Panel A------------
sns.lineplot(
    data=scaled_pvals,
    x='pvalue',
    y='scaling_factor',
    hue='penalty_factor',
    palette=palette,
    ax=axA
    )

axA.annotate(f'$\itp$=0.05', xy=(0.3, 0.01), xytext=(0.25, 0.01))
axA.axvline(0.05, color='black', linestyle='--', linewidth=0.5)

axA.set_xlim(1, 0)
axA.set_xlabel('p-value')
axA.set_ylabel('Sacling factor')

axA.legend(title='Penalty', bbox_to_anchor=(1.0, 1.0), frameon=False)

# ------------Panel B------------
sns.scatterplot(
    data=dataset, 
    x='dataset', 
    y='value', 
    color='black', 
    ax=axB
    )
sns.boxplot(
    data=dataset.groupby('dataset').mean().reset_index(), 
    x='dataset', 
    y='value', 
    medianprops={'color':'black'},
    ax=axB
    )
for penalty in factors:
    sns.boxplot(data=smoothed_data[smoothed_data['penalty_factor'] == penalty], x='dataset', y='value', medianprops={'color':palette[penalty]}, ax=axB)

for x, (data_number, df) in enumerate(params.groupby('dataset')):
    pval = df['pvalue'].tolist()[0]
    axB.annotate(f'$\itp$={round(pval, 4)}', xy=(data_number, 100), xytext=(x, 85), ha='center')

ax=axB.axhline(500, color='black', linestyle='--', linewidth=0.5)

axB.set(xlabel='Example Datasets', ylim=(-5, 1005))
axB.set_ylabel('Value', labelpad=0.01)
axB.set_xticks(ticks=axB.get_xticks(), labels=['Consistent', 'Variable'])

custom_lines = [Line2D([0], [0], color=palette[factor], lw=2) for factor in factors]
axB.legend(title='Penalty', handles=custom_lines, labels=factors, bbox_to_anchor=(1.0, 1.0), frameon=False)

# ------------Panel C------------

mean_image = summary[['x', 'y', 'noised_mean']].pivot('x', 'y', f'noised_mean').values
axC1.imshow(mean_image, cmap=plt.cm.gray)
axC1.set(xticks=[], yticks=[])
axC1.axis('off')

for ax, penalty_factor in zip([axC2, axC3, axC4, axC5], [2, 20, 200, 2000]):
    smooth_image = summary[['x', 'y', f'{penalty_factor}_smooth_value']].pivot('x', 'y', f'{penalty_factor}_smooth_value').values
    ax.imshow(smooth_image, cmap=plt.cm.gray)
    ax.annotate(f'Penalty = {penalty_factor}', xy=(48, 5), ha='right', color='white')
    ax.set(xticks=[], yticks=[])
    ax.axis('off')

# Add annotations   
trans = axC1.get_xaxis_transform() 
axC1.annotate('Mean from 5\nnoised images', xy=(25, -.1), xycoords=trans, ha="center", va="top")

trans = axC3.get_xaxis_transform() 
axC3.annotate('Scaled from 5 noised images', xy=(55, -.2), xycoords=trans, ha="center", va="top")

# Get the bounding boxes of the axes including text decorations
axes = np.array([axC2, axC3, axC4, axC5])
r = fig.canvas.get_renderer()
get_bbox = lambda ax: ax.get_tightbbox(r).transformed(fig.transFigure.inverted())
bboxes = np.array(list(map(get_bbox, axes)), mtransforms.Bbox).reshape(axes.shape)

#Get the minimum and maximum extent, get the coordinate half-way between those
xmax = np.array(list(map(lambda b: b.x1, bboxes.flat))).reshape(axes.shape).max()
xmin = np.array(list(map(lambda b: b.x0, bboxes.flat))).reshape(axes.shape).min()

# Draw a horizontal line at those coordinates
line = plt.Line2D([0.245, 0.891],[0.1, 0.1], transform=fig.transFigure, color="black")
fig.add_artist(line)

# Figure admin
gs.tight_layout(fig)
plt.savefig(f'{output_folder}S1_Optimisation.svg', bbox_inches='tight')
plt.show()
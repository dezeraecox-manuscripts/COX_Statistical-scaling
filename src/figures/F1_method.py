import matplotlib.patches as patches
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

input_folder = 'data/simulated-data/images/'
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
    'original': 'black',
    'noised_mean': '#E3B504',
    '20_smooth_value':'#B0185E'
    }

# ===========Read in image examples===========
original_image = np.load(f'{input_folder}original_array.npy')
original_image = original_image[:50, :50]

noised_images = np.load(f'{input_folder}noised_array_stack.npy')
noised_images = noised_images[:, :50, :50]
noised_100_mean = np.mean(noised_images, axis=0)
noise_ind = noised_images[1, :50, :50].copy()


# ==============Read in datasets==============

summary = pd.read_csv(f'{input_folder}smoothed_summary.csv')
summary.drop([col for col in summary.columns.tolist()
             if 'Unnamed: ' in col], axis=1, inplace=True)

original = summary[['x', 'y', 'original']].pivot('x', 'y', 'original').values

noised_5_mean = summary[['x', 'y', 'noised_mean']].pivot('x', 'y', 'noised_mean').values

smooth = summary[['x', 'y', '20_smooth_value']].pivot(
    'x', 'y', '20_smooth_value').values


linescan = summary[summary['x'] == 49].copy()
linescan_noised = pd.melt(
    linescan.copy(),
    id_vars=['x', 'y'], 
    value_vars=[f'noised_{x+1}' for x in range(5)], 
    var_name='replicate', 
    value_name='value'
)
linescan_noised['replicate'] = linescan_noised['replicate'].str.split('_').str[-1].astype(int)
linescan_summary = pd.melt(
    linescan.copy(),
    id_vars=['x', 'y'], 
    value_vars=['original', 'noised_mean', '20_smooth_value'], 
    var_name='Data type', 
    value_name='value'
)


# ==============Generate figure==============

# Figure defaults
fig = plt.figure(figsize=(18*cm, 16*cm))

# make outer gridspec
gs = GridSpec(3, 1, figure=fig, height_ratios=[1.5, 1, 1], hspace=0.1)
# make nested gridspecs
gs0 = GridSpecFromSubplotSpec(1, 1, subplot_spec=gs[0], wspace=0.5)
gs1 = GridSpecFromSubplotSpec(1, 5, subplot_spec=gs[1], wspace=0.1)
gs2 = GridSpecFromSubplotSpec(1, 1, subplot_spec=gs[2], wspace=0)
# add axes
axA = fig.add_subplot(gs0[0, 0])
axB1 = fig.add_subplot(gs1[0, 0])
axB2 = fig.add_subplot(gs1[0, 1])
axB3 = fig.add_subplot(gs1[0, 2])
axB4 = fig.add_subplot(gs1[0, 3])
axB5 = fig.add_subplot(gs1[0, 4])
axC = fig.add_subplot(gs2[0, 0])

axes = {
    'A': [axA,  (-0.5, -0.15)],
    'B': [axB1,  (-0.5, -0.15)],
    'C': [axC,  (-0.5, -0.15)],
}

# Panel labels
for label, (ax, xy) in axes.items():
    # label physical distance to the left and up:
    trans = mtransforms.ScaledTranslation(xy[0], xy[1], fig.dpi_scale_trans)
    ax.text(0.0, 1.1, label, transform=ax.transAxes + trans,
            fontsize=12, va='bottom', fontweight='bold')

# ------------Panel A------------
axA.axis('off')

# ------------Panel B------------
axs = [axB1, axB2, axB3, axB4, axB5]
images = [original, noise_ind, noised_100_mean, noised_5_mean, smooth]
labels = [
    'Ground truth',
    'Individual\nsample image',
    'Sample mean of\n100 images',
    'Sample mean of\n5 images',
    'Scaled from\n5 images',
]

for ax, img, label in zip(axs, images, labels):
    ax.imshow(img, cmap='Greys_r')
    ax.axis('off')
    trans = ax.get_xaxis_transform() 
    ax.annotate(label, xy=(25, -.04), xycoords=trans, ha="center", va="top")


# ------------Panel C------------
sns.lineplot(data = linescan_noised, x='y', y='value', color='grey', label='noise_range', ax=axC)
sns.lineplot(data = linescan_summary[linescan_summary['Data type'] != 'original'], x='y', y='value', hue='Data type', palette=palette, hue_order=['noised_mean', '20_smooth_value', ], ax=axC)
sns.lineplot(data = linescan_summary[linescan_summary['Data type'] == 'original'], x='y', y='value', hue='Data type', palette=palette, hue_order=['original', ], ax=axC, linestyle = '--')

axC.set_xlim(np.min(linescan_noised['y'])-1, np.max(linescan_noised['y'])+1)
axC.set_ylim(-1.2, 1.7)
axC.set_ylabel('Value')
axC.set_xlabel('Position')

handles, labels = axC.get_legend_handles_labels()
axC.legend([handles[x] for x in [3, 0, 1, 2]], ['Ground truth', 'Noise range', 'Sample value', 'Scaled value'], loc='lower right', frameon=False, labelspacing = 0.1)

boxes = [
    (14, -1.1, 2, 2.7),
    (28, -1.1, 2, 2.7),
]

for x, y, width, height in boxes:
    # Create a Rectangle patch
    rect = patches.Rectangle((x, y), width, height, linewidth=1, edgecolor='black', facecolor='none')
    # Add the patch to the Axes
    axC.add_patch(rect)
    
# Figure admin
gs.tight_layout(fig)
plt.savefig(f'{output_folder}F1_Simulated.svg')
plt.show()
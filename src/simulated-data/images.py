import os
import pandas as pd
import numpy as np
from skimage.draw import disk
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

from src.utils import pval_smoothing

from loguru import logger

logger.info('Import OK')

num_noise_images = 5
data_folder = 'data/simulated-data/images/'
output_folder = 'data/simulated-data/images/'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

np.random.seed(1234)

## ==================SIMULATED IMAGE DATASET==================
# Define simple image of a circle
img = np.zeros((100, 100), dtype=np.uint8)
img[disk((50, 50), 25)] = 1
plt.imshow(img, cmap=plt.cm.gray)

# Add noise using random array
noised_images = np.array([
    img+np.random.normal(loc=0, scale=0.5,size=img.shape)
    for _ in range(100)
])

# Save to numpy array
np.save(f'{data_folder}original_array.npy', img)
np.save(f'{data_folder}noised_array_stack.npy', np.stack(noised_images))

# ----------------Read in saved circle arrays----------------
original = np.load(f'{output_folder}original_array.npy')
noised_images = np.load(f'{output_folder}noised_array_stack.npy')

# Extract point by point info for a quarter of the image (enough to show contrast)
point_data = {(x, y): [original[x, y]] + [noised_images[i, x, y] for i in range(num_noise_images)] for x in range(50) for y in range(50) }
point_data = pd.DataFrame(point_data).T.reset_index().rename(columns={'level_0':'x', 'level_1':'y', 0: 'original'})
# NOTE: original image will be labelled replicate 0, then noised images 1-num_noise_images

# Collect noised data for later comparison
noised_data = point_data.copy()
noised_data['noised_mean'] = noised_data[[col for col in point_data.columns.tolist() if type(col) == int]].mean(axis=1)
noised_data.columns = [f'noised_{col}' if type(col) == int else col for col in noised_data.columns]

# melt to enable smoothing
point_data = pd.melt(
    point_data,
    id_vars=['x', 'y', 'original'], 
    value_vars=[col for col in point_data.columns.tolist() if type(col) == int], 
    var_name='replicate', 
    value_name='value'
    )

# Apply pval_smoothing independently based on original pixel mean
summary = noised_data.copy()
for penalty_factor in [2, 20, 200, 2000]:
    smoothed_data = [pval_smoothing(
            df.reset_index(),
            sample_cols=['value'], 
            group_cols=['x', 'y', 'original'], 
            popmean=value, 
            penalty_factor=penalty_factor, 
        ) for value, df in point_data.groupby('original')]
    smoothed_data = pd.concat(smoothed_data).reset_index().rename(columns={'value': f'{penalty_factor}_smooth_value'})
    # Add mean noised values, calculate residuals
    summary = pd.merge(summary, smoothed_data, on=['x', 'y', 'original'])

for col in ['noised_1', 'noised_mean']+[col for col in summary.columns.tolist() if 'smooth' in col]:
    summary[f'{col}_res'] = summary[col] - summary['original']

# Save to csv
summary.to_csv(f'{output_folder}smoothed_summary.csv')


## ==================REAL IMAGE DATASET==================

from skimage.data import colorwheel

color_img = colorwheel()
plt.imshow(color_img)

channel_1 = color_img[:, :, 0]
plt.imshow(channel_1)

channel_2 = color_img[:, :, 1]
channel_3 = color_img[:, :, 2]

# Add noise using random array
noised_images_1 = np.array([
    channel_1+10*np.random.normal(loc=0, scale=0.5,size=channel_1.shape)
    for _ in range(100)
])

noised_images_2 = np.array([
    channel_2+10*np.random.normal(loc=0, scale=0.5,size=channel_2.shape)
    for _ in range(100)
])

noised_images_3 = np.array([
    channel_3+10*np.random.normal(loc=0, scale=0.5,size=channel_3.shape)
    for _ in range(100)
])


# Save to numpy array
np.save(f'{data_folder}original_colorwheel.npy', channel_1)
np.save(f'{data_folder}noised_colorwheel_stack_1.npy', np.stack(noised_images_1))
np.save(f'{data_folder}noised_colorwheel_stack_2.npy', np.stack(noised_images_2))
np.save(f'{data_folder}noised_colorwheel_stack_3.npy', np.stack(noised_images_3))

# ----------------Read in saved circle arrays----------------
original = np.load(f'{output_folder}original_colorwheel.npy')
noised_images_1 = np.load(f'{output_folder}noised_colorwheel_stack_1.npy')
noised_images_2 = np.load(f'{output_folder}noised_colorwheel_stack_2.npy')
noised_images_3 = np.load(f'{output_folder}noised_colorwheel_stack_3.npy')

for x, noised_images in enumerate([noised_images_1, noised_images_2, noised_images_3]):

    point_data = {(x, y): [original[x, y]] + [noised_images[i, x, y] for i in range(num_noise_images)] for x in range(370) for y in range(371) }
    point_data = pd.DataFrame(point_data).T.reset_index().rename(columns={'level_0':'x', 'level_1':'y', 0: 'original'})
    # NOTE: original image will be labelled replicate 0, then noised images 1-num_noise_images

    # Collect noised data for later comparison
    noised_data = point_data.copy()
    noised_data['noised_mean'] = noised_data[[col for col in point_data.columns.tolist() if type(col) == int]].mean(axis=1)
    noised_data.columns = [f'noised_{col}' if type(col) == int else col for col in noised_data.columns]

    # melt to enable smoothing
    point_data = pd.melt(
        point_data,
        id_vars=['x', 'y', 'original'], 
        value_vars=[col for col in point_data.columns.tolist() if type(col) == int], 
        var_name='replicate', 
        value_name='value'
        )

    # Apply pval_smoothing independently based on original pixel mean
    summary = noised_data.copy()
    for penalty_factor in [2, 20, 200, 2000]:
        smoothed_data = [pval_smoothing(
                df.reset_index(),
                sample_cols=['value'], 
                group_cols=['x', 'y', 'original'], 
                popmean=value, 
                penalty_factor=penalty_factor, 
            ) for value, df in point_data.groupby('original')]
        smoothed_data = pd.concat(smoothed_data).reset_index().rename(columns={'value': f'{penalty_factor}_smooth_value'})
        # Add mean noised values, calculate residuals
        summary = pd.merge(summary, smoothed_data, on=['x', 'y', 'original'])

    for col in ['noised_1', 'noised_mean']+[col for col in summary.columns.tolist() if 'smooth' in col]:
        summary[f'{col}_res'] = summary[col] - summary['original']

    # Save to csv
    summary.to_csv(f'{output_folder}smoothed_colorwheel_{x}.csv')

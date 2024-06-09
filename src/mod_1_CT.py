"""!@file mod_1_CT.py

@brief This file contains code for the segmentation of the CT scan image

@detail Uses...

@author T.Breitburd on 09/06/2024"""

import numpy as np
import matplotlib.pyplot as plt
import os
import skimage
from skimage.filters import threshold_otsu
from skimage.measure import label
from skimage.morphology import disk, closing
from skimage.segmentation import flood_fill

# Load the image and reduce to 1 channel
ct = skimage.io.imread("./data/CT.png")
ct = ct[:, :, 0]

# ----------------------------------------
# First identify where the lungs are
# ----------------------------------------

# Threshold the image
threshold = threshold_otsu(ct)
ct_thresh = ct < threshold

# Apply closing to remove the small objects
ct_masked = skimage.morphology.closing(ct_thresh, disk(3))
# Identify regions in the thresholded image
ct_regions = label(ct_masked)
n_regions = len(np.unique(ct_regions))

# The lung regions are going be the 2 smallest ones
region_sizes = np.bincount(ct_regions.flatten())

# Get indices of 2 smallest values
indices = np.argpartition(region_sizes, 2)[:2]

# Get seeds for the region growing
idx1 = len(np.argwhere(ct_regions == indices[0])) // 2
idx2 = len(np.argwhere(ct_regions == indices[1])) // 2

seed1 = tuple(np.argwhere(ct_regions == indices[0])[idx1])
seed2 = tuple(np.argwhere(ct_regions == indices[1])[idx2])

# ----------------------------------------
# Region Growing
# ----------------------------------------

# First one
mask_flood = flood_fill(ct, seed1, new_value=255, tolerance=25)
# Then the other
mask_flood = flood_fill(mask_flood, seed2, new_value=255, tolerance=25)

# Threshold the image
threshold = threshold_otsu(mask_flood)

binary = mask_flood > threshold

# Apply closing to get rid of inter-lung tissue
masked = closing(binary, disk(3))

# ---------------------------------------
# Plot the results
# ------------------ --------------------

fig, ax = plt.subplots(2, 2, figsize=(10, 10))

ax[0, 0].imshow(ct, cmap="gray")
ax[0, 0].set_title("Original CT")

ax[0, 1].imshow(ct_regions, cmap="tab20")
ax[0, 1].set_title("Regions")

ax[1, 0].imshow(mask_flood, cmap="gray")
ax[1, 0].set_title("Region Growing")

ax[1, 1].imshow(masked, cmap="gray")
ax[1, 1].set_title("Segmented CT")

plt.suptitle("CT Segmentation")
plt.tight_layout()
# Save the plot
cur_dir = os.getcwd()
plots_dir = os.path.join(cur_dir, "Plots")
os.makedirs(plots_dir, exist_ok=True)

plot_dir = os.path.join(plots_dir, "CT_Segmentation.png")
plt.savefig(plot_dir)

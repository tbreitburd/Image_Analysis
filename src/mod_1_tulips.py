"""!@file mod_1_tulips.py

@brief This file contains code for the segmentation of the tulips image

@details The tulips are segmented using the Hue channel of the HSV color space.
The image is thresholded using Otsu's method, and opening is applied
to remove the small white spots. The Chan-Vese segmentation algorithm is then applied
to segment the tulips. The final mask is then opened to remove any remaining noise.

@author T.Breitburd on 09/06/2024"""


import skimage
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
from skimage.morphology import opening, disk
from skimage.segmentation import chan_vese
import os
from plot_funcs import plot_tulips_hsv, plot_hue_hist

# Set the random seed
np.random.seed(75016)

# Load the image
tulips = skimage.io.imread("./data/noisy_flower.jpg")
tulips = tulips[:, :, :3]

# ----------------------------------------
# Preprocess the image
# ----------------------------------------

# Switch to Hue-Saturation-Value (HSV) color space
tulips_hsv = skimage.color.rgb2hsv(tulips)

# Plot grayscale image of each channel of the HSV image
plot_tulips_hsv(tulips_hsv)

# Plot the histogram of the Hue channel
plot_hue_hist(tulips_hsv[:, :, 0])

# ----------------------------------------
# Segmentation
# ----------------------------------------

# Threshold the image on hue channel
thresh = threshold_otsu(tulips_hsv[:, :, 0])
tulips_mask = tulips_hsv[:, :, 0] > thresh

# Apply opening to remove the small white spots, and keep the larger ones (purple tulips)
op_mask = opening(tulips_mask)

print("Image thresholded.")

# Gaussian blur to smooth the mask
blurred_mask = skimage.filters.gaussian(op_mask, sigma=2)

# Apply the Chan-Vese segmentation, get
cv_mask = chan_vese(blurred_mask, mu=0.1, lambda2=1.5)

# Apply opening
cv_mask_op = opening(cv_mask, disk(4))

print("Segmentation done.")

# ----------------------------------------
# Plot the results
# ----------------------------------------

fig, ax = plt.subplots(2, 2, figsize=(10, 10))

ax[0, 0].imshow(tulips)
ax[0, 0].set_title("Original Image")
ax[0, 0].grid(False)

ax[0, 1].imshow(blurred_mask, cmap="gray")
ax[0, 1].set_title("Blurred Binary Image")
ax[0, 1].grid(False)

ax[1, 0].imshow(cv_mask, cmap="gray")
ax[1, 0].set_title("Chan-Vese Segmentation")
ax[1, 0].grid(False)

ax[1, 1].imshow(cv_mask_op, cmap="gray")
ax[1, 1].set_title("Opened Final Image")
ax[1, 1].grid(False)

plt.tight_layout()
# Save the plot
cur_dir = os.getcwd()
plots_dir = os.path.join(cur_dir, "Plots")
os.makedirs(plots_dir, exist_ok=True)

plot_dir = os.path.join(plots_dir, "tulip_segmentation.png")
plt.savefig(plot_dir)

plt.close()

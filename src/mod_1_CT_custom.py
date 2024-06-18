"""!@file mod_1_CT_custom.py

@brief CT Segmentation using custom region growing algorithm

@details This module contains the code for the segmentation of a CT image.
The lungs are segmented using a custom region growing algorithm, with the seeds
being the 2 smallest regions in the thresholded image.

@author T. Breitburd on 14/06/2024
"""


import numpy as np
import skimage
import matplotlib.pyplot as plt
from skimage.measure import label
from skimage.morphology import disk, closing
import os

# Load the image and reduce to 1 channel
ct = skimage.io.imread("./data/CT.png")
ct = ct[:, :, 0]

# ----------------------------------------
# Define some functions
# ----------------------------------------


def otsu_threshold(image):
    """!@brief Function to apply Otsu's thresholding to an image,
    from https://www.baeldung.com/cs/otsu-segmentation, section 2.4

    @param image the image to threshold, numpy array

    @return the thresholded image, numpy array"""

    # Discretize the problem of finding the optimal threshold by binning the
    # data

    # Get the histogram of the image
    hist, bins = np.histogram(image, bins=256, density=True)

    # Calculate bin centres
    bin_centres = (bins[:-1] + bins[1:]) / 2

    # Because histogram is normalized, we can calculate the probabilities
    # of the classes, for all possible thresholds
    P0 = np.cumsum(hist)
    P1 = np.cumsum(hist[::-1])[::-1]

    # Calculate the class means
    M0 = np.cumsum(hist * bin_centres) / P0
    M1 = (np.cumsum((hist * bin_centres)[::-1]) / P1[::-1])[::-1]

    # Calculate the inter-class variance
    var = P0 * P1 * ((M0 - M1) ** 2)

    # Find the threshold that maximizes the inter-class variance
    threshold = bins[np.argmax(var)]

    # Apply the threshold
    image_ = image < threshold

    return image_


def region_growing(image, seed, threshold=0.2):
    """!@brief Perform region growing algorithm on an image,
    from https://sbme-tutorials.github.io/2019/cv/notes/6_week6.html

    @param image the image to segment, numpy array
    @param seed the seed pixel, tuple
    @param threshold the threshold to use, float

    @return the segmented image, numpy array"""

    # Get the image dimensions
    rows, cols = image.shape

    # Initialize the segmented image and the list of unsegmented pixels
    is_segmented = np.zeros_like(image, dtype=bool)
    unsegmented_pxl = [seed]
    seed_value = image[seed]
    image_ = np.copy(image)

    # While there are unsegmented pixels
    while unsegmented_pxl:
        # Get the next unssegmented pixel
        x, y = unsegmented_pxl.pop(0)
        if not is_segmented[x, y]:
            is_segmented[x, y] = True

            # Get the neighbors
            neighbors = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
            for x_temp, y_temp in neighbors:
                # Check if the pixel is in the image (for edges) and not segmented
                if (
                    0 <= x_temp < rows
                    and 0 <= y_temp < cols
                    and not is_segmented[x_temp, y_temp]
                ):
                    # Check for similarity
                    if (
                        np.abs(float(image[x_temp, y_temp]) - float(seed_value)) / 255
                        < threshold
                    ):
                        unsegmented_pxl.append((x_temp, y_temp))

    # Set the segmented region to white
    for i in range(rows):
        for j in range(cols):
            if is_segmented[i, j]:
                image_[i, j] = 255
    return image_


# ----------------------------------------
# First identify where the lungs are
# ----------------------------------------

# Threshold the image
ct_thresh = otsu_threshold(ct)

# Apply closing to remove the small objects
ct_masked = closing(ct_thresh, disk(3))

print("Image thesholded.")

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

print("Seeds found.")

# ----------------------------------------
# Region Growing
# ----------------------------------------

# First one
mask_flood = region_growing(ct, seed1, threshold=0.1)
# Then the other
mask_flood = region_growing(mask_flood, seed2, threshold=0.1)

# Threshold the image again
threshold = 254

binary = mask_flood > threshold

# Apply closing to get rid of inter-lung tissue
masked = closing(binary, disk(3))

print("Region growing done.")

# ---------------------------------------
# Plot the results
# ---------------------------------------

plt.style.use("seaborn-v0_8-darkgrid")

fig, ax = plt.subplots(2, 2, figsize=(10, 10))

ax[0, 0].imshow(ct, cmap="gray")
ax[0, 0].set_title("(a) Original CT")
ax[0, 0].grid(False)

ax[0, 1].imshow(ct_regions, cmap="tab20")
ax[0, 1].set_title("(b) Regions")
ax[0, 1].grid(False)

ax[1, 0].imshow(mask_flood, cmap="gray")
ax[1, 0].set_title("(c) Region Growing")
ax[1, 0].grid(False)

ax[1, 1].imshow(masked, cmap="gray")
ax[1, 1].set_title("(d) Segmented CT")
ax[1, 1].grid(False)

plt.suptitle("CT Segmentation")
plt.tight_layout()
# Save the plot
cur_dir = os.getcwd()
plots_dir = os.path.join(cur_dir, "Plots")
os.makedirs(plots_dir, exist_ok=True)

plot_dir = os.path.join(plots_dir, "CT_Segmentation_custom.png")
plt.savefig(plot_dir)

plt.close()

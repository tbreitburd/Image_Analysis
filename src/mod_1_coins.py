"""!@file mod_1_coins.py

@brief This module contains code for the segmentation of the coins

@details This module contains the code for the segmentation of the coins.
This is done by first removing the "corruption" lines

"""
import numpy as np
import skimage
from skimage.restoration import inpaint, rolling_ball
from skimage.exposure import rescale_intensity
from skimage.color import label2rgb
from skimage.measure import label
from skimage import measure
from skimage.morphology import closing, disk
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


# ----------------------------------------
# Preprocessing
# ----------------------------------------

# Set seed
np.random.seed(75016)

# First load the image, and keep only 1 channel
coins = skimage.io.imread("./data/coins.png")
coins = coins[:, :, 0]

# Now we will remove the "corruption" lines
mask = coins == 0
coins_inpaint = inpaint.inpaint_biharmonic(coins, mask)

# To facilitate the segmentation, we increase the contrast
coins_rescaled = rescale_intensity(coins_inpaint, in_range=(0.2, 0.8), out_range=(0, 1))

# Shift image back to 0-255
coins_rescaled = np.array(coins_rescaled * 255, dtype=np.uint8)

# ----------------------------------------
# Remove background with rolling ball algorithm
# ----------------------------------------

# Top left corner is too bright so, first attempt to remove the background
# using a rolling ball algorithm
background = rolling_ball(coins_rescaled, radius=30)

# Subtract the background
coins_no_background = coins_rescaled - background


# ----------------------------------------
# Use K-means to segment the coins
# ----------------------------------------

# Reshape the image to 1D
coins_reshaped = coins_no_background.reshape(-1, 1)

# Use K-means to segment the coins
kmeans = KMeans(n_clusters=2, random_state=0, n_init=10).fit(coins_reshaped)
coins_segmented = kmeans.labels_.reshape(coins_no_background.shape)

# Close the image
coins_segmented = closing(coins_segmented, disk(2))

# ----------------------------------------
# Label the coins
# ----------------------------------------

# Label the coins
coins_labelled = label(coins_segmented)

# Convert the labels to RGB
coins_labelled_rgb = label2rgb(coins_labelled, image=coins_segmented, bg_label=0)


fig = plt.figure(figsize=(10, 6))
plt.imshow(coins_labelled_rgb)

for region in measure.regionprops(coins_labelled):
    y, x = region.centroid
    plt.text(
        x, y, str(region.label), color="red", fontsize=12, ha="center", va="center"
    )

plt.show()


# ----------------------------------------
# Plot the results
# ----------------------------------------

"""!@file mod_1_coins.py

@brief This module contains code for the segmentation of the coins

@details This module contains the code for the segmentation of the coins.
This is done by first removing the "corruption" lines

"""
import numpy as np
import os
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
coins_labelled_rgb = label2rgb(coins_labelled, bg_label=0)

# ----------------------------------------
# Plot the results
# ----------------------------------------

# Coins that we want are regioms 7, 16, 23 and 27


def plot_segmented_coin(coin_number):
    # Get the coin region
    region_props = measure.regionprops(coins_labelled)
    coin_region = region_props[coin_number - 1]

    # Get the bounding box
    min_y, min_x, max_y, max_x = coin_region.bbox

    min_y, min_x = max(min_y - 10, 0), max(min_x - 10, 0)
    max_y, max_x = min(max_y + 10, coins.shape[0]), min(max_x + 10, coins.shape[1])

    # Plot the steps of the segmentation
    plt.figure(figsize=(8, 8))
    plt.subplot(221)
    plt.imshow(coins[min_y:max_y, min_x:max_x], cmap="gray")
    plt.title("Original image")
    plt.axis("off")

    plt.subplot(222)
    plt.imshow(coins_rescaled[min_y:max_y, min_x:max_x], cmap="gray")
    plt.title("Inpainted and Rescaled Image")
    plt.axis("off")

    plt.subplot(223)
    plt.imshow(coins_no_background[min_y:max_y, min_x:max_x], cmap="gray")
    plt.title("Background removed")
    plt.axis("off")

    plt.subplot(224)
    plt.imshow(coins_labelled_rgb[min_y:max_y, min_x:max_x])
    plt.title("K-Means Segmentation and Labeling")
    plt.axis("off")

    plt.suptitle("Coins Segmentation")
    plt.tight_layout()
    # Save the plot
    cur_dir = os.getcwd()
    plots_dir = os.path.join(cur_dir, "Plots")
    os.makedirs(plots_dir, exist_ok=True)

    plot_dir = os.path.join(plots_dir, "Coin_Segmentation_" + str(coin_number) + ".png")
    plt.savefig(plot_dir)


# Plot the segmented coins
plot_segmented_coin(7)
plot_segmented_coin(16)
plot_segmented_coin(23)
plot_segmented_coin(27)

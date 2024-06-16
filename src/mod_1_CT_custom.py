import numpy as np
import skimage
import matplotlib.pyplot as plt


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
    P1 = 1 - P0

    # Calculate the class means
    M0 = np.cumsum(hist * bin_centres) / P0
    M1 = np.cumsum((hist * bin_centres)[::-1]) / P1[::-1]

    # Calculate the inter-class variance
    var = P0 * P1 * ((M0 - M1) ** 2)

    # Find the threshold that maximizes the inter-class variance
    threshold = bins[np.argmax(var)]

    return image > threshold


ct_thresh = otsu_threshold(ct)

plt.imshow(ct_thresh, cmap="gray")
plt.grid(False)
plt.show()

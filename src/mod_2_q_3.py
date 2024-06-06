"""!@file mod_2_q_2.py
@brief Python script for question 2 of module 2."""

import pywt
import numpy as np
import matplotlib.pyplot as plt
from plot_funcs import plot_riverside, plot_daubechies, plot_reconstruct_image

# ------------------------------------------
# a) Load the image and get the Daubechies
# wavelet transform
# ------------------------------------------

riverside = plt.imread(
    "./data/river_side.jpeg",
    format="jpeg",
)[:, :, 0]

# Plot the riverside image
plot_riverside(riverside)

# Get the Daubechies wavelet transform
daub = pywt.dwt2(riverside, wavelet="db3")

approx, (hori, vert, diag) = daub

# Plot the Daubechies wavelet coefficients
plot_daubechies(
    approx, hori, vert, diag, "Daubechies Wavelet Coefficients", "q3_daubechies.png"
)

# Reconstruct the image
reconstructed = pywt.idwt2(daub, wavelet="db3")

# Plot the reconstructed image
plot_reconstruct_image(riverside, reconstructed, "q3_riverside_reconstructed.png")

# Plot the difference between the original and reconstructed image
plot_riverside(
    riverside - reconstructed,
    "Difference between original and reconstructed image",
    "q3_riverside_diff.png",
)


# ------------------------------------------
# b) Threshold the wavelet coefficients
# ------------------------------------------


# Threshold the wavelet coefficients
def threshold_riverside(threshold, approx, wavelet_coeffs):
    """!@brief Threshold the wavelet coefficients of the riverside image.

    @param threshold (float): The threshold value.
    @param approx (numpy.ndarray): The approximation coefficients.
    @param wavelet_coeffs (tuple): The wavelet coefficients.

    @return (tuple): The thresholded approximation and wavelet coefficients.
    """

    LH_thresh = pywt.threshold(
        wavelet_coeffs[0],
        threshold * np.max(wavelet_coeffs[0]),
        mode="hard",
        substitute=0,
    )
    HL_thresh = pywt.threshold(
        wavelet_coeffs[1],
        threshold * np.max(wavelet_coeffs[1]),
        mode="hard",
        substitute=0,
    )
    HH_thresh = pywt.threshold(
        wavelet_coeffs[2],
        threshold * np.max(wavelet_coeffs[2]),
        mode="hard",
        substitute=0,
    )

    return approx, (LH_thresh, HL_thresh, HH_thresh)


# Retain only top 15% of the wavelet coefficients
threshold = 0.85
approx, (hori_15, vert_15, diag_15) = threshold_riverside(
    threshold, approx, (hori, vert, diag)
)

# Plot the thresholded wavelet coefficients
plot_daubechies(
    approx,
    hori_15,
    vert_15,
    diag_15,
    "Top 15% Thresholded Daubechies \n Wavelet Coefficients",
    "q3_daubechies_thresh_15.png",
)


# ------------------------------------------
# c) Repeat the thresholding for different
# percentages and reconstruct the image
# ------------------------------------------

# Retain only top 20% of the wavelet coefficients
threshold = 0.80
approx, (hori_20, vert_20, diag_20) = threshold_riverside(
    threshold, approx, (hori, vert, diag)
)

# Plot the thresholded wavelet coefficients
plot_daubechies(
    approx,
    hori_20,
    vert_20,
    diag_20,
    "Top 20% Thresholded Daubechies \n Wavelet Coefficients",
    "q3_daubechies_thresh_20.png",
)

# Retain only top 10% of the wavelet coefficients
threshold = 0.90
approx, (hori_10, vert_10, diag_10) = threshold_riverside(
    threshold, approx, (hori, vert, diag)
)

# Plot the thresholded wavelet coefficients
plot_daubechies(
    approx,
    hori_10,
    vert_10,
    diag_10,
    "Top 10% Thresholded Daubechies \n Wavelet Coefficients",
    "q3_daubechies_thresh_10.png",
)

# Retain only top 5% of the wavelet coefficients
threshold = 0.95
approx, (hori_5, vert_5, diag_5) = threshold_riverside(
    threshold, approx, (hori, vert, diag)
)

# Plot the thresholded wavelet coefficients
plot_daubechies(
    approx,
    hori_5,
    vert_5,
    diag_5,
    "Top 5% Thresholded Daubechies \n Wavelet Coefficients",
    "q3_daubechies_thresh_5.png",
)

# Retain only top 2.5% of the wavelet coefficients
threshold = 0.975
approx, (hori_2_5, vert_2_5, diag_2_5) = threshold_riverside(
    threshold, approx, (hori, vert, diag)
)

# Plot the thresholded wavelet coefficients
plot_daubechies(
    approx,
    hori_2_5,
    vert_2_5,
    diag_2_5,
    "Top 2.5% Thresholded Daubechies \n Wavelet Coefficients",
    "q3_daubechies_thresh_2_5.png",
)


# Reconstruction of the images

# Reconstruct the image with 15% thresholded coefficients
reconstructed_15 = pywt.idwt2((approx, (hori_15, vert_15, diag_15)), wavelet="db3")

# Plot the reconstructed image
plot_reconstruct_image(riverside, reconstructed_15, "q3_riverside_reconstructed_15.png")

# Reconstruct the image with 20% thresholded coefficients
reconstructed_20 = pywt.idwt2((approx, (hori_20, vert_20, diag_20)), wavelet="db3")

# Plot the reconstructed image
plot_reconstruct_image(riverside, reconstructed_20, "q3_riverside_reconstructed_20.png")

# Reconstruct the image with 10% thresholded coefficients
reconstructed_10 = pywt.idwt2((approx, (hori_10, vert_10, diag_10)), wavelet="db3")

# Plot the reconstructed image
plot_reconstruct_image(riverside, reconstructed_10, "q3_riverside_reconstructed_10.png")

# Reconstruct the image with 5% thresholded coefficients
reconstructed_5 = pywt.idwt2((approx, (hori_5, vert_5, diag_5)), wavelet="db3")

# Plot the reconstructed image
plot_reconstruct_image(riverside, reconstructed_5, "q3_riverside_reconstructed_5.png")

# Reconstruct the image with 2.5% thresholded coefficients
reconstructed_2_5 = pywt.idwt2((approx, (hori_2_5, vert_2_5, diag_2_5)), wavelet="db3")

# Plot the reconstructed image
plot_reconstruct_image(
    riverside, reconstructed_2_5, "q3_riverside_reconstructed_2_5.png"
)

# ------------------------------------------
# Plot the difference between the original
# and reconstructed images
# ------------------------------------------

# Plot the difference between the original and reconstructed image
plot_riverside(
    riverside - reconstructed_15,
    "Difference between original and \n reconstructed image (15% threshold)",
    "q3_riverside_diff_15.png",
)
plot_riverside(
    riverside - reconstructed_20,
    "Difference between original and \n reconstructed image (20% threshold)",
    "q3_riverside_diff_20.png",
)
plot_riverside(
    riverside - reconstructed_10,
    "Difference between original and \n reconstructed image (10% threshold)",
    "q3_riverside_diff_10.png",
)
plot_riverside(
    riverside - reconstructed_5,
    "Difference between original and \n reconstructed image (5% threshold)",
    "q3_riverside_diff_5.png",
)
plot_riverside(
    riverside - reconstructed_2_5,
    "Difference between original and \n reconstructed image (2.5% threshold)",
    "q3_riverside_diff_2_5.png",
)

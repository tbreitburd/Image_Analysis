"""!@file mod_2_q_2.py
@brief Python script for question 2 of module 2."""

import pywt
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
plot_daubechies(approx, hori, vert, diag)

# ------------------------------------------
# b) Reconstruct the image
# ------------------------------------------

# Reconstruct the image
reconstructed = pywt.idwt2(daub, wavelet="db3")

# Plot the reconstructed image
plot_reconstruct_image(riverside, reconstructed)

# Plot the difference between the original and reconstructed image
plot_riverside(
    riverside - reconstructed,
    "Difference between original and reconstructed image",
    "q3_riverside_diff.png",
)

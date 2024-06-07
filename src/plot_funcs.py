"""!@file plot_funcs.py
@brief Python script containing all plotting functions
for the Image Analysis Coursework

@details List of functions

@author T. Breitburd on 04/06/2024"""


import os
import matplotlib.pyplot as plt

# Define the plotting style
plt.style.use("seaborn-darkgrid")


def plot_fitted_lines(x, y, params, title, path):
    """!@brief Function to plot the fitted lines for the given parameters

    @param x x values, numpy array
    @param params a, b the slope and intercept of the fitted line, tuple

    @return None"""

    plt.figure(figsize=(5, 4))
    plt.scatter(x, y, label="Data")
    plt.plot(x, params[0] * x + params[1], label="Fitted line")
    plt.title(title)
    plt.legend()

    # Save the plot
    cur_dir = os.getcwd()
    plots_dir = os.path.join(cur_dir, "Plots")
    os.makedirs(plots_dir, exist_ok=True)

    plot_dir = os.path.join(plots_dir, path)
    plt.savefig(plot_dir)


def plot_signal_vector(vector):
    plt.figure(figsize=(5, 4))
    plt.stem(vector)
    plt.title("Measured Signal")

    # Save the plot
    cur_dir = os.getcwd()
    plots_dir = os.path.join(cur_dir, "Plots")
    os.makedirs(plots_dir, exist_ok=True)

    plot_dir = os.path.join(plots_dir, "q2.2_signal_vector.png")
    plt.savefig(plot_dir)


def plot_riverside(image, title="Riverside Image", path="q2.3_riverside.png"):
    """!@brief Function to plot the riverside image

    @param image the image to plot, numpy array

    @return None"""

    plt.figure(figsize=(5, 4))
    plt.imshow(image, cmap="gray")
    plt.grid(False)
    plt.title(title)

    # Save the plot
    cur_dir = os.getcwd()
    plots_dir = os.path.join(cur_dir, "Plots")
    os.makedirs(plots_dir, exist_ok=True)

    plot_dir = os.path.join(plots_dir, path)
    plt.savefig(plot_dir)


def plot_daubechies(approx, hori, vert, diag, title, path):
    """!@brief Function to plot the Daubechies wavelet coefficients

    @param approx the approximation coefficients, numpy array
    @param hori the horizontal detail coefficients, numpy array
    @param vert the vertical detail coefficients, numpy array
    @param diag the diagonal detail coefficients, numpy array

    @return None"""

    plt.figure(figsize=(10, 8))
    plt.subplot(221)
    plt.imshow(approx, cmap="gray")
    plt.grid(False)
    plt.title("Approximation")

    plt.subplot(222)
    plt.imshow(hori, cmap="gray")
    plt.grid(False)
    plt.title("Horizontal Detail Coefficients")

    plt.subplot(223)
    plt.imshow(vert, cmap="gray")
    plt.grid(False)
    plt.title("Vertical Detail Coefficients")

    plt.subplot(224)
    plt.imshow(diag, cmap="gray")
    plt.grid(False)
    plt.title("Diagonal Detail Coefficients")

    plt.suptitle(title)

    # Save the plot
    cur_dir = os.getcwd()
    plots_dir = os.path.join(cur_dir, "Plots")
    os.makedirs(plots_dir, exist_ok=True)

    plot_dir = os.path.join(plots_dir, path)
    plt.savefig(plot_dir)


def plot_reconstruct_image(im, reconstructed_im, path):
    """!@brief Function to plot the reconstructed image

    @param im the original image, numpy array
    @param reconstructed_im the reconstructed image, numpy array

    @return None"""

    plt.figure(figsize=(10, 8))
    plt.subplot(121)
    plt.imshow(im, cmap="gray")
    plt.grid(False)
    plt.title("Original Image")

    plt.subplot(122)
    plt.imshow(reconstructed_im, cmap="gray")
    plt.grid(False)
    plt.title("Reconstructed Image")

    # Save the plot
    cur_dir = os.getcwd()
    plots_dir = os.path.join(cur_dir, "Plots")
    os.makedirs(plots_dir, exist_ok=True)

    plot_dir = os.path.join(plots_dir, path)
    plt.savefig(plot_dir)

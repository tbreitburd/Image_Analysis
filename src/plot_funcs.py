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

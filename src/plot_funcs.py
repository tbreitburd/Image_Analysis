"""!@file plot_funcs.py
@brief Python script containing all plotting functions 
for the Image Analysis Coursework

@details List of functions

@author T. Breitburd on 04/06/2024"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# Define the plotting style 
plt.style.use('seaborn-darkgrid')

def plot_fitted_lines(x, params, ):
    """!@brief Function to plot the fitted lines for the given parameters

    @param x x values, numpy array
    @param params a, b the slope and intercept of the fitted line, tuple

    @return None"""

    plt.figure()
    plt.plot(x, params[0]*x + params[1], label='Line 1')
    plt.legend()
    plt.show()
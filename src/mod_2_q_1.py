"""!@file mod_2_q_1.py
@brief Python script for question 1 of module 2."""

import numpy as np
from scipy.optimize import minimize
from plot_funcs import plot_fitted_lines

# https://docs.scipy.org/doc/scipy/tutorial/optimize.html#
# constrained-minimization-of-multivariate-scalar-functions-minimize


# Load the data
y_line = np.loadtxt("./data/y_line.txt")
y_outlier = np.loadtxt("./data/y_outlier_line.txt")

# Define the x values, assuming they are given by indices
x = np.arange(len(y_line))


# Define the L1 functions to minimize
def l1_function_noise(params):
    a, b = params
    return np.sum(np.abs(y_line - (a * x + b)))


def l1_function_outlier(params):
    a, b = params
    return np.sum(np.abs(y_outlier - (a * x + b)))


# Initial guess for parameters
initial_guess = [1, 0]

# Minimize the L1 function, using the SLSQP method,
# as it used for constrained problems
result_noise = minimize(l1_function_noise, initial_guess, method="SLSQP")
result_outlier = minimize(l1_function_outlier, initial_guess, method="SLSQP")

# Get the results
print("---------------------------")
print("Results for L1 minimization")
print("---------------------------")

# For the noisy data
print("For the noisy data:")
print("a =", result_noise.x[0])
print("b =", result_noise.x[1])

# For the data with outliers
print("For the data with outliers:")
print("a =", result_outlier.x[0])
print("b =", result_outlier.x[1])

# Plot the results
plot_fitted_lines(
    x, y_line, result_noise.x, "L1 Fitted line for noisy data", "l1_noisy_data.png"
)
plot_fitted_lines(
    x,
    y_outlier,
    result_outlier.x,
    "L1 Fitted line for data with outliers",
    "l1_outlier_data.png",
)

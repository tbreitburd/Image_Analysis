"""!@file mod_2_q_1.py

@brief Python script for question 1 of module 2.

@details This script contains code to fit a line to data using L1 and L2 minimization.

@author T. Breitburd on 09/06/2024"""

import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
from plot_funcs import plot_fitted_lines

# https://docs.scipy.org/doc/scipy/tutorial/optimize.html#
# constrained-minimization-of-multivariate-scalar-functions-minimize

# Load the data
y_line = np.loadtxt("./data/y_line.txt")
y_outlier = np.loadtxt("./data/y_outlier_line.txt")

# Define the x values, assuming they are given by indices
x = np.arange(len(y_line))

# ---------------------------
# First for L1 minimization
# ---------------------------


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

# ---------------------------
# Now for L2 minimization
# ---------------------------

# Fit a linear regression model for the noisy data
model_noise = LinearRegression()
model_noise.fit(x.reshape(-1, 1), y_line)

# Fit a linear regression model for the data with outliers
model_outlier = LinearRegression()
model_outlier.fit(x.reshape(-1, 1), y_outlier)

# Get the results
print("---------------------------")
print("Results for L2 minimization")
print("---------------------------")

# For the noisy data
print("For the noisy data:")
print("a =", model_noise.coef_[0])
print("b =", model_noise.intercept_)

# For the data with outliers
print("For the data with outliers:")
print("a =", model_outlier.coef_[0])
print("b =", model_outlier.intercept_)

# Plot the results
plot_fitted_lines(
    x,
    y_line,
    [model_noise.coef_[0], model_noise.intercept_],
    "L2 Fitted line for noisy data",
    "l2_noisy_data.png",
)
plot_fitted_lines(
    x,
    y_outlier,
    [model_outlier.coef_[0], model_outlier.intercept_],
    "L2 Fitted line for data with outliers",
    "l2_outlier_data.png",
)

# Get R-squared for both fits of the noisy data
r2_l2 = model_noise.score(x.reshape(-1, 1), y_line)

print("R-squared for L2 fit of noisy data:", r2_l2)

# Get fitted line for L1 fit of noisy data
y_line_l1 = result_noise.x[0] * x + result_noise.x[1]

# Get R-squared for L1 fit of noisy data
r2_l1 = 1 - np.sum((y_line - y_line_l1) ** 2) / np.sum((y_line - np.mean(y_line)) ** 2)

print("R-squared for L1 fit of noisy data:", r2_l1)

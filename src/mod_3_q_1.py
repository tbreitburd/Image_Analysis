"""!@file mod_3_q_1.py

@brief This file contains the code for the 1st quesiton of the third module of the assignment.

@details This code tests the convergence of the gradient descent minimization of the
objective function f(x) = 1/2 * x1^2 + x2^2.

@author T. Breitburd on 12/06/2024
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# ---------------------------------------------
# Define some functions
# ---------------------------------------------


# Define the objective function from the paper
def f(x):
    """!@brief Objective function

    @param x: The input vector [x1, x2]
    @return The value of the objective function
    """
    x1, x2 = x

    return (1 / 2) * x1**2 + x2**2


# Define the gradient of that objective function
# for gradient descent
def grad_f(x):
    """!@brief Gradient of the objective function

    @param x: The input vector [x1, x2]
    @return The gradient of the objective function
    """
    x1, x2 = x

    return np.array([x1, 2 * x2])


# Define the epsilon function, the error between the current value and the objective value
def eps(x):
    """!@brief Epsilon, the error between the current value and the objective value

    @param x: The input vector [x1, x2]
    @return The value of epsilon
    """

    return np.linalg.norm(x - x_obj)


# ---------------------------------------------
# Define some parameters
# ---------------------------------------------

# Define the initial and objective values
x_init = np.array([1, 1])
x_obj = np.array([0, 0])

# Calculate the initial difference
epsilon = f(x_init) - f(x_obj)
print("---------------------------------")
print(f"Initial difference = {epsilon}")

# Define the learning rate
L = 2
eta = 1 / L


# ---------------------------------------------
# Perform the gradient descent
# ---------------------------------------------

# Initialize the current value
x_star = x_init.copy()
i = 0

x1 = [x_init[0]]
x2 = [x_init[1]]

# Perform the gradient descent
while epsilon > 1e-2:
    # Update the current value
    x_star = x_star - eta * grad_f(x_star)

    # Append the current value
    x1.append(x_star[0])
    x2.append(x_star[1])

    # Calculate the new error
    epsilon = f(x_star) - f(x_obj)

    # Count the iteration
    i += 1


# ---------------------------------------------
# Print the results
# ---------------------------------------------
print("---------------------------------")
print(f"Final value = {x_star}")
print(f"Final difference = {epsilon}")
print(f"Number of iterations = {i}")


# ---------------------------------------------
# Plot some results
# ---------------------------------------------

plt.style.use("seaborn-v0_8-darkgrid")

plt.figure(figsize=(4, 4))

plt.plot(x1, "o-", label="x1")
plt.plot(x2, "x-", label="x2")

plt.xlabel("Iteration")
plt.ylabel("Value")
plt.legend()
plt.title("Convergence of the gradient descent")

plt.tight_layout()

# Save the plot
cur_dir = os.getcwd()
plots_dir = os.path.join(cur_dir, "Plots")
os.makedirs(plots_dir, exist_ok=True)

plot_dir = os.path.join(plots_dir, "q3_1_Convergence.png")
plt.savefig(plot_dir)

plt.close()

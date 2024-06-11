"""!@file mod_3_q_1.py

@brief This file contains the code for the 1st quesiton of the third module of the assignment.

@details This code tests the convergence of the gradient descent minimization of the
objective function
"""

import numpy as np


def f(x):
    """!@brief Objective function

    @param x: The input vector [x1, x2]
    @return The value of the objective function
    """
    x1, x2 = x

    return (1 / 2) * x1**2 + x2**2


def grad_f(x):
    """!@brief Gradient of the objective function

    @param x: The input vector [x1, x2]
    @return The gradient of the objective function
    """
    x1, x2 = x

    return np.array([x1, 2 * x2])


x_init = np.array([1, 1])

x_obj = np.array([0, 0])

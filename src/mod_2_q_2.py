"""!@file mod_2_q_2.py
@brief Python script for question 2 of module 2."""


import numpy as np
from plot_funcs import plot_signal_vector

# ------------------------------------------
# a) Generate vector, 10 non-zero entries
# ------------------------------------------

# Set seed
np.random.seed(75016)

# Create a vector of length 100
L = 100
vector = np.zeros(L)

# Set 10 non-zero coeffs [0,1]
idx = np.random.permutation(L)[:10]
vector[idx] = np.random.choice(range(1, 101), 10) / 100


# ------------------------------------------
# b) Add random Gaussian noise
# ------------------------------------------

# Add Gaussian noise
sigma = 0.05
vector = vector + np.random.normal(0, sigma, L)

# Plot the vector
plot_signal_vector(vector)

# ------------------------------------------
# c)-d) Uniformly/Randomly undersample and
# compute 4 * zero-filled inverse FT
# ------------------------------------------

# Take 32 uniformly spaced samples from the noisy vector
unif_mask = np.linspace(0, 99, 32, dtype=int)
data_unif = vector[unif_mask]

# Take 32 randomly spaced samples from the noisy vector
rand_mask = np.random.choice(range(100), 32, replace=False)
data_rand = vector[rand_mask]

# Compute the inverse FT and multiply by 4
unif_sig = np.fft.ifft(data_unif) * 4
rand_sig = np.fft.ifft(data_rand) * 4

# Zero-fill the signals
alia_signal = np.zeros(100, dtype=complex)
alia_signal[unif_mask] = unif_sig

noisy_signal = np.zeros(100, dtype=complex)
noisy_signal[rand_mask] = rand_sig

# Plot the signals
plot_signal_vector(
    np.abs(alia_signal), "Uniformly Undersampled Signal", "q2.2_uniform_signal.png"
)
plot_signal_vector(
    np.abs(noisy_signal), "Randomly Undersampled Signal", "q2.2_random_signal.png"
)

# ------------------------------------------
# e) Reconstruct the signal using the ISTA
# ------------------------------------------

# Define the iterative soft thresholding algorithm


def soft_thresh(x, lam):
    return np.sign(x) * np.maximum(np.abs(x) - lam, 0)


def ista(y, A, lam, num_iter=100):
    x_hat = np.zeros(100)
    return x_hat

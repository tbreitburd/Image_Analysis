"""!@file mod_2_q_2.py

@brief Python script for question 2 of module 2.

@details This script contains code to...

@author T. Breitburd on 09/06/2024"""


import numpy as np
from plot_funcs import plot_signal_vector
import matplotlib.pyplot as plt

# ------------------------------------------
# Define the signal reconstruction functions
# ------------------------------------------


def fftc(x):
    """!@brief Compute the centered FFT of a signal.

    @param x: The input signal.

    @return The centered FFT of the input signal.
    """

    return 1 / np.sqrt(len(x)) * np.fft.fftshift(np.fft.fft(np.fft.ifftshift(x)))


def ifftc(x):
    """!@brief Compute the centered inverse FFT of a signal.

    @param x: The input signal.

    @return The centered inverse FFT of the input signal.
    """

    return np.sqrt(len(x)) * np.fft.ifftshift(np.fft.ifft(np.fft.fftshift(x)))


def SoftThresh(x, lam):
    """!@brief Perform soft thresholding on a signal.

    @param x: The input signal.
    @param lam: The soft threshold value.

    @return The soft thresholded signal.
    """

    # Compute the difference between the signal and the threshold
    diff = abs(x) - lam

    # Apply the soft thresholding
    soft_thresh = (diff > 0.0) * diff * x / abs(x)

    return soft_thresh


# Define the ISTA function
def iterative_soft_thresholding(X, Y, lam, n_iter=100):
    """!@brief Perform iterative soft thresholding on a signal.

    @param X: The signal in the frequency domain.
    @param Y: The observed signal.
    @param lam: The soft threshold value.
    @param n_iter: The number of iterations.

    @return The reconstructed signal.
    """

    for i in range(n_iter):
        # Compute the inverse FT of the signal
        x = ifftc(X)
        # Apply soft-thresholding
        x = SoftThresh(x, lam)
        # Compute the FT of the signal back
        # and apply data consistency constraint
        X = fftc(x)
        X[Y != 0] = Y[Y != 0]

    return ifftc(X)  # Final inverse transform to time domain


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

# Compute the FT of the noisy vector
vector_ft = fftc(vector)

# Take 32 uniformly spaced samples from the noisy vector
unif_mask = np.linspace(0, 99, 32, dtype=int)
X_unif = np.zeros(100, dtype=complex)
X_unif[unif_mask] = vector_ft[unif_mask]
x_unif = ifftc(X_unif) * 4

# Take 32 randomly spaced samples from the noisy vector
rand_mask = np.random.choice(100, 32, replace=False)
X_rand = np.zeros(100, dtype=complex)
X_rand[rand_mask] = vector_ft[rand_mask]
x_rand = ifftc(X_rand) * 4

# Plot the signals
plot_signal_vector(
    np.real(x_unif), "Uniformly Undersampled Signal", "q2.2_uniform_signal.png"
)
plot_signal_vector(
    np.real(x_rand), "Randomly Undersampled Signal", "q2.2_random_signal.png"
)

# ------------------------------------------
# e) Reconstruct the signal using the ISTA
# ------------------------------------------

X_unif_temp = X_unif.copy()
X_rand_temp = X_rand.copy()

x_unif_rec = iterative_soft_thresholding(X_unif_temp, X_unif, 0.04)
x_rand_rec = iterative_soft_thresholding(X_rand_temp, X_rand, 0.04)


# ------------------------------------------
# Plot the reconstructed signals
# ------------------------------------------

plt.figure(figsize=(12, 12))

plt.subplot(5, 1, 1)
plt.stem(np.real(vector), label="Original Signal")
plt.legend()

plt.subplot(5, 1, 2)
plt.stem(np.real(x_unif), label="Undersampled Noisy Signal (Uniform)")
plt.legend()

plt.subplot(5, 1, 3)
plt.stem(np.real(x_rand), label="Undersampled Noisy Signal (Random)")
plt.legend()

plt.subplot(5, 1, 4)
plt.stem(np.real(x_unif_rec), label="Reconstructed Signal (Uniform)")
plt.legend()

plt.subplot(5, 1, 5)
plt.stem(np.real(x_rand_rec), label="Reconstructed Signal (Random)")
plt.legend()

plt.tight_layout()
plt.show()

"""!@file mod_3_LGD.py

@author T.Breitburd and Course Instructor on 12/06/24
"""

import os
import numpy as np
import astra
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
import torch
import odl
import odl.contrib.torch as odl_torch
from neur_nets import LGD_net

astra.test()


# -----------------------------------------------------------
# Plot functions from the given notebook (i.e. authored by the course instructor)
# -----------------------------------------------------------


def plot_grd_truth_FBP(phantom_np, data_np, fbp_np, data_range, psnr_fbp, ssim_fbp):
    """!@brief Function to plot the ground-truth, sinogram, and FBP image

    @param phantom_np the ground-truth image, numpy array
    @param data_np the sinogram data, numpy array
    @param fbp_np the FBP image, numpy array
    @param data_range the data range, float
    @param psnr_fbp the PSNR of the FBP image, float
    @param ssim_fbp the SSIM of the FBP image, float

    @return None"""

    plt.figure(figsize=(9, 4))

    plt.subplot(131)
    plt.imshow(phantom_np.transpose(), cmap="bone")
    plt.xticks([])
    plt.yticks([])
    plt.title("ground-truth")

    plt.subplot(132)
    plt.imshow(data_np, cmap="bone")
    plt.xticks([])
    plt.yticks([])
    plt.title("sinogram")

    plt.subplot(133)
    plt.imshow(fbp_np.transpose(), cmap="bone")
    plt.xticks([])
    plt.yticks([])
    plt.title("FBP")
    data_range = np.max(phantom_np) - np.min(phantom_np)
    psnr_fbp = compare_psnr(phantom_np, fbp_np, data_range=data_range)
    ssim_fbp = compare_ssim(phantom_np, fbp_np, data_range=data_range)
    plt.xlabel("PSNR: {:.2f} dB, SSIM: {:.2f}".format(psnr_fbp, ssim_fbp))
    plt.gcf().set_size_inches(9.0, 6.0)

    # Save the plot
    cur_dir = os.getcwd()
    plots_dir = os.path.join(cur_dir, "Plots")
    os.makedirs(plots_dir, exist_ok=True)

    plot_dir = os.path.join(plots_dir, "q3.1_grd_truth_FBP.png")
    plt.savefig(plot_dir)


def plot_ADMM_FBP(
    phantom_np, fbp_np, x_admm_np, data_range, psnr_fbp, ssim_fbp, psnr_tv, ssim_tv
):
    """!@brief Function to plot the ground-truth, FBP, and ADMM image

    @param phantom_np the ground-truth image, numpy array
    @param fbp_np the FBP image, numpy array
    @param x_admm_np the ADMM image, numpy array
    @param data_range the data range, float
    @param psnr_fbp the PSNR of the FBP image, float
    @param ssim_fbp the SSIM of the FBP image, float
    @param psnr_tv the PSNR of the TV image, float

    @return None"""

    plt.figure(figsize=(9, 4))

    plt.subplot(131)
    plt.imshow(phantom_np.transpose(), cmap="bone")
    plt.xticks([])
    plt.yticks([])
    plt.title("ground-truth")

    plt.subplot(132)
    plt.imshow(fbp_np.transpose(), cmap="bone")
    plt.xticks([])
    plt.yticks([])
    plt.title("FBP")
    plt.xlabel("PSNR: {:.2f} dB, SSIM: {:.2f}".format(psnr_fbp, ssim_fbp))

    plt.subplot(133)
    plt.imshow(x_admm_np.transpose(), cmap="bone")
    plt.xticks([])
    plt.yticks([])
    plt.title("TV")
    psnr_tv = compare_psnr(phantom_np, x_admm_np, data_range=data_range)
    ssim_tv = compare_ssim(phantom_np, x_admm_np, data_range=data_range)
    plt.xlabel("PSNR: {:.2f} dB, SSIM: {:.2f}".format(psnr_tv, ssim_tv))
    plt.gcf().set_size_inches(9.0, 6.0)

    # Save the plot
    cur_dir = os.getcwd()
    plots_dir = os.path.join(cur_dir, "Plots")
    os.makedirs(plots_dir, exist_ok=True)

    plot_dir = os.path.join(plots_dir, "q3.1_grd_truth_FBP_ADMM.png")
    plt.savefig(plot_dir)


def plot_ADMM_FBP_LGD(
    phantom_np,
    fbp_np,
    x_admm_np,
    lgd_recon_np,
    data_range,
    psnr_fbp,
    ssim_fbp,
    psnr_tv,
    ssim_tv,
):
    """!@brief Function to plot the ground-truth, FBP, ADMM, and LGD image

    @param phantom_np the ground-truth image, numpy array
    @param fbp_np the FBP image, numpy array
    @param x_admm_np the ADMM image, numpy array
    @param lgd_recon_np the LGD image, numpy array
    @param data_range the data range, float
    @param psnr_fbp the PSNR of the FBP image, float
    @param ssim_fbp the SSIM of the FBP image, float
    @param psnr_tv the PSNR of the TV image, float
    @param ssim_tv the SSIM of the TV image, float

    @return None"""

    plt.figure(figsize=(12, 4))

    plt.subplot(141)
    plt.imshow(phantom_np.transpose(), cmap="bone")
    plt.xticks([])
    plt.yticks([])
    plt.title("ground-truth")

    plt.subplot(142)
    plt.imshow(fbp_np.transpose(), cmap="bone")
    plt.xticks([])
    plt.yticks([])
    plt.title("FBP")
    plt.xlabel("PSNR: {:.2f} dB, SSIM: {:.2f}".format(psnr_fbp, ssim_fbp))

    plt.subplot(143)
    plt.imshow(x_admm_np.transpose(), cmap="bone")
    plt.xticks([])
    plt.yticks([])
    plt.title("TV")
    plt.xlabel("PSNR: {:.2f} dB, SSIM: {:.2f}".format(psnr_tv, ssim_tv))
    plt.gcf().set_size_inches(9.0, 6.0)

    plt.subplot(144)
    plt.imshow(lgd_recon_np.transpose(), cmap="bone")
    plt.xticks([])
    plt.yticks([])
    plt.title("LGD")
    psnr_lgd = compare_psnr(phantom_np, lgd_recon_np, data_range=data_range)
    ssim_lgd = compare_ssim(phantom_np, lgd_recon_np, data_range=data_range)
    plt.xlabel("PSNR: {:.2f} dB, SSIM: {:.2f}".format(psnr_lgd, ssim_lgd))
    plt.gcf().set_size_inches(12.0, 6.0)

    # Save the plot
    cur_dir = os.getcwd()
    plots_dir = os.path.join(cur_dir, "Plots")
    os.makedirs(plots_dir, exist_ok=True)

    plot_dir = os.path.join(plots_dir, "q3.1_grd_truth_FBP_ADMM_LGD.png")
    plt.savefig(plot_dir)


# -----------------------------------------------------------
# Set up the forward operator (ray transform) in ODL
# -----------------------------------------------------------
print("Setting up the forward operator in ODL...")

# Reconstruction space: functions on the rectangle [-20, 20]^2
img_size = 256  # discretized with 256 samples per dimension
reco_space = odl.uniform_discr(
    min_pt=[-20, -20], max_pt=[20, 20], shape=[img_size, img_size], dtype="float32"
)
# Make a parallel beam geometry with flat detector, using number of angles = num_angles
num_angles = 30
geometry = odl.tomo.parallel_beam_geometry(reco_space, num_angles=num_angles)

# Create the forward operator, adjoint operator, and the FBO operator in ODL
fwd_op_odl = odl.tomo.RayTransform(reco_space, geometry)
fbp_op_odl = odl.tomo.fbp_op(fwd_op_odl, filter_type="Ram-Lak", frequency_scaling=0.6)
adj_op_odl = fwd_op_odl.adjoint

# Create phantom and noisy projection data in ODL
phantom_odl = odl.phantom.shepp_logan(reco_space, modified=True)
data_odl = fwd_op_odl(phantom_odl)
data_odl += odl.phantom.white_noise(fwd_op_odl.range) * np.mean(data_odl) * 0.1
fbp_odl = fbp_op_odl(data_odl)

# convert the image and the sinogram to numpy arrays
phantom_np = phantom_odl.__array__()
fbp_np = fbp_odl.__array__()
data_np = data_odl.__array__()
print("Sinogram size = {}".format(data_np.shape))

# Compute the PSNR and SSIM between the ground truth and the FBP reconstruction
data_range = np.max(phantom_np) - np.min(phantom_np)
psnr_fbp = compare_psnr(phantom_np, fbp_np, data_range=data_range)
ssim_fbp = compare_ssim(phantom_np, fbp_np, data_range=data_range)

# -----------------------------------------------------------
# Display the ground truth, FBP reconstruction, and the sinogram
# -----------------------------------------------------------
plot_grd_truth_FBP(phantom_np, data_np, fbp_np, data_range, psnr_fbp, ssim_fbp)


# -----------------------------------------------------------
# Let's solve the TV reconstruction problem
# using the linearized ADMM algorithm (implemented in ODL).
# -----------------------------------------------------------
print("Solving the TV reconstruction problem using ADMM...")

# In this example we solve the optimization problem:
# min_x f(x) + g(Lx) = ||A(x) - y||_2^2 + lam * ||grad(x)||_1,
# Where:
# - ``A`` is a parallel beam ray transform,
# - ``grad`` is the spatial gradient,
# - ``y`` given noisy data.

# The problem is rewritten in decoupled form as:
# min_x g(L(x))
# with a separable sum ``g`` of functionals and the stacked operator ``L``:

# g(z) = ||z_1 - g||_2^2 + lam * ||z_2||_1,
#                ( A(x)    )
#     z = L(x) = ( grad(x) ).

# Gradient operator for the TV part
grad = odl.Gradient(reco_space)

# Stacking of the two operators
L = odl.BroadcastOperator(fwd_op_odl, grad)

# Data matching and regularization functionals
data_fit = odl.solvers.L2NormSquared(fwd_op_odl.range).translated(data_odl)
lam = 0.015
reg_func = lam * odl.solvers.L1Norm(grad.range)
g = odl.solvers.SeparableSum(data_fit, reg_func)

# We don't use the f functional, setting it to zero
f = odl.solvers.ZeroFunctional(L.domain)


# --- Select parameters and solve using ADMM ---

# Estimated operator norm, add 10 percent for some safety margin
op_norm = 1.1 * odl.power_method_opnorm(L, maxiter=20)

niter = 200  # Number of iterations
sigma = 2.0  # Step size for g.proximal
tau = sigma / op_norm**2  # Step size for f.proximal

# Optionally pass a callback to the solver to display intermediate results
callback = odl.solvers.CallbackPrintIteration(step=10) & odl.solvers.CallbackShow(
    step=10
)

# Choose a starting point
x_admm_odl = L.domain.zero()

# Run the algorithm
odl.solvers.admm_linearized(x_admm_odl, f, g, L, tau, sigma, niter, callback=None)
x_admm_np = x_admm_odl.__array__()

# Let's display the image reconstructed by ADMM and compare it with FBP
plot_ADMM_FBP(phantom_np, fbp_np, x_admm_np)

# -----------------------------------------------------------
# Set up the LGD algorithm in PyTorch
# -----------------------------------------------------------

# Check if a GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Now, we need to cast the ODL operators as torch operators
# so that we can integrate them with other learnable units.
# For brevity, we will omit the suffix _torch in the variable and operator names.
fwd_op = odl_torch.OperatorModule(fwd_op_odl).to(device)
adj_op = odl_torch.OperatorModule(fwd_op_odl.adjoint).to(device)
fbp_op = odl_torch.OperatorModule(fbp_op_odl).to(device)

# Let's compute a reasonable initial value for the step-size as step_size = 1/L,
# where L is the spectral norm of the forward operator.
op_norm = 1.1 * odl.power_method_opnorm(fwd_op_odl)
step_size = 1 / op_norm


# Initialize the LGD network
lgd_net = LGD_net(fwd_op=fwd_op, adj_op=adj_op, step_size=step_size).to(
    device
)  # realize the network and export it to GPU

# Print the number of learnable parameters in the LGD network
num_learnable_params = sum(p.numel() for p in lgd_net.parameters() if p.requires_grad)
print("number of model parameters = {}".format(num_learnable_params))

# Convert the noisy sinogram data to a torch tensor
y = torch.from_numpy(data_np).to(device).unsqueeze(0)

# Compute the FBP reconstruction as the initial guess
x_init = fbp_op(y)

# Convert the ground-truth image to a torch tensor
ground_truth = torch.from_numpy(phantom_np).to(device).unsqueeze(0)

# Define the loss and the optimizer
mse_loss = torch.nn.MSELoss()
optimizer = torch.optim.Adam(lgd_net.parameters(), lr=1e-4)
num_epochs = 2000

# Training loop
for epoch in range(0, num_epochs):
    # ----------------------------------------
    # Authored section of code by T. Breitburd
    # ----------------------------------------

    # Zero the gradients
    optimizer.zero_grad()

    # Pass the input through the network, to get the reconstruction
    recon = lgd_net(y, x_init)

    # Compute the loss
    loss = mse_loss(recon, ground_truth)

    # Backward pass and optimization step
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print("Epoch = {}, Loss = {}".format(epoch, loss.item()))

# Convert the reconstruction to a numpy array
lgd_recon_np = recon.detach().cpu().numpy().squeeze()

# Let's display the reconstructed images by LGD and compare it with FBP and ADMM
plot_ADMM_FBP_LGD(phantom_np, fbp_np, x_admm_np, lgd_recon_np)

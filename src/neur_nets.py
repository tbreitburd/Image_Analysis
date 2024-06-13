"""!@file neur_nets.py

@brief This file contains the neural network classes to be used in the LGD algorithm,
as well as the definitiom of some of the operators used in the LGD algorithm.

@details The prox_net class defines the proximal operator of the gradient of the data term,
while the LGD_net class defines the LGD algorithm, which itself uses the prox_net class.

@author T.Breitburd and Course Instructor on 12/06/24
"""

import torch
import torch.nn as nn

# Check if a GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class prox_net(nn.Module):
    """!@brief Proximal operator of the gradient of the data term.

    @details This class defines the
    proximal operator of the gradient of the data term, which is used in the LGD algorithm.

    @param n_in_channels: number of input channels
    @param n_out_channels: number of output channels
    @param n_filters: number of filters
    @param kernel_size: size of the convolutional kernel

    @return dx:
    """

    def __init__(self, n_in_channels=2, n_out_channels=1, n_filters=32, kernel_size=3):
        super(prox_net, self).__init__()

        # Define the padding, and convolutional layers
        self.pad = (kernel_size - 1) // 2

        self.conv1 = nn.Conv2d(
            n_in_channels,
            out_channels=n_filters,
            kernel_size=kernel_size,
            stride=1,
            padding=self.pad,
            bias=True,
        )

        self.conv2 = nn.Conv2d(
            n_filters,
            n_filters,
            kernel_size=kernel_size,
            stride=1,
            padding=self.pad,
            bias=True,
        )

        self.conv3 = nn.Conv2d(
            n_filters,
            out_channels=1,
            kernel_size=kernel_size,
            stride=1,
            padding=self.pad,
            bias=True,
        )

        # Define the activation functions
        self.act1 = nn.PReLU(num_parameters=1, init=0.25)
        self.act2 = nn.PReLU(num_parameters=1, init=0.25)

    # -----------------------------------------------------------
    # Authored section of code by T. Breitburd
    # -----------------------------------------------------------
    def forward(self, x, u):
        """!@brief Forward pass of the proximal operator.

        @details This function computes the forward pass of the proximal operator.

        @param x: current iterate x of the LGD algorithm
        @param u: gradient of the loss function at x

        @return dx: output of the proximal operator
        """

        # Pass the input through the network
        dx = self.act1(self.conv1(torch.cat((x, u), dim=0)))
        dx = self.act2(self.conv2(dx))
        dx = self.conv3(dx)

        return dx


class LGD_net(nn.Module):
    """!@brief LGD algorithm.

    @details This class defines the
    LGD algorithm, which itself uses the prox_net class.

    @param fwd_op: forward operator
    @param adj_op: adjoint operator
    @param niter: number of iterations
    @param step_size: step size of the LGD algorithm


    @return x: output/reconstruction of the LGD algorithm
    """

    def __init__(self, fwd_op, adj_op, step_size, niter=5):
        super(LGD_net, self).__init__()

        self.niter = niter

        self.prox = nn.ModuleList([prox_net().to(device) for i in range(self.niter)])

        self.step_size = nn.Parameter(step_size * torch.ones(self.niter).to(device))

        self.fwd_op = fwd_op
        self.adj_op = adj_op

    # -----------------------------------------------------------
    # Authored section of code by T. Breitburd
    # -----------------------------------------------------------
    def forward(self, y, x_init):
        """!@brief Forward pass of the LGD algorithm.

        @details This function computes the forward pass of the LGD algorithm.

        @param y: input data
        @param x_init: initial value of x

        @return x: output/reconstruction of the LGD algorithm
        """

        # Initialize x
        x = x_init

        # Loop over the number of iterations
        for k in range(self.niter):
            # Compute the gradient, using the adjoint and forward operators
            grad = self.adj_op(self.fwd_op(x) - y)

            # Compute the next value of x
            x = x + self.step_size[k] * self.prox[k](x, grad)

        return x

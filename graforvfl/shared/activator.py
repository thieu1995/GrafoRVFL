#!/usr/bin/env python
# Created by "Thieu" at 10:02, 10/08/2023 ----------%
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

"""
This module provides a comprehensive collection of activation functions used in machine learning and deep learning.
Activation functions play a critical role in neural networks by introducing non-linearity, enabling the network to
learn and approximate complex patterns in data.

Functions:
----------
- none(x):
    A no-op function that returns the input as is.

- relu(x):
    Rectified Linear Unit (ReLU), returns the input if positive, otherwise returns zero.

- leaky_relu(x, alpha=0.01):
    Leaky ReLU allows a small gradient when the input is negative.

- celu(x, alpha=1.0):
    Continuously Differentiable Exponential Linear Unit, a smooth alternative to ReLU.

- prelu(x, alpha=0.5):
    Parametric ReLU, where the slope for negative inputs is a learnable parameter.

- gelu(x, alpha=0.044715):
    Gaussian Error Linear Unit, combines tanh approximation for smooth activation.

- elu(x, alpha=1):
    Exponential Linear Unit, returns an exponential for negative inputs.

- selu(x, alpha=1.67326324, scale=1.05070098):
    Scaled Exponential Linear Unit, normalizes outputs for self-normalizing networks.

- rrelu(x, lower=1./8, upper=1./3):
    Randomized Leaky ReLU, introduces randomized slopes for negative inputs.

- tanh(x):
    Hyperbolic tangent function, outputs values between -1 and 1.

- hard_tanh(x, lower=-1., upper=1.):
    A clipped version of the tanh function.

- sigmoid(x):
    Logistic sigmoid function, outputs values between 0 and 1.

- hard_sigmoid(x, lower=-2.5, upper=2.5):
    A piecewise linear approximation of the sigmoid function.

- log_sigmoid(x):
    Logarithmic sigmoid function for numerical stability.

- swish(x):
    Swish (or SiLU), smooth and bounded non-linearity.

- hard_swish(x, lower=-3., upper=3.):
    A piecewise linear approximation of the swish function.

- soft_plus(x, beta=1.0):
    Smooth approximation of the ReLU function.

- mish(x, beta=1.0):
    Mish activation, smooth non-monotonic function.

- soft_sign(x):
    Smooth approximation of the sign function.

- tanh_shrink(x):
    Difference between input and tanh, providing a shrinkage effect.

- soft_shrink(x, alpha=0.5):
    Threshold-based shrinkage operator with soft boundaries.

- hard_shrink(x, alpha=0.5):
    Hard thresholding function with a predefined alpha.

- softmin(x):
    Normalizes the negative inputs into a probability distribution.

- softmax(x):
    Converts inputs into a probability distribution over multiple classes.

- log_softmax(x):
    Numerically stable logarithmic version of softmax.

Aliases:
--------
- silu(x): Alias for swish(x).
"""

import numpy as np


def none(x):
    return x


def relu(x):
    return np.maximum(0, x)


def leaky_relu(x, alpha=0.01):
    return np.maximum(alpha * x, x)


def celu(x, alpha=1.0):
    return np.maximum(0, x) + np.minimum(0, alpha*(np.exp(x / alpha) - 1))


def prelu(x, alpha=0.5):
    return np.where(x < 0, alpha*x, x)


def gelu(x, alpha=0.044715):
    return x/2 * (1 + np.tanh(np.sqrt(2.0/np.pi) * (x + alpha*x**3)))


def elu(x, alpha=1):
    return np.where(x < 0, alpha * (np.exp(x) - 1), x)


def selu(x, alpha=1.67326324, scale=1.05070098):
    return np.where(x < 0, scale*alpha*(np.exp(x) - 1), scale*x)


def rrelu(x, lower=1./8, upper=1./3):
    alpha = np.random.uniform(lower, upper)
    return np.where(x < 0, alpha*x, x)


def tanh(x):
    return np.tanh(x)


def hard_tanh(x, lower=-1., upper=1.):
    return np.where(x < lower, -1, np.where(x > upper, upper, x))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def hard_sigmoid(x, lower=-2.5, upper=2.5):
    return np.where(x < lower, 0, np.where(x > upper, 1, 0.2*x + 0.5))


def log_sigmoid(x):
    return -np.log(1 + np.exp(-x))


def swish(x):
    # = silu (pytorch)
    return x / (1. + np.exp(-x))


def hard_swish(x, lower=-3., upper=3.):
    return np.where(x <= lower, 0, np.where(x >= upper, x, x*(x + 3)/6))


def soft_plus(x, beta=1.0):
    return 1.0/beta * np.log(1 + np.exp(beta * x))


def mish(x, beta=1.0):
    return x * np.tanh(1.0/beta * np.log(1 + np.exp(beta * x)))


def soft_sign(x):
    return x / (1 + np.abs(x))


def tanh_shrink(x):
    return x - np.tanh(x)


def soft_shrink(x, alpha=0.5):
    return np.where(x < -alpha, x + alpha, np.where(x > alpha, x - alpha, 0))


def hard_shrink(x, alpha=0.5):
    return np.where((x > -alpha) & (x < alpha), x, 0)


def softmin(x):
    exp_x = np.exp(-x)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def log_softmax(x):
    log_exp_x = x - np.max(x, axis=-1, keepdims=True)
    log_exp_x = log_exp_x - np.log(np.sum(np.exp(log_exp_x), axis=-1, keepdims=True))
    return log_exp_x


silu = swish

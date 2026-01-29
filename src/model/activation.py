from typing import Dict, Optional, Callable, Any
import numpy as np


def tanh(x):
    return np.tanh(x)


def tanh_derivative(x):
    t = np.tanh(x)
    return 1 - t * t


def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return (x > 0).astype(float)


def leaky_relu(x, alpha: float = 0.25):
    return np.where(x > 0, x, alpha * x)


def leaky_relu_derivative(x, alpha=0.25):
    return np.where(x > 0, 1.0, alpha)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)


def softplus(x):
    return np.log(1 + np.exp(x))


def softplus_derivative(x):
    return sigmoid(x)


def swish(x, beta: float = 1.0):
    return x * sigmoid(beta * x)


def swish_derivative(x, beta=1.0):
    s = sigmoid(beta * x)
    return s + beta * x * s * (1 - s)


def elu(x, alpha: float = 1.0):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))


def elu_derivative(x, alpha=1.0):
    return np.where(x > 0, 1.0, alpha * np.exp(x))


def identity(x):
    return x

def identity_derivative(x):
    return np.ones_like(x)
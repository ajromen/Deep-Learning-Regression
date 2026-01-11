from typing import Dict, Optional, Callable, Any
import numpy as np

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha: float = 0.25):
    return np.where(x > 0, x, alpha * x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softplus(x):
    return np.log(1 + np.exp(x))

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)

def swish(x, beta: float = 1.0):
    return x * sigmoid(beta * x)

def elu(x, alpha: float = 1.0):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

activation_functions: Dict[str, Optional[Callable[..., Any]]] = {
    "tanh": tanh,
    "relu": relu,
    "lrelu": leaky_relu,
    "sigmoid": sigmoid,
    "softplus": softplus,
    "softmax": softmax,
    "swish": swish,
    "elu": elu,
    "all": None
}
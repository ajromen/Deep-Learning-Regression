from src.model.activation import *
from src.optimizers.adagrad import Adagrad
from src.optimizers.adam import Adam
from src.optimizers.gradient_descent import GradientDescent
from src.optimizers.rmsprop import RMSProp

optimizer_functions = {
    "gradient": GradientDescent,
    "adagrad": Adagrad,
    "rmsprop": RMSProp,
    "adam": Adam,
    "all": None
}

# samo sredina bez inputa i outputa oni se dodaju naknadno
layers_types: dict[str, list[int] | None] = {
    "shallow": [100],
    "deep": [30, 30, 30],
    "narrow": [10, 10],
    "default": [50, 50],
    "all": None
}

activation_functions = {
    "tanh": (tanh, tanh_derivative),
    "relu": (relu, relu_derivative),
    "lrelu": (leaky_relu, leaky_relu_derivative),
    "sigmoid": (sigmoid, sigmoid_derivative),
    "softplus": (softplus, softplus_derivative),
    "swish": (swish, swish_derivative),
    "elu": (elu, elu_derivative),
    "all": None
}

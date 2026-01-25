import numpy as np


class Layer:
    def __init__(self, input_size: int, output_size: int, activation, activation_derivative):
        self.W = np.random.randn(input_size, output_size) * 0.01
        self.b = np.zeros((1, output_size))
        self.activation = activation
        self.activation_derivative = activation_derivative
        self.dW = None
        self.X = None
        self.Z = None
        self.A = None
        self.db = None

    def forward(self, X):
        self.X = X
        self.Z = X @ self.W + self.b
        self.A = self.activation(self.Z)
        return self.A

    def backward(self, dA):
        dZ = dA * self.activation_derivative(self.Z)
        self.dW = self.X.T @ dZ
        self.db = np.sum(dZ, axis=0, keepdims=True)
        return dZ @ self.W.T

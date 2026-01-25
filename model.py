from encodings import latin_1
import numpy as np

import options
from layer import Layer
from lsloss import LSLoss
from optimizers.optimizer import Optimizer


class Model:
    def __init__(self, data: np.ndarray, train_size: float, input_size: int, output_size: int, learning_rate: float,
                 epochs: int, layers_sizes, optimizer, activation, name: str = ""):
        self.data = data
        self.train_size = train_size
        self.learning_rate = learning_rate
        self.input_size = input_size
        self.output_size = output_size
        self.epochs = epochs
        self.optimizer: Optimizer = optimizer(self.learning_rate)
        self.activation, self.activation_derivative = activation
        self.layers_sizes = [input_size] + options.layers_types[layers_sizes] + [output_size]
        self.loss = LSLoss()
        self.name = name

        self._split_train_test()
        self._create_layers()

    def _split_train_test(self):
        n = len(self.data)
        split_idx = int(n * self.train_size)
        train = self.data[:split_idx]
        test = self.data[split_idx:]

        self.x_train = train[:, :-self.output_size]
        self.y_train = train[:, -self.output_size:]

        self.x_test = test[:, :-self.output_size]
        self.y_test = test[:, -self.output_size:]

    def _create_layers(self):
        layers = []
        for i in range(len(self.layers_sizes) - 1):
            layers.append(
                Layer(self.layers_sizes[i], self.layers_sizes[i + 1], self.activation, self.activation_derivative)
            )
        self.layers: list[Layer] = layers

    def run(self):
        X = self.x_train
        Y = self.y_train
        for _ in range(self.epochs):
            y_hat = self.forward_pass(X)
            loss = self.loss.calculate_loss(y_hat, Y)
            self.backward_pass(y_hat, Y)
            self.update_params()

    def forward_pass(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward_pass(self, y_hat, Y):
        grad = self.loss.calculate_loss_derivative(y_hat, Y)
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def update_params(self):
        for layer in self.layers:
            self.optimizer.step(layer)

    def test(self):
        X = self.x_test
        Y = self.y_test
        y_hat = self.forward_pass(X)
        return self.loss.calculate_loss(y_hat, Y)

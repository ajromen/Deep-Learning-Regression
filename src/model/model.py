from encodings import latin_1
import numpy as np

from src.model import options
from src.model.layer import Layer
from src.model.lsloss import LSLoss
from src.optimizers.optimizer import Optimizer


class Model:
    def __init__(self, train_data: np.ndarray, test_data: np.ndarray,  input_size: int, output_size: int, learning_rate: float,
                 epochs: int, layers_sizes, optimizer, activation, batch_size: int = 32, name: str = "", c_names ="",verbose = False):
        self.train_data = train_data
        self.test_data = test_data
        self.learning_rate = learning_rate
        self.input_size = input_size
        self.output_size = output_size
        self.epochs = epochs
        self.optimizer: Optimizer = optimizer(self.learning_rate)
        self.activation, self.activation_derivative = activation
        self.layers_sizes = [input_size] + options.layers_types[layers_sizes] + [output_size]
        self.loss = LSLoss()
        self.name = name
        self.batch_size = batch_size
        self.c_names = c_names
        self.verbose = False

        self._split_train_test()
        self._create_layers()

    def _split_train_test(self):

        self.x_train = self.train_data[:, :-self.output_size]
        self.y_train = self.train_data[:, -self.output_size:]

        self.x_test = self.test_data[:, :-self.output_size]
        self.y_test = self.test_data[:, -self.output_size:]

    def _create_layers(self):
        layers = []
        for i in range(len(self.layers_sizes) - 1):
            layers.append(
                Layer(self.layers_sizes[i], self.layers_sizes[i + 1], self.activation, self.activation_derivative)
            )
        self.layers: list[Layer] = layers

        # poslednji ne bi trebao da ima activation? mozda
        # self.layers[-1].activation = act.identity
        # self.layers[-1].activation_derivative = act.identity_derivative

    def _run_batches(self, X, Y):
        n = X.shape[0]
        indices = np.random.permutation(n)

        for start in range(0, n, self.batch_size):
            end = start + self.batch_size
            batch_idx = indices[start:end]
            yield X[batch_idx], Y[batch_idx]

    def run(self):
        X = self.x_train
        Y = self.y_train
        for i in range(self.epochs):
            for Xb, Yb in self._run_batches(X, Y):
                y_hat = self.forward_pass(Xb)
                self.backward_pass(y_hat, Yb)
                self.update_params()
            if i%1000==0 and self.verbose:
                print("\tLoss at iteration "+str(i)+". "+str(self.test()))

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

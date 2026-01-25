from typing import override

from layer import Layer
from optimizers.optimizer import Optimizer


class GradientDescent(Optimizer):
    def __init__(self, lr):
        super().__init__(lr)

    def step(self, layer: Layer):
        layer.W -= self.lr * layer.dW
        layer.b -= self.lr * layer.db

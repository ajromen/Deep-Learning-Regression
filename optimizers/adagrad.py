import numpy as np

from layer import Layer
from optimizers.optimizer import Optimizer


class Adagrad(Optimizer):
    def __init__(self, lr: float, eps=1e-8):
        super().__init__(lr)
        self.eps = eps
        self.cache = {}

    def step(self, layer: Layer):
        if layer not in self.cache:
            self.cache[layer] = (
                np.zeros_like(layer.W),
                np.zeros_like(layer.b)
            )
        hW, hb = self.cache[layer]
        hW += layer.dW ** 2
        hb += layer.db ** 2
        layer.W -= self.lr * layer.dW / (np.sqrt(hW) + self.eps)
        layer.b -= self.lr * layer.db / (np.sqrt(hb) + self.eps)
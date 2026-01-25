import numpy as np
from layer import Layer
from optimizers.optimizer import Optimizer


class RMSProp(Optimizer):
    def __init__(self, lr: float, rho: float = 0.9, eps: float = 1e-8):
        super().__init__(lr)
        self.rho = rho
        self.eps = eps
        self.cache = {}

    def step(self, layer: Layer):
        if layer not in self.cache:
            self.cache[layer] = (
                np.zeros_like(layer.W),
                np.zeros_like(layer.b)
            )

        hW, hb = self.cache[layer]

        hW[:] = self.rho * hW + (1 - self.rho) * (layer.dW ** 2)
        hb[:] = self.rho * hb + (1 - self.rho) * (layer.db ** 2)

        layer.W -= self.lr * layer.dW / (np.sqrt(hW) + self.eps)
        layer.b -= self.lr * layer.db / (np.sqrt(hb) + self.eps)

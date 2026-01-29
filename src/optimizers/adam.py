import numpy as np
from src.model.model import Layer
from src.optimizers.optimizer import Optimizer


class Adam(Optimizer):
    def __init__(
        self,
        lr: float,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8
    ):
        super().__init__(lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.cache = {}

    def step(self, layer: Layer):
        if layer not in self.cache:
            self.cache[layer] = {
                "mW": np.zeros_like(layer.W),
                "vW": np.zeros_like(layer.W),
                "mb": np.zeros_like(layer.b),
                "vb": np.zeros_like(layer.b),
            }

        self.t += 1
        c = self.cache[layer]

        # momentum
        c["mW"] = self.beta1 * c["mW"] + (1 - self.beta1) * layer.dW
        c["mb"] = self.beta1 * c["mb"] + (1 - self.beta1) * layer.db

        # RMS
        c["vW"] = self.beta2 * c["vW"] + (1 - self.beta2) * (layer.dW ** 2)
        c["vb"] = self.beta2 * c["vb"] + (1 - self.beta2) * (layer.db ** 2)

        mW_hat = c["mW"] / (1 - self.beta1 ** self.t)
        mb_hat = c["mb"] / (1 - self.beta1 ** self.t)
        vW_hat = c["vW"] / (1 - self.beta2 ** self.t)
        vb_hat = c["vb"] / (1 - self.beta2 ** self.t)

        layer.W -= self.lr * mW_hat / (np.sqrt(vW_hat) + self.eps)
        layer.b -= self.lr * mb_hat / (np.sqrt(vb_hat) + self.eps)

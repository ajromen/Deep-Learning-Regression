from src.model.model import Layer
from abc import abstractmethod


class Optimizer:
    def __init__(self, lr):
        self.lr = lr

    @abstractmethod
    def step(self,  layer: Layer):
        raise NotImplementedError("Must be overridden")

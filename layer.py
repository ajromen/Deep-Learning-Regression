import numpy as np

class Layer:
    def __init__(self, input_size: int, output_size: int, activation):
        self.W = np.random.randn(input_size, output_size) * 0.01
        self.b = np.zeros((1, output_size))
        self.activation = activation
        
    def forward(self, X):
        self.X = X
        self.Z = self.W@X+self.b
        self.A = self.activation(self.Z)
        return self.A
    
    def backward(self, dA):
        pass
from encodings import latin_1
import numpy as np
from layer import Layer

# samo sredina bez inputa i outputa oni se dodaju naknadno
layers_types: dict[str, list[int]|None] = {
    "shallow": [100],
    "deep": [30, 30, 30],
    "narrow": [10, 10],
    "default": [50, 50],
    "all": None
}

class Model:
    def __init__(self, data: np.ndarray, train_size: float, input_size: int, output_size: int, learning_rate: float, epochs: int, layers_sizes, optimizer, activation):
        self.data = data
        self.train_size = train_size
        self.learning_rate = learning_rate
        self.input_size = input_size
        self.output_size = output_size
        self.epochs = epochs
        self.optimizer = optimizer
        self.activation = activation
        
        self.layers_sizes = layers_sizes
        self.layers_sizes.append(output_size)
        self.layers_sizes.insert(0,input_size)
        
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
        
        for i in range(len(self.layers_sizes)-1):
            layers.append(
                Layer(self.layers_sizes[i],self.layers_sizes[i+1], self.activation)
            )
        self.layers: list[Layer] = layers
        
    def run(self):
        for _ in range(self.epochs):
            y_hat = self.forward_pass()
            self.backward_pass()
            self.get_loss()
            self.update_params()
    
    def forward_pass(self):
        X = self.x_train
        for layer in self.layers:
            X = layer.forward(X)
        return X
    
    def get_loss(self):
        pass
    
    
    def backward_pass(self):
        pass
        
    
    def update_params(self):
        pass
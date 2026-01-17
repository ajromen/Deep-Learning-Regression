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
    def __init__(self, data: np.ndarray, train_size: float, input_size: int, output_size: int, learning_rate: float, epochs: int, layers_sizes: str, optimizer, activation):
        self.data = data
        self.train_size = train_size
        self.learning_rate = learning_rate
        self.input_size = input_size
        self.output_size = output_size
        self.epochs = epochs
        self.optimizer = optimizer
        self.activation = activation
        
        
        tmp  = layers_types[layers_sizes]
        if tmp is None:
            tmp = []
        self.layers_sizes = tmp
        self.layers_sizes.append(output_size)
        self.layers_sizes.insert(0,input_size)
        
        self._split_train_test()
        self._create_layers()
        
    def _split_train_test(self):
        n = len(self.data)
        split_idx = int(n * self.train_size)
        self.train_data = self.data[:split_idx]
        self.test_data = self.data[split_idx:]
        
    def _create_layers(self):
        layers = []
        
        for i in range(len(self.layers_sizes)-1):
            layers.append(
                Layer(self.layers_sizes[i],self.layers_sizes[i+1], self.activation)
            )
        
    def run(self):
        for _ in range(self.epochs):
            self.forward_pass()
            self.backward_pass()
            self.get_loss()
            self.update_params()
    
    def forward_pass(self):
        pass
    
    def get_loss(self):
        pass
    
    
    def backward_pass(self):
        pass
        
    
    def update_params(self):
        pass
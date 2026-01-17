from ast import mod
from flags import parse_flags
import pandas as pd
import numpy as np
from flags import Options

from model import Model
from activation import activation_functions
from model import layers_types
from optimizer import optimizer_functions
from dataclasses import dataclass

def main():
    options = parse_flags()
    
    
    models = create_models(options)
    
    for model in models:
        model.run()


def create_models(opt: Options)-> list[Model]:
    models = []
    
    df = pd.read_csv(opt.data_path)
    data = df.to_numpy()
    
    number_of_cols = data.shape[1]
    input_size = number_of_cols - opt.output_cols
    output_size = opt.output_cols
    
    activation = activation_functions.get(opt.activation)
    optimizer = optimizer_functions.get(opt.optimizer)
    layers = opt.layers
    
    
    if opt.activation=='all':
        for k, v in activation_functions.items():
            if k=='all':
                continue
            models.append(Model(data, opt.train_size, input_size, output_size, opt.learning_rate, opt.epochs, layers, optimizer, v))
            
    elif opt.optimizer=='all':
        for k,v in optimizer_functions.items():
            if k == 'all':
                continue
            models.append(Model(data, opt.train_size, input_size, output_size, opt.learning_rate, opt.epochs, layers, v, activation))
            
    elif opt.layers == 'all':
        for k,v in layers_types.items():
            if k == 'all':
                continue
            models.append(Model(data, opt.train_size, input_size, output_size, opt.learning_rate, opt.epochs, k, optimizer, activation))
            
    else:
        models.append(Model(data, opt.train_size, input_size, output_size, opt.learning_rate,opt.epochs,layers,optimizer,activation))
    
    return models

if __name__ == "__main__":
    main()
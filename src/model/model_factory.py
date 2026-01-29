import time
from typing import List, Dict

import numpy as np

from src.flags import Flags
from src.model.model import Model
from src.model.options import optimizer_functions, activation_functions,layers_types


def create_models(f: Flags, data, c_names) -> List[Model]:
    models = []

    np.random.shuffle(data)

    number_of_cols = data.shape[1]
    input_size = number_of_cols - f.output_cols
    output_size = f.output_cols

    activation = activation_functions[f.activation]
    optimizer = optimizer_functions[f.optimizer]
    layers = layers_types[f.layers]

    # da bi moglo da se uporedi moraju da se uzmu isti podaci za treniranje
    n = len(data)
    split_idx = int(n * f.train_size)
    train = data[:split_idx]
    test = data[split_idx:]

    if f.activation == 'all':
        for k, v in activation_functions.items():
            if k == 'all':
                continue
            name = f.optimizer + '_' + k + '_' + str(layers)
            models.append(
                Model(train, test, input_size, output_size, f.learning_rate, f.epochs, f.layers, optimizer,
                      v, name=name, batch_size=f.batch_size, c_names=c_names, verbose=f.verbose))

    elif f.optimizer == 'all':
        for k, v in optimizer_functions.items():
            if k == 'all':
                continue
            name = k + '_' + f.activation + '_' + str(layers)
            models.append(
                Model(train, test, input_size, output_size, f.learning_rate, f.epochs, f.layers, v,
                      activation, name=name, batch_size=f.batch_size, c_names=c_names, verbose=f.verbose))

    elif f.layers == 'all':
        for k, v in layers_types.items():
            if k == 'all':
                continue
            name = f.optimizer + '_' + f.activation + '_' + str(v)
            models.append(
                Model(train, test, input_size, output_size, f.learning_rate, f.epochs, k, optimizer,
                      activation, name=name, batch_size=f.batch_size, c_names=c_names, verbose=f.verbose))

    else:
        name = f.optimizer + '_' + f.activation + '_' + str(layers)
        models.append(
            Model(train, test, input_size, output_size, f.learning_rate, f.epochs, f.layers, optimizer,
                  activation, name=name, batch_size=f.batch_size, c_names=c_names, verbose=f.verbose))

    return models

def train_models(models, verbose) -> List[Dict]:
    result = []
    for i, model in enumerate(models):
        if verbose:
            print(model.name + " started")

        t0 = time.time()
        loss_before = model.test()
        model.run()
        loss_after = model.test()
        improvement = loss_before - loss_after
        t1 = time.time()
        training_time = (t1 - t0)

        if verbose:
            print(model.name + " finished, improvement: " + str(improvement) + ", time: " + str(training_time))

        result.append([
            i + 1,
            model.name,
            f"{loss_before:.4f}",
            f"{loss_after:.4f}",
            f"{improvement:.4f}",
            f"{training_time:.2f}s"
        ])
    return result
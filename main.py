import time
from typing import List

import numpy as np
from pandas import DataFrame
from tabulate import tabulate

from flags import parse_flags
import pandas as pd
from flags import Flags

from model import Model
from options import activation_functions, layers_types, optimizer_functions
from visualisation import visualize_models


def main():
    flags = parse_flags()

    df = pd.read_csv(flags.data_path)
    data = df.to_numpy()
    models: List[Model] = create_models(flags, data)

    verbose = flags.verbose

    print(f"Epochs: {flags.epochs}")
    print("Data Preview:")

    print(tabulate(df.head(3), headers='keys', tablefmt='simple_grid'))

    res = []
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

        res.append([
            i + 1,
            model.name,
            f"{loss_before:.4f}",
            f"{loss_after:.4f}",
            f"{improvement:.4f}"
            f"{training_time:.4f}"
        ])

    columns = ["", "Model name", "Loss before", "Loss after", "Improvement", "Training time"]
    print("\nTraining Summary:")
    print(tabulate(res, headers=columns, tablefmt='simple_grid', numalign="right"))

    if not flags.visualise: return

    if flags.activation == "all":
        visualize_models(models, data, mode="activation")
    elif flags.layers == "all":
        visualize_models(models, data, mode="layers")
    elif flags.optimizer == "all":
        visualize_models(models, data, mode="optimizer")
    else:
        visualize_models(models, data)


def create_models(f: Flags, data):
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
                Model(train, test, input_size, output_size, f.learning_rate, f.epochs, f.layers,
                      optimizer,
                      v, name=name, batch_size=f.batch_size))

    elif f.optimizer == 'all':
        for k, v in optimizer_functions.items():
            if k == 'all':
                continue
            name = k + '_' + f.activation + '_' + str(layers)
            models.append(
                Model(train, test, input_size, output_size, f.learning_rate, f.epochs, f.layers, v,
                      activation, name=name, batch_size=f.batch_size))

    elif f.layers == 'all':
        for k, v in layers_types.items():
            if k == 'all':
                continue
            name = f.optimizer + '_' + f.activation + '_' + str(v)
            models.append(
                Model(train, test, input_size, output_size, f.learning_rate, f.epochs, k, optimizer,
                      activation, name=name, batch_size=f.batch_size))

    else:
        name = f.optimizer + '_' + f.activation + '_' + str(layers)
        models.append(
            Model(train, test, input_size, output_size, f.learning_rate, f.epochs, f.layers, optimizer,
                  activation, name=name, batch_size=f.batch_size))

    return models


if __name__ == "__main__":
    main()

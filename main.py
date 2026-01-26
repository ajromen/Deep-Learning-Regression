from typing import List

from pandas import DataFrame
from tabulate import tabulate

from flags import parse_flags
import pandas as pd
from flags import Options

from model import Model
from options import activation_functions, layers_types, optimizer_functions
from visualisation import visualize_models


def main():
    options = parse_flags()

    df = pd.read_csv(options.data_path)
    models:List[Model] = create_models(options,df)

    print(f"Epochs: {options.epochs}")
    print("Data Preview:")

    print(tabulate(df.head(3), headers='keys', tablefmt='simple_grid'))

    res = []
    for i, model in enumerate(models):
        loss_before = model.test()
        model.run()
        loss_after = model.test()
        improvement = loss_before - loss_after

        res.append([
            i + 1,
            model.name,
            f"{loss_before:.4f}",
            f"{loss_after:.4f}",
            f"{improvement:.4f}"
        ])

    columns = ["", "Model name", "Loss before", "Loss after", "Improvement"]
    print("\nTraining Summary:")
    print(tabulate(res, headers=columns, tablefmt='simple_grid', numalign="right"))

    if options.activation == "all":
        visualize_models(models, mode="activation")
    elif options.layers == "all":
        visualize_models(models, mode="layers")
    elif options.optimizer == "all":
        visualize_models(models, mode="optimizer")
    else:
        visualize_models(models)


def create_models(opt: Options, df:DataFrame):
    models = []

    data = df.to_numpy()

    number_of_cols = data.shape[1]
    input_size = number_of_cols - opt.output_cols
    output_size = opt.output_cols

    activation = activation_functions[opt.activation]
    optimizer = optimizer_functions[opt.optimizer]
    layers = layers_types[opt.layers]

    if opt.activation == 'all':
        for k, v in activation_functions.items():
            if k == 'all':
                continue
            name = opt.optimizer + '_' + k + '_' + str(layers)
            models.append(
                Model(data, opt.train_size, input_size, output_size, opt.learning_rate, opt.epochs, opt.layers,
                      optimizer,
                      v, name))

    elif opt.optimizer == 'all':
        for k, v in optimizer_functions.items():
            if k == 'all':
                continue
            name = k + '_' + opt.activation + '_' + str(layers)
            models.append(
                Model(data, opt.train_size, input_size, output_size, opt.learning_rate, opt.epochs, opt.layers, v,
                      activation, name))

    elif opt.layers == 'all':
        for k, v in layers_types.items():
            if k == 'all':
                continue
            name = opt.optimizer + '_' + opt.activation + '_' + str(v)
            models.append(
                Model(data, opt.train_size, input_size, output_size, opt.learning_rate, opt.epochs, k, optimizer,
                      activation, name))

    else:
        name = opt.optimizer + '_' + opt.activation + '_' + str(layers)
        models.append(
            Model(data, opt.train_size, input_size, output_size, opt.learning_rate, opt.epochs, opt.layers, optimizer,
                  activation, name))

    return models


if __name__ == "__main__":
    main()

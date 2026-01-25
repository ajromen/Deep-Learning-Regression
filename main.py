from typing import List

from flags import parse_flags
import pandas as pd
from flags import Options

from model import Model
from options import activation_functions, layers_types, optimizer_functions


def main():
    options = parse_flags()

    models: List[Model] = create_models(options)

    for i, model in enumerate(models):
        print(str(i + 1) + ". "+model.name)
        print("\tLoss before training: " + str(model.test()))
        model.run()
        print("\tLoss after training: " + str(model.test()))


def create_models(opt: Options) -> list[Model]:
    models = []

    df = pd.read_csv(opt.data_path)
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
                Model(data, opt.train_size, input_size, output_size, opt.learning_rate, opt.epochs, layers, optimizer,
                      v, name))

    elif opt.optimizer == 'all':
        for k, v in optimizer_functions.items():
            if k == 'all':
                continue
            name = k + '_' + opt.activation + '_' + str(layers)
            models.append(Model(data, opt.train_size, input_size, output_size, opt.learning_rate, opt.epochs, layers, v,
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
            Model(data, opt.train_size, input_size, output_size, opt.learning_rate, opt.epochs, layers, optimizer,
                  activation, name))

    return models


if __name__ == "__main__":
    main()

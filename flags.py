import argparse
import os.path
from options import activation_functions, layers_types, optimizer_functions
from dataclasses import dataclass

@dataclass
class Options:
    activation: str
    optimizer: str
    data_path: str
    epochs: int
    layers: str
    train_size: float
    output_cols: int
    learning_rate: float


def data_type(value):
    value = value.strip()
    if not os.path.exists(value):
        raise argparse.ArgumentTypeError("Path does not exist")
    return value


def parse_flags() -> Options:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-a", "--activation",
        choices=activation_functions.keys(),
        default="relu",
        help="Activation function used in hidden layers (relu - default)"
    )

    parser.add_argument(
        "-o", "--optimizer",
        default="gradient",
        choices=optimizer_functions.keys(),
        help="Optimization algorithm used for training (gradient - default)"
    )

    parser.add_argument(
        "-d", "--data",
        type=data_type,
        metavar="PATH",
        default="./data/test.csv",
        help="Path to input data"
    )

    parser.add_argument(
        "-e", "--epochs",
        default=1000,
        metavar="N",
        type=int,
        help="Number of training epochs (full passes over the dataset)"
    )

    parser.add_argument(
        '-l', "--layers",
        default="default",
        choices=layers_types.keys(),
        help="Choose from predetermined layers (default - in, 50, 50, out)"
    )
    
    parser.add_argument(
        '--train-size',
        type=float,
        default=0.8,
        help="Fraction of data used for training (0.8 = 80%%)"
    )

    parser.add_argument(
        '--output-cols',
        default=1,
        type=int,
        metavar='N',
        help="Number of columns from input data that are used as outputs"
    )

    parser.add_argument(
        '--lr',
        default=1e-3,
        type=float,
        metavar="FLOAT",
        help="Learning rate"
    )

    args = parser.parse_args()

    # check if multiple flags are set to 'all'
    num = 0
    if args.layers == 'all': num += 1
    if args.optimizer == 'all': num += 1
    if args.activation == 'all': num += 1

    if num > 1:
        raise argparse.ArgumentTypeError(
            "Only one of 'optimizer','activation' and 'layers' flags can be set to all at a time")

    opt = Options(args.activation, args.optimizer,args.data,args.epochs, args.layers, args.train_size, args.output_cols, args.lr)

    return opt

import time
from typing import List

from tabulate import tabulate
from src.flags import parse_flags
import pandas as pd
from src.model.model import Model
from src.model import model_factory
from src.visualisation import visualise

def main():
    flags = parse_flags()

    df, data = read_input_data(flags.data_path)

    columns = df.columns
    models: List[Model] = model_factory.create_models(flags, data, columns)

    verbose = flags.verbose

    result = model_factory.train_models(models, verbose)
    print_summary(result, flags.epochs, df)

    visualise(models, flags, data)

def read_input_data(path: str):
    df = pd.read_csv(path)
    data = df.to_numpy()
    return df, data

def print_summary(res, epochs, df):
    print(f"Epochs: {epochs}")
    print("Data Preview:")
    print(tabulate(df.head(3), headers='keys', tablefmt='simple_grid'))

    columns = ["", "Model name", "Loss before", "Loss after", "Improvement", "Training time"]
    print("\nTraining Summary:")
    print(tabulate(res, headers=columns, tablefmt='simple_grid', numalign="right", stralign="right"))

if __name__ == "__main__":
    main()

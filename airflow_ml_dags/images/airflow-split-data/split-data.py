import os
import click

import pandas as pd
from sklearn.model_selection import train_test_split


@click.command()
@click.option('--input-dir')
@click.option('--output-dir')
def split_data(input_dir: str, output_dir: str):
    X = pd.read_csv(os.path.join(input_dir, 'data.csv'))
    y = pd.read_csv(os.path.join(input_dir, 'target.csv'))

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33)

    os.makedirs(output_dir, exist_ok=True)

    X_train.to_csv(os.path.join(output_dir, 'X_train.csv'), index=False)
    X_val.to_csv(os.path.join(output_dir, 'X_val.csv'), index=False)
    y_train.to_csv(os.path.join(output_dir, 'y_train.csv'), index=False)
    y_val.to_csv(os.path.join(output_dir, 'y_val.csv'), index=False)


if __name__ == '__main__':
    split_data()
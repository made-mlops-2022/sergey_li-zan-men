import os
import pickle

import click
import pandas as pd
from sklearn.linear_model import LogisticRegression

import mlflow

@click.command('train')
@click.option('--input-dir')
@click.option('--output-dir')
def train(input_dir: str, output_dir: str):
    X = pd.read_csv(os.path.join(input_dir, 'X_train_processed.csv'))
    y = pd.read_csv(os.path.join(input_dir, 'y_train_processed.csv'))

    model = LogisticRegression()
    model.fit(X, y)

    # model_params = model.get_params()
    # mlflow.set_tracking_uri("http://localhost:4999")
    # with mlflow.start_run(run_name='train'):
    #     for param in model_params:
    #         mlflow.log_param(param, model_params[param])

    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, 'model.pkl'), "wb") as f:
        pickle.dump(model, f)


if __name__ == '__main__':
    train()

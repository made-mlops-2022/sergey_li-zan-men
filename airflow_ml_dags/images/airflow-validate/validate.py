import json
import os
import pickle

import click
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import mlflow


@click.command()
@click.option('--input-dir')
@click.option('--output-dir')
@click.option('--model-dir')
def validate(input_dir: str, output_dir: str, model_dir: str):
    with open(os.path.join(model_dir, 'model.pkl'), 'rb') as f:
        model = pickle.load(f)

    X = pd.read_csv(os.path.join(input_dir, 'X_val_processed.csv'))
    target = pd.read_csv(os.path.join(input_dir, 'y_val_processed.csv'))
    predicts = model.predict(X)

    metrics = dict()
    metrics['accuracy_score'] = accuracy_score(target, predicts)
    metrics['precision_score'] = precision_score(target, predicts)
    metrics['recall_score'] = recall_score(target, predicts)
    metrics['f1_score'] = f1_score(target, predicts)

    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, 'metrics.txt'), 'w') as f:
        json.dump(metrics, f)

    # mlflow.set_tracking_uri("http://localhost:4999")
    # run_name = input_dir.split('/')[-1]
    # with mlflow.start_run(run_name=run_name):
    #     for metric, score in metrics.items():
    #         mlflow.log_metric(metric, score)


if __name__ == '__main__':
    validate()

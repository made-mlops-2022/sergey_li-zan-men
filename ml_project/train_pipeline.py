import json

import click
import pandas as pd

from src.data import read_data, split_train_val_data
from src.features import extract_target, build_transformer
from src.models import evaluate_model, train_model, predict_model, dump_model
from src.enities import read_train_pipeline_params, TrainingPipelineParams


def train_pipeline(
        train_pipeline_params: TrainingPipelineParams
) -> tuple[str, dict[str, float]]:
    data = read_data(train_pipeline_params.path_to_data)

    train_df, val_df = split_train_val_data(
        data, train_pipeline_params.splitting_params
    )
    train_df, train_target = extract_target(train_df, train_pipeline_params.feature_params)
    val_df, val_target = extract_target(val_df, train_pipeline_params.feature_params)

    transformer = build_transformer(train_pipeline_params.feature_params)
    train_features = transformer.fit_transform(train_df)

    model = train_model(train_features, train_target, train_pipeline_params.train_params)

    val_features = transformer.transform(val_df)
    predicts = predict_model(val_features, model)

    metrics = evaluate_model(predicts, val_target)

    with open(train_pipeline_params.path_to_metrics, "w") as f:
        json.dump(metrics, f)

    path_to_model = dump_model(model, train_pipeline_params.path_to_output)
    return path_to_model, metrics


@click.command()
@click.argument("config_path", default='configs/train_log_reg_with_scaler.yaml',
                type=click.Path(exists=True))
def train_pipeline_command(config_path: str):
    params = read_train_pipeline_params(config_path)
    train_pipeline(params)


if __name__ == '__main__':
    train_pipeline_command()

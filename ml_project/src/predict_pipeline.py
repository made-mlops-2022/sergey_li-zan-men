import click
import pandas as pd

from src.data import read_data
from src.features import build_transformer
from src.models import load_model, predict_model
from src.enities import read_predict_pipeline_params, PredictPipelineParams


def predict_pipeline(
        predict_pipeline_params: PredictPipelineParams
) -> str:
    data = read_data(predict_pipeline_params.path_to_data)

    transformer = build_transformer(predict_pipeline_params.feature_params)

    features = transformer.fit_transform(data)

    model = load_model(predict_pipeline_params.path_to_model)

    predicts = predict_model(features, model)

    pd.Series(predicts).to_csv(predict_pipeline_params.path_to_output, index=False)

    return predict_pipeline_params.path_to_output


@click.command()
@click.argument("config_path", default='configs/predict_config.yaml',
                type=click.Path(exists=True))
def predict_pipeline_command(config_path: str):
    params = read_predict_pipeline_params(config_path)
    predict_pipeline(params)


if __name__ == '__main__':
    predict_pipeline_command()

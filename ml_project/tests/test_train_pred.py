import os

import pandas as pd

from generate_synthetic_data import generate_synthetic_data
from src.enities import (
    TrainingPipelineParams,
    SplittingParams,
    FeatureParams,
    TrainingParams,
    PredictPipelineParams
)
from src.train_pipeline import train_pipeline
from src.predict_pipeline import predict_pipeline

cat_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
num_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
generate_synthetic_data()


def test_train_pipeline():
    path_to_data = os.path.join(
        'tests', 'test_data', 'test_train.csv'
    )
    path_to_output = os.path.join(
        'tests', 'test_results', 'test_model.pkl'
    )
    path_to_metrics = os.path.join(
        'tests', 'test_results', 'test_val_metrics.json'
    )
    splitting_params = SplittingParams(0.33, 42)
    feature_params = FeatureParams(cat_cols, num_cols, 'condition', True)
    train_params = TrainingParams('LogisticRegression')

    train_pipeline_params = TrainingPipelineParams(
        path_to_data=path_to_data,
        path_to_output=path_to_output,
        path_to_metrics=path_to_metrics,
        splitting_params=splitting_params,
        feature_params=feature_params,
        train_params=train_params
    )

    assert not os.path.exists(path_to_output), f'{path_to_output} exists'
    assert not os.path.exists(path_to_metrics), f'{path_to_metrics} exists'

    res_path, metrics = train_pipeline(train_pipeline_params)
    assert os.path.exists(path_to_output), f'{path_to_output} not exists'
    assert os.path.exists(path_to_metrics), f'{path_to_metrics} not exists'

    for metric, score in metrics.items():
        assert 0 <= score <= 1, f'{metric} is not a good value'


def test_predict_pipeline():
    path_to_data = os.path.join('tests', 'test_data', 'test_predict.csv')
    path_to_output = os.path.join('tests', 'test_results', 'test_res_predict.csv')
    path_to_model = os.path.join('tests', 'test_results', 'test_model.pkl')

    feature_params = FeatureParams(cat_cols, num_cols, use_scaler=True)

    train_pipeline_params = PredictPipelineParams(
        path_to_data=path_to_data,
        path_to_output=path_to_output,
        path_to_model=path_to_model,
        feature_params=feature_params
    )

    assert not os.path.exists(path_to_output), f'{path_to_output} exists'

    res_path = predict_pipeline(train_pipeline_params)
    assert os.path.exists(path_to_output), f'{path_to_output} not exists'

    predicts = pd.read_csv(res_path).iloc[0]
    assert 0 <= predicts.sum() <= predicts.shape[0], 'predicts is not a good value'

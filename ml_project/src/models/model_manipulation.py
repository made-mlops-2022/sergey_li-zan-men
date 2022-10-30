import numpy as np
import pandas as pd
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score
)

from ..enities import TrainingParams

ClassifierModel = RandomForestClassifier | LogisticRegressionCV


def train_model(
        features: pd.DataFrame,
        target: pd.Series,
        train_params: TrainingParams
) -> ClassifierModel:
    if train_params.model_type == 'LogisticRegression':
        model = LogisticRegressionCV(
            Cs=10, penalty='l2', random_state=train_params.random_state
        )
    elif train_params.model_type == 'RandomForestClassifier':
        model = RandomForestClassifier(
            criterion='entropy', max_depth=3,
            n_estimators=100,
            random_state=train_params.random_state
        )
    else:
        raise NotImplementedError('Model type not implemented')

    model.fit(features, target)
    return model


def predict_model(
        features: pd.DataFrame,
        model: ClassifierModel,
) -> np.ndarray:
    return model.predict(features)


def evaluate_model(
        predicts: np.ndarray, target: pd.Series
) -> dict[str, float]:
    metrics = dict()
    metrics['accuracy_score'] = accuracy_score(target, predicts)
    metrics['precision_score'] = precision_score(target, predicts)
    metrics['recall_score'] = recall_score(target, predicts)
    metrics['f1_score'] = f1_score(target, predicts)
    return metrics


def dump_model(model: ClassifierModel, path_to_output: str) -> str:
    with open(path_to_output, "wb") as f:
        pickle.dump(model, f)
    return path_to_output


def load_model(path_to_model: str) -> ClassifierModel:
    with open(path_to_model, 'rb') as f:
        model = pickle.load(f)
    return model

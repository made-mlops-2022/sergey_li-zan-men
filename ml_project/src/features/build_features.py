import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from src.enities import FeatureParams


def extract_target(
        df: pd.DataFrame, feature_params: FeatureParams
) -> tuple[pd.DataFrame, pd.Series]:
    target = df[feature_params.target_col]
    df_without_target = df.drop(feature_params.target_col, axis=1)
    return df_without_target, target


def __build_categorical_pipeline(feature_params: FeatureParams) -> Pipeline:
    cat_pipeline = Pipeline([
        ("impute", SimpleImputer(missing_values=np.nan, strategy="most_frequent")),
        ('ohe', OneHotEncoder(drop='first'))
    ])
    return cat_pipeline


def __build_numerical_pipeline(feature_params: FeatureParams) -> Pipeline:
    num_pipeline = Pipeline([
        ("impute", SimpleImputer(missing_values=np.nan, strategy="mean"))
    ])

    if feature_params.use_scaler:
        num_pipeline.steps.append(('scaling', StandardScaler()))
    return num_pipeline


def build_transformer(feature_params: FeatureParams) -> ColumnTransformer:
    transformer = ColumnTransformer(
        [
            (
                'numerical_preprocess',
                __build_numerical_pipeline(feature_params),
                feature_params.numerical_features
            ),
            (
                'categorical_preprocess',
                __build_categorical_pipeline(feature_params),
                feature_params.categorical_features
            )
        ]
    )
    return transformer

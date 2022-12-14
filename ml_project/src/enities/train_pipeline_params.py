from dataclasses import dataclass
import yaml
from marshmallow_dataclass import class_schema

from .split_params import SplittingParams
from .feature_params import FeatureParams
from .train_params import TrainingParams


@dataclass()
class TrainingPipelineParams:
    path_to_data: str
    path_to_output: str
    path_to_metrics: str
    splitting_params: SplittingParams
    feature_params: FeatureParams
    train_params: TrainingParams


TrainingPipelineParamsSchema = class_schema(TrainingPipelineParams)


def read_train_pipeline_params(config_path: str) -> TrainingPipelineParams:
    with open(config_path, "r") as config:
        schema = TrainingPipelineParamsSchema()
        return schema.load(yaml.safe_load(config))

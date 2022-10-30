from dataclasses import dataclass, field
from typing import Optional


@dataclass()
class FeatureParams:
    categorical_features: list[str]
    numerical_features: list[str]
    features_to_drop: list[str]
    target_col: Optional[str]
    use_scaler: bool = field(default=False)

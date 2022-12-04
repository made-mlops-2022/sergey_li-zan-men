from dataclasses import dataclass, field
from typing import Optional
from typing import List


@dataclass()
class FeatureParams:
    categorical_features: List[str]
    numerical_features: List[str]
    target_col: Optional[str] = field(default=None)
    use_scaler: bool = field(default=False)

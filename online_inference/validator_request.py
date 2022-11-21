from typing import Literal, Union

from pydantic import BaseModel, validator
from fastapi import HTTPException


class RequestData(BaseModel):
    age: int
    sex: Literal[0, 1]
    cp: Literal[0, 1, 2, 3]
    trestbps: int
    chol: int
    fbs: Literal[0, 1]
    restecg: Literal[0, 1, 2]
    thalach: int
    exang: Literal[0, 1]
    oldpeak: float
    slope: Literal[0, 1, 2]
    ca: Literal[0, 1, 2, 3]
    thal: Literal[0, 1, 2]

    @staticmethod
    def check_between_borders(
            name: str,
            l_border: Union[int, float],
            r_border: Union[int, float],
            value: Union[int, float]
    ) -> Union[int, float]:
        if value < l_border or value > r_border:
            raise HTTPException(
                status_code=400,
                detail=f'ERROR: {name} cannot be more than {r_border} and less than {l_border}'
            )
        return value

    @validator('age')
    def check_age(cls, v):
        print(v)
        return cls.check_between_borders('age', 0, 100, v)

    @validator('trestbps')
    def check_trestbps(cls, v):
        return cls.check_between_borders('trestbps', 93, 201, v)

    @validator('chol')
    def check_chol(cls, v):
        return cls.check_between_borders('chol', 125, 565, v)

    @validator('thalach')
    def check_thalach(cls, v):
        return cls.check_between_borders('thalach', 70, 203, v)

    @validator('oldpeak')
    def check_oldpeak(cls, v):
        return cls.check_between_borders('oldpeak', 0, 6.2, v)

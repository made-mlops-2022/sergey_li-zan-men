import json
import os
import pickle
from typing import Optional

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer

from fastapi import FastAPI
from fastapi_health import health

from validator_request import RequestData

app = FastAPI()

model: Optional[LogisticRegression] = None
transformer: Optional[ColumnTransformer] = None


def load_model():
    global model
    path_to_model = os.getenv('PATH_TO_MODEL')
    with open(path_to_model, 'rb') as f:
        model = pickle.load(f)


def load_transformer():
    global transformer
    path_to_transformer = os.getenv('PATH_TO_TRANSFORMER')
    with open(path_to_transformer, 'rb') as f:
        transformer = pickle.load(f)


@app.on_event('startup')
async def startup():
    load_model()
    load_transformer()


@app.post('/predict')
async def get_predict(data: RequestData):
    data = pd.DataFrame([data.dict()])
    x_test = transformer.transform(data)
    prediction = model.predict(x_test)
    return {'prediction': str(prediction)}


@app.get('/')
async def root():
    return 'Hello!'


def check_model():
    return all(obj is not None for obj in (model, transformer))


app.add_api_route("/health", health([check_model]))

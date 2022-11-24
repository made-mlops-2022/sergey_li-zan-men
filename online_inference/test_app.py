import json

from fastapi.testclient import TestClient
from main import app, load_transformer, load_model

client = TestClient(app)
load_model()
load_transformer()

request = {
    'age': 32,
    'sex': 0,
    'cp': 3,
    'trestbps': 105,
    'chol': 158,
    'fbs': 1,
    'restecg': 2,
    'thalach': 75,
    'exang': 1,
    'oldpeak': 5.23,
    'slope': 1,
    'ca': 2,
    'thal': 0,
}


def test_good_request():
    response = client.post(
        '/predict', data=json.dumps(request)
    )
    assert response.status_code == 200
    ans = json.loads(response.text)['prediction']
    assert ans == '[1]' or ans == '[0]'


def test_wrong_age():
    step = request.copy()
    step['age'] = 140
    response = client.post(
        '/predict', data=json.dumps(step)
    )
    assert response.status_code == 400
    assert response.json()['detail'] == 'ERROR: age cannot be more than 100 and less than 0'


def test_wrong_sex():
    step = request.copy()
    step['sex'] = 4

    response = client.post(
        '/predict', data=json.dumps(step)
    )
    assert response.status_code == 422
    assert response.json()['detail'][0]['msg'] == \
           'unexpected value; permitted: 0, 1'


def test_wrong_oldpeak():
    step = request.copy()
    step['oldpeak'] = 6.3

    response = client.post(
        '/predict', data=json.dumps(step)
    )
    assert response.status_code == 400
    assert response.json()['detail'] == 'ERROR: oldpeak cannot be more than 6.2 and less than 0'


def test_miss_field():
    step = request.copy()
    del step['sex']

    response = client.post(
        '/predict', data=json.dumps(step)
    )
    assert response.status_code == 422
    assert response.json()['detail'][0]['msg'] == \
           'field required'

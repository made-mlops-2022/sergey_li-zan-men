import json
import requests
import pandas as pd

x_test = pd.read_csv('synthetic_test_predict.csv').to_dict(orient="records")

for row in x_test:

    print(row)

    response = requests.post(
        'http://127.0.0.1:8000/predict',
        json.dumps(row)
    )
    print(response.status_code)
    print(json.loads(response.text)['prediction'])

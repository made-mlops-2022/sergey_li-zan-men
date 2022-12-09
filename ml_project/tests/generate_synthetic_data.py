import pandas as pd
import os
from numpy.random import randint, choice, uniform, seed


def get_synthetic_data(n_rows: int, seed_val: int, need_target: bool = True) -> pd.DataFrame:
    seed(seed_val)
    data = {
        'age': randint(29, 78, size=n_rows),
        'sex': choice(2, size=n_rows),
        'cp': choice(4, size=n_rows),
        'trestbps': randint(94, 201, size=n_rows),
        'chol': randint(126, 565, size=n_rows),
        'fbs': choice(2, size=n_rows),
        'restecg': choice(3, size=n_rows),
        'thalach': randint(71, 203, size=n_rows),
        'exang': choice(2, size=n_rows),
        'oldpeak': uniform(0, 6.2, size=n_rows),
        'slope': choice(3, size=n_rows),
        'ca': choice(4, size=n_rows),
        'thal': choice(3, size=n_rows),
    }
    if need_target:
        data['condition'] = choice(2, size=n_rows)
    df = pd.DataFrame(data)
    return df


def generate_synthetic_data():
    test_train_df = get_synthetic_data(300, 42)
    test_train_df.to_csv(
        os.path.join('tests', 'test_data', 'test_train.csv'),
        index=False
    )

    test_predict_df = get_synthetic_data(50, 42, False)
    test_predict_df.to_csv(
        os.path.join('tests', 'test_data', 'test_predict.csv'),
        index=False
    )

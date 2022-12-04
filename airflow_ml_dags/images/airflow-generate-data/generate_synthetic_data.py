import os
import pandas as pd
from numpy.random import randint, choice, uniform
import click

TARGET_COL = 'condition'

DATA_FILENAME = 'data.csv'
TARGET_FILENAME = 'target.csv'


def get_synthetic_data(n_rows: int) -> pd.DataFrame:
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
        'condition': choice(2, size=n_rows)
    }
    df = pd.DataFrame(data)
    return df


@click.command()
@click.option("--output-dir")
def generate_synthetic_data(output_dir: str):
    df = get_synthetic_data(1200)

    target = df[TARGET_COL]
    df.drop(TARGET_COL, axis=1, inplace=True)

    os.makedirs(output_dir, exist_ok=True)

    df.to_csv(os.path.join(output_dir, DATA_FILENAME), index=False)

    target.to_csv(os.path.join(output_dir, TARGET_FILENAME), index=False)


if __name__ == '__main__':
    generate_synthetic_data()

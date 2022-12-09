import os
import pandas as pd
import click
import shutil
import pickle

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

CAT_FEATURES = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
NUM_FEATURES = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']


@click.command()
@click.option('--input-dir')
@click.option('--output-dir')
@click.option('--transformer-output-dir')
def preprocess(input_dir: str, output_dir: str, transformer_output_dir):
    X_train = pd.read_csv(os.path.join(input_dir, 'X_train.csv'))
    X_val = pd.read_csv(os.path.join(input_dir, 'X_val.csv'))

    transformer = ColumnTransformer([
        ('cat_preprocess', OneHotEncoder(drop='first'), CAT_FEATURES),
        ('num_preprocess', StandardScaler(), NUM_FEATURES)
    ])

    X_train_processed = transformer.fit_transform(X_train)
    X_val_processed = transformer.transform(X_val)

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(transformer_output_dir, exist_ok=True)

    pd.DataFrame(X_train_processed).to_csv(os.path.join(output_dir, 'X_train_processed.csv'), index=False)
    pd.DataFrame(X_val_processed).to_csv(os.path.join(output_dir, 'X_val_processed.csv'), index=False)
    shutil.copyfile(os.path.join(input_dir, 'y_train.csv'), os.path.join(output_dir, 'y_train_processed.csv'))
    shutil.copyfile(os.path.join(input_dir, 'y_val.csv'), os.path.join(output_dir, 'y_val_processed.csv'))

    with open(os.path.join(transformer_output_dir, 'transformer.pkl'), 'wb') as f:
        pickle.dump(transformer, f)


if __name__ == '__main__':
    preprocess()

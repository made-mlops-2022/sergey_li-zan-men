import os
from datetime import timedelta

from airflow.models import Variable


def check_file(path_to_file: str):
    return os.path.exists(path_to_file)


DATA_DIR = Variable.get('data_dir')

default_args = {
    "owner": "lzmsergey",
    "email": ["lizanmensergej@gmail.com"],
    "retries": 3,
    "retry_delay": timedelta(minutes=5),
}

from datetime import datetime

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount

from utils import default_args, DATA_DIR

with DAG(
        'predict',
        default_args=default_args,
        schedule_interval='@daily',
        start_date=datetime(2022, 11, 10)
) as dag:
    predictor = DockerOperator(
        image='airflow-predict',
        command='--input-dir /data/raw/{{ ds }} '
                '--output-dir /data/predictions/{{ ds }} '
                '--model-dir /data/models/{{ var.value.get("data_for_prod") }}',
        task_id='docker-airflow-predict',
        auto_remove=True,
        mounts=[Mount(source=DATA_DIR, target='/data', type='bind')]
    )

    predictor


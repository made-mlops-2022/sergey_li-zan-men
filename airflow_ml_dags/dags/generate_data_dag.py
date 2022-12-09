from datetime import datetime

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount

from utils import default_args, DATA_DIR

with DAG(
        'generate_data',
        default_args=default_args,
        schedule_interval='@daily',
        start_date=datetime(2022, 11, 10)
) as dag:
    generator = DockerOperator(
        image='airflow-generate-data',
        command='--output-dir /data/raw/{{ ds }}',
        task_id='docker-airflow-generate-data',
        auto_remove=True,
        mounts=[Mount(source=DATA_DIR, target='/data', type='bind')]
    )

    generator


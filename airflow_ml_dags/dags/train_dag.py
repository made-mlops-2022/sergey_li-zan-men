from datetime import datetime


from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.sensors.python import PythonSensor

from docker.types import Mount

from utils import default_args, check_file, DATA_DIR

with DAG(
        'train_model',
        default_args=default_args,
        schedule_interval='@weekly',
        start_date=datetime(2022, 11, 10)
) as dag:
    check_data = PythonSensor(
        task_id='docker-check-data',
        python_callable=check_file,
        op_args=['/opt/airflow/data/raw/{{ ds }}/data.csv'],
        timeout=3600,
        poke_interval=5
    )

    check_target = PythonSensor(
        task_id='docker-check-target',
        python_callable=check_file,
        op_args=['/opt/airflow/data/raw/{{ ds }}/target.csv'],
        timeout=3600,
        poke_interval=5
    )

    splitter = DockerOperator(
        image='airflow-split-data',
        command='--input-dir /data/raw/{{ ds }} --output-dir /data/splitted/{{ ds }}',
        task_id='docker-airflow-split-data',
        auto_remove=True,
        mounts=[Mount(source=DATA_DIR, target='/data', type='bind')]
    )

    preprocessor = DockerOperator(
        image='airflow-preprocess',
        command='--input-dir /data/splitted/{{ ds }} '
                '--output-dir /data/processed/{{ ds }} '
                '--transformer-output-dir /data/models/{{ ds }}',
        task_id='docker-airflow-preprocess',
        auto_remove=True,
        mounts=[Mount(source=DATA_DIR, target='/data', type='bind')]
    )

    trainer = DockerOperator(
        image='airflow-train',
        command='--input-dir /data/processed/{{ ds }} --output-dir /data/models/{{ ds }}',
        task_id='docker-airflow-train',
        auto_remove=True,
        mounts=[Mount(source=DATA_DIR, target='/data', type='bind')]
    )

    validator = DockerOperator(
        image='airflow-validate',
        command='--input-dir /data/processed/{{ ds }} '
                '--output-dir /data/val_metrics/{{ ds }} '
                '--model-dir /data/models/{{ ds }}',
        task_id='docker-airflow-validate',
        auto_remove=True,
        mounts=[Mount(source=DATA_DIR, target='/data', type='bind')]
    )

    [check_data, check_target] >> splitter >> preprocessor >> trainer >> validator

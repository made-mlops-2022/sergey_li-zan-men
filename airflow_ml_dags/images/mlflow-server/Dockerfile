FROM python:3.8.15-slim-buster

COPY requirements.txt /

RUN python3 -m pip install --upgrade pip && pip install -r /requirements.txt

EXPOSE 4999

CMD mlflow server --host 0.0.0.0 --port 4999

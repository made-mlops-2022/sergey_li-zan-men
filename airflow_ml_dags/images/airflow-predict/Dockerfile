FROM python:3.8.15-slim-buster
COPY requirements.txt /
RUN python3 -m pip install --upgrade pip && pip install -r /requirements.txt

COPY predict.py /predict.py

ENTRYPOINT ["python", "predict.py"]
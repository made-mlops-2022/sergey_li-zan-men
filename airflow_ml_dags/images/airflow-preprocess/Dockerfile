FROM python:3.8.15-slim-buster
COPY requirements.txt /
RUN python3 -m pip install --upgrade pip && pip install -r /requirements.txt

COPY preprocess.py /preprocess.py

ENTRYPOINT ["python", "preprocess.py"]
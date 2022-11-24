FROM python:3.9.15-slim

WORKDIR ./online_inference

COPY requirements.txt main.py validator_request.py test_app.py model.pkl transformer.pkl ./
RUN python3 -m pip install --upgrade pip && pip3 install -r requirements.txt

ENV PATH_TO_MODEL="model.pkl"
ENV PATH_TO_TRANSFORMER="transformer.pkl"

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
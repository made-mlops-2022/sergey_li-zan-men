# MLops MADE Homework 1 
### Li-Zan-Men Sergey

Used dataset [Heart Disease Cleveland UCI](https://www.kaggle.com/datasets/cherngs/heart-disease-cleveland-uci)

### Usage
1. Install dependencies:
```
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```
2. Get EDA report with **dataprep**:
```
python src/generate_eda_report.py
```
3. Run first way training model (Logistic Regression with scaling):
```
python src/train_pipeline.py
```
OR
```
python src/train_pipeline.py configs/train_log_reg_with_scaler.yaml
```
4. Run second way training model (Random Forest without scaling):
```
python src/train_pipeline.py configs/train_random_forest_without_scaler.yaml
```
5. Run predict:
```
python src/predict_pipeline.py
```
OR
```
python src/predict_pipeline.py configs/predict_config.yaml
```
6. Run tests:
```
pytest tests/
```

# MLops MADE Homework 2 
### Li-Zan-Men Sergey

Used dataset [Heart Disease Cleveland UCI](https://www.kaggle.com/datasets/cherngs/heart-disease-cleveland-uci)

### Usage
1. Build docker image:
```
docker build -t lizanmensergej/online_model:3 .
```
OR pull docker image
```
docker pull lizanmensergej/online_model:3
```
2. Run app
```
docker run --name online_model --rm -d -p 8000:8000 lizanmensergej/online_model:3
```
3. Run requests:
```
python get_requests.py
```
4. Run tests:
```
docker exec -it online_model bash
pytest test_app.py
```

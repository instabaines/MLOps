import requests

url='http://127.0.0.1:9696/predict'
ride={
    'year':2021,
    'month':'04'
}
response=requests.post(url,json=ride)
print(response.json())
import pandas as pd
import requests
service_name = 'sales'
host = f'{service_name}.default.example.com'
actual_domain = 'http://localhost:8081'
url =f'{actual_domain}/v1/models/{service_name}:predict'

headers ={'Host':host}

df=pd.read_csv('../data/raw/test.csv')
df = df.merge(pd.read_csv('../data/raw/store.csv'),on='store')
   
request = {
    "instances":df
}

response = requests.post(url,json=request,headers=headers)
print(response.json())
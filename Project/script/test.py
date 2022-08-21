import pandas as pd
import requests
import numpy as np
service_name = 'sales'
host = f'{service_name}.default.example.com'
actual_domain = 'http://localhost:8081'
url =f'{actual_domain}/v1/models/{service_name}:predict'

headers ={'Host':host}


   
request = {
    "instances":[
        {'Id': 1,
 'Store': 1,
 'DayOfWeek': 4,
 'Date': '2015-09-17',
 'Open': 1.0,
 'Promo': 1,
 'StateHoliday': '0',
 'SchoolHoliday': 0,
 'StoreType': 'c',
 'Assortment': 'a',
 'CompetitionDistance': 1270.0,
 'CompetitionOpenSinceMonth': 9.0,
 'CompetitionOpenSinceYear': 2008.0,
 'Promo2': 0,
 'Promo2SinceWeek': 0,
 'Promo2SinceYear': 0,
 'PromoInterval': 0}
    ]
}

response = requests.post(url,json=request,headers=headers)
print(response.json())
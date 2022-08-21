from predict import predict
import requests

request ={'Id': 1,
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

url = 'http://192.168.23.104:9696/predict'
response=requests.post(url,json=request)
print(response.json())

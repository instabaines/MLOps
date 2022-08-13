from concurrent.futures import process
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class TransformData(BaseEstimator,TransformerMixin):
    def __init__(self):
        pass

    def fit_transform(self,data,y=0):
        return self.transform(data)
    def fit(self,data):
        pass
    def transform(self,data,y=0):
        # label encode some features
        isTest= False if 'Sales' in data.columns else True
        mappings = {'0':0, 'a':1, 'b':2, 'c':3, 'd':4}
        data.StoreType.replace(mappings, inplace=True)
        data.Assortment.replace(mappings, inplace=True)
        data.StateHoliday.replace(mappings, inplace=True)
        
        # extract some features from date column  
        
        data['Date'] = pd.to_datetime(data['Date'], infer_datetime_format=True)
        
        data['Month'] = data.Date.dt.month
        data['Year'] = data.Date.dt.year
        data['Day'] = data.Date.dt.day
        data['WeekOfYear'] = data.Date.dt.weekofyear
        
        # calculate competiter open time in months
        data['CompetitionOpen'] = 12 * (data.Year - data.CompetitionOpenSinceYear) + \
            (data.Month - data.CompetitionOpenSinceMonth)
        data['CompetitionOpen'] = data['CompetitionOpen'].apply(lambda x: x if x > 0 else 0)
        
        # calculate promo2 open time in months
        data['PromoOpen'] = 12 * (data.Year - data.Promo2SinceYear) + \
            (data.WeekOfYear - data.Promo2SinceWeek) / 4.0
        data['PromoOpen'] = data['PromoOpen'].apply(lambda x: x if x > 0 else 0)
                                                    
        # Indicate whether the month is in promo interval
        month2str = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', \
                7:'Jul', 8:'Aug', 9:'Sept', 10:'Oct', 11:'Nov', 12:'Dec'}
        data['month_str'] = data.Month.map(month2str)

        def check(row):
            if isinstance(row['PromoInterval'],str) and row['month_str'] in row['PromoInterval']:
                return 1
            else:
                return 0
            
        data['IsPromoMonth'] =  data.apply(lambda row: check(row),axis=1)    
        
        # select the features we need
        features = ['Store', 'DayOfWeek', 'Promo', 'StateHoliday', 'SchoolHoliday',
        'StoreType', 'Assortment', 'CompetitionDistance',
        'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'Promo2',
        'Promo2SinceWeek', 'Promo2SinceYear', 'Year', 'Month', 'Day',
        'WeekOfYear', 'CompetitionOpen', 'PromoOpen', 'IsPromoMonth']  
        # if not isTest:
        #     features.append('Sales')
        #     # only use data of Sales>0 and Open is 1
        #     data=data[(data.Open != 0)&(data.Sales >0)]
            
        data = data[features]
        
        return data







def load_data(path_to_train,path_to_store):
    na_value=['',' ','nan','Nan','NaN','na']
    train=pd.read_csv(path_to_train,na_values=na_value)
    store=pd.read_csv(path_to_store,na_values=na_value)
    

    #fill missing value
    store.fillna(0, inplace=True)
  
    train = pd.merge(train, store, on='Store')
    return train






def prepare_data(path_to_train,path_to_store):
    train= load_data(path_to_train,path_to_store)
    train = train.sort_values(['Date'],ascending = False)
    train =process(train)
    return train


# def main(path):
    


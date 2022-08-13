from prepare import load_data
from sklearn.pipeline import Pipeline
import mlflow
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
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

mlflow.set_tracking_uri("sqlite:///backend.db")
mlflow.set_experiment("random_forest_pipeline_experiment")
#import the necessary libraries




# define eval metrics
def rmspe(y, yhat):
    return np.sqrt(np.mean((yhat/y-1) ** 2))

def rmspe_xg(yhat, y):
    y = np.expm1(y.get_label())
    yhat = np.expm1(yhat)
    return "rmspe", rmspe(y,yhat)

# def train_model(x_train,y_train,x_valid,y_valid):
processsor = TransformData()

def prepare_data_for_training(train):
    train=train[(train.Open != 0)&(train.Sales >0)]
    split_index = 6*7*1115
    valid = train[:split_index] 
    train = train[split_index:]
    valid.sort_index(inplace = True)
    train.sort_index(inplace = True)
    # split x and y
    x_train, y_train = train.drop(columns = ['Sales']), np.log1p(train['Sales'])
    x_valid, y_valid = valid.drop(columns = ['Sales']), np.log1p(valid['Sales'])
    return x_train,y_train,x_valid,y_valid

def train():
    with mlflow.start_run():
        paths=['../data/raw/train.csv','../data/raw/store.csv']
        train=load_data(paths[0],paths[1])
        train=train.sort_values(['Date'],ascending = False)
        for n_estimators in [15,20,50,100]:
            n_estimators = 15
            pipe = Pipeline([('processor', TransformData()), ('clf', RandomForestRegressor(n_estimators = n_estimators ))])
        # train=processsor.fit_transform(train,isTest=False)
            mlflow.log_param('path_to_data',paths)
            mlflow.log_param('n_estimators',n_estimators)
            x_train,y_train,x_valid,y_valid = prepare_data_for_training(train)
            # clf = RandomForestRegressor(n_estimators = n_estimators )
            pipe.fit(x_train, y_train)
            # validation
            y_pred = pipe.predict(x_valid)
            error = rmspe(np.expm1(y_valid), np.expm1(y_pred))
            mlflow.log_metric("error", error)
            print('RMSPE: {:.4f}'.format(error))
                

            mlflow.sklearn.log_model(pipe, artifact_path="models")
        print(f"default artifacts URI: '{mlflow.get_artifact_uri()}'")


if __name__=='__main__':
    train()
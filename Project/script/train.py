from prepare import prepare_data
#import the necessary libraries

import numpy as np
from sklearn.ensemble import RandomForestRegressor


# define eval metrics
def rmspe(y, yhat):
    return np.sqrt(np.mean((yhat/y-1) ** 2))

def rmspe_xg(yhat, y):
    y = np.expm1(y.get_label())
    yhat = np.expm1(yhat)
    return "rmspe", rmspe(y,yhat)

def train_model(x_train,y_train,x_valid,y_valid):
    clf = RandomForestRegressor(n_estimators = 15)
    clf.fit(x_train, y_train)
    # validation
    y_pred = clf.predict(x_valid)
    error = rmspe(np.expm1(y_valid), np.expm1(y_pred))
    print('RMSPE: {:.4f}'.format(error))

def prepare_data_for_training(train):
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
    train=prepare_data('Project/data/raw/train.csv','Project/data/raw/store.csv')
    x_train,y_train,x_valid,y_valid = prepare_data_for_training(train)
    train_model(x_train,y_train,x_valid,y_valid)


    # split x and y
    x_train, y_train = train.drop(columns = ['Sales']), np.log1p(train['Sales'])

if __name__=='__main__':
    train()
from prepare import load_data,TransformData
from sklearn.pipeline import Pipeline
import mlflow

mlflow.set_tracking_uri("sqlite:///backend.db")
mlflow.set_experiment("random_forest_pipeline_experiment")
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
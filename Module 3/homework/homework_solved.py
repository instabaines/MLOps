from copyreg import pickle
import pandas as pd
from pendulum import datetime
import pendulum

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from prefect import flow, task,get_run_logger
from prefect.task_runners import SequentialTaskRunner

@task
def read_data(path):
    df = pd.read_parquet(path)
    return df
@task
def prepare_features(df, categorical, train=True):
    logger = get_run_logger()
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    mean_duration = df.duration.mean()
    if train:
        logger.info(f"The mean duration of training is {mean_duration}")
    else:
        logger.info(f"The mean duration of validation is {mean_duration}")
    
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df
@task
def train_model(df, categorical):
    logger = get_run_logger()
    train_dicts = df[categorical].to_dict(orient='records')
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts) 
    y_train = df.duration.values

    logger.info(f"The shape of X_train is {X_train.shape}")
    logger.info(f"The DictVectorizer has {len(dv.feature_names_)} features")

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_train)
    mse = mean_squared_error(y_train, y_pred, squared=False)
    logger.info(f"The MSE of training is: {mse}")
    return lr, dv
@task
def run_model(df, categorical, dv, lr):
    val_dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(val_dicts) 
    y_pred = lr.predict(X_val)
    y_val = df.duration.values
    logger = get_run_logger()
    mse = mean_squared_error(y_val, y_pred, squared=False)
    logger.info(f"The MSE of validation is: {mse}")
    return
@task
def get_paths(date=None):
    if date==None:
        date=pendulum.now()
        # date=str(date.date())
        train_date=date.subtract(months=2).strftime("%Y-%m")
        test_date=date.subtract(months=1).strftime("%Y-%m")
    else:
        dt=date.split('-') if '-' in date else "date".split('/')
        dt=[int(x) for x in dt]
        date=datetime(*dt)
        train_date=date.subtract(months=2).strftime("%Y-%m")
        test_date=date.subtract(months=1).strftime("%Y-%m")
    train_path=f"../data/fhv_tripdata_{train_date}.parquet"
    test_path=f"../data/fhv_tripdata_{test_date}.parquet"

    return train_path,test_path
@task
def save_model(date,model,vect):
    import pickle
    with open(f"../models/model-{date}.bin",'wb') as f:
        pickle.dump(model,f)
    with open(f"../models/dv-{date}.b",'wb') as f:
        pickle.dump(vect,f)
@flow(task_runner=SequentialTaskRunner())
def main(date="2021-08-15"):
    train_path,val_path = get_paths(date).result()
    categorical = ['PUlocationID', 'DOlocationID']

    df_train = read_data(train_path)
    df_train_processed = prepare_features(df_train, categorical)

    df_val = read_data(val_path)
    df_val_processed = prepare_features(df_val, categorical, False)

    # train the model
    lr, dv = train_model(df_train_processed, categorical).result()
    run_model(df_val_processed, categorical, dv, lr)
    save_model(date,lr, dv)

from prefect.deployments import DeploymentSpec
from prefect.orion.schemas.schedules import CronSchedule
from prefect.flow_runners import SubprocessFlowRunner
from datetime import timedelta


DeploymentSpec(
flow=main,
name="model_training_by_baines",
schedule=CronSchedule(
    cron="0 9 15 * *",
)
,
tags=['homework'],
flow_runner=SubprocessFlowRunner(),
)

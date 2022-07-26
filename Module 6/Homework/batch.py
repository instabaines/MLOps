#!/usr/bin/env python
# coding: utf-8

import sys
import pickle
import pandas as pd



def read_data(filename,categorical):
    df = pd.read_parquet(filename)
    
    return df

def preprare_data(df,categorical):
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df

def predict(dicts):
    with open('model.bin', 'rb') as f_in:
        dv, lr = pickle.load(f_in)

    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)
    return y_pred


def main(year,month):
    year = int(year)
    month = int(month)

    input_file = f'https://raw.githubusercontent.com/alexeygrigorev/datasets/master/nyc-tlc/fhv/fhv_tripdata_{year:04d}-{month:02d}.parquet'
    output_file = f'taxi_type=fhv_year={year:04d}_month={month:02d}.parquet'


    

    categorical = ['PUlocationID', 'DOlocationID']
    df = read_data(input_file,categorical)
    df= preprare_data(df,month,year,categorical)
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

    dicts = df[categorical].to_dict(orient='records')
    
    y_pred=predict(dicts)

    print('predicted mean duration:', y_pred.mean())


    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['predicted_duration'] = y_pred

    df_result.to_parquet(output_file, engine='pyarrow', index=False)

if __name__=='__main__':
    main('2021','02')



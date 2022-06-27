#!/usr/bin/env python
# coding: utf-8




import pickle
import pandas as pd
import argparse




categorical = ['PUlocationID', 'DOlocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df




def scoring(year, month,dv,lr):
    
    df = read_data(f'https://nyc-tlc.s3.amazonaws.com/trip+data/fhv_tripdata_{year}-{month}.parquet')
    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)
    print(f"The mean duration is {y_pred.mean()}")
    return y_pred



def main():
    parser = argparse.ArgumentParser(description='Get year and month')
    parser.add_argument('year',
                       metavar='year',
                       type=int,
                       help='the year')
    parser.add_argument('month',
                       metavar='month',
                       type=str,
                       help='month in digit format')
    args = parser.parse_args()
    year=args.year
    month=args.month
    with open('../model.bin', 'rb') as f_in:
        dv, lr = pickle.load(f_in)
    scoring(year, month,dv, lr)

if __name__=="__main__":
    main()
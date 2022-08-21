from distutils.log import debug
import pickle
from flask import Flask, request,jsonify
import numpy as np
import os
import requests
from pymongo import MongoClient

EVIDENTLY_SERVICE_ADDRESS = os.getenv('EVIDENTLY_SERVICE', 'http://127.0.0.1:5000')
MONGODB_ADDRESS = os.getenv("MONGODB_ADDRESS", "mongodb://127.0.0.1:27017")


model = pickle.load(open('model.pkl','rb'))
mongo_client = MongoClient(MONGODB_ADDRESS)
db = mongo_client.get_database("prediction_service")
collection = db.get_collection("data")

app = Flask('Sales prediction')
def predict(data):
    y=model.predict(data)
    return np.expm1(y[0])

def save_to_db(record, prediction):
    rec = record.copy()
    rec['prediction'] = prediction
    collection.insert_one(rec)


def send_to_evidently_service(record, prediction):
    rec = record.copy()
    rec['prediction'] = prediction
    requests.post(f"{EVIDENTLY_SERVICE_ADDRESS}/iterate/taxi", json=[rec])


@app.route('/predict',methods=['POST'])
def predict_enpoint():
    data=request.get_json()
    pred = predict(data)

    save_to_db(data, float(pred))
    send_to_evidently_service(data, float(pred))

    result ={
        'Predicted Sales':pred
    }
    return jsonify(result)

if __name__=='__main__':
    app.run(debug=True,host='0.0.0.0',port=9696)
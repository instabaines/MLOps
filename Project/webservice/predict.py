from distutils.log import debug
import pickle
from flask import Flask, request,jsonify
import numpy as np
model = pickle.load(open('model.pkl','rb'))

app = Flask('Sales prediction')
def predict(data):
    y=model.predict(data)
    return np.expm1(y[0])
@app.route('/predict',methods=['POST'])
def predict_enpoint():
    data=request.get_json()
    pred = predict(data)

    result ={
        'Predicted Sales':pred
    }
    return jsonify(result)

if __name__=='__main__':
    app.run(debug=True,host='0.0.0.0',port=9696)
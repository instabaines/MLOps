from starter import scoring
from flask  import Flask, request,jsonify

app=Flask('duration prediction')
@app.route('/predict',methods=['POST'])
def endpoint():
    ride=request.get_json()
    pred = scoring(ride)
    result ={
        'duration':pred.mean()
    }
    return jsonify(result)

if __name__=="__main__":
    app.run(debug=True,host='0.0.0.0',port=9696)
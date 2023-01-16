import pickle
from flask import Flask,request,jsonify,render_template
import pandas as pd
import numpy as np

app = Flask(__name__)
lrmodel=pickle.load(open('clinical_data_model.pickle','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data=np.array(list(data.values())).reshape(1,-1)
    output=lrmodel.predict(new_data)
    print(output[0])
    return jsonify(output[0])


if __name__ == "__main__":
    app.run(debug=True)

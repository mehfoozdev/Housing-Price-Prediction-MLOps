from flask import Flask, request, app, jsonify, url_for, render_template
import json
import pickle
import numpy as np
import pandas as pd


app = Flask(__name__)

## Load the model
regmodel = pickle.load(open('regmodel.pkl','rb'))
scalar = pickle.load(open('scaling.pkl','rb'))



# Route for a home page
@app.route('/')
def home():
    return render_template('home.html')


# Route for a prediction
@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']
    print("Data : ", data.values())

    transformed_data = scalar.transform(np.array(list(data.values())).reshape(1,-1))
    print("Transformed Data : ", transformed_data)

    prediction = regmodel.predict(transformed_data)
    print("Prediction : ", prediction)

    

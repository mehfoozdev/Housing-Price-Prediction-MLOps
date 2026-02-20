from flask import Flask, request, jsonify, url_for, render_template, redirect
import pickle
import numpy as np

app = Flask(__name__)

## Load the model
regmodel = pickle.load(open('regmodel.pkl','rb'))
scalar = pickle.load(open('scaling.pkl','rb'))

# Route for a home page
@app.route('/')
def home():
    return render_template('home.html', prediction=None)

# Route for a prediction
@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']   
    transformed_data = scalar.transform(np.array(list(data.values())).reshape(1,-1))   
    prediction = regmodel.predict(transformed_data)
    print("Prediction : ", prediction[0])
    
    return jsonify(prediction[0])


@app.route("/predict", methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    transformed_input = scalar.transform(np.array(data).reshape(1,-1))
    
    output = regmodel.predict(transformed_input)[0]
    print("Predicted Price:", output)
    
    # Redirect back with prediction as query parameter
    return render_template('home.html', prediction=output)

if __name__ == "__main__":
    app.run(debug=True)
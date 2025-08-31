from flask import Flask, request , jsonify , render_template
import pickle
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.preprocessing import StandardScaler 
application = Flask(__name__)
app = application

## import ridge and standard scaler
ridge_model = pickle.load(open('models/ridge.pkl' , 'rb'))
standard_scaler = pickle.load(open('models/scaler.pkl' , 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata' , methods = ['GET' , 'POST'])
def predict_datapoint():
    if request.method == "POST":
         # Collect inputs from form
        Temperature = float(request.form['Temperature'])
        RH = float(request.form['RH'])
        Ws = float(request.form['Ws'])
        Rain = float(request.form['Rain'])
        FFMC = float(request.form['FFMC'])
        DMC = float(request.form['DMC'])
        ISI = float(request.form['ISI'])
        Classes = float(request.form['Classes'])   # careful: this is usually the target!
        Region = float(request.form['Region'])

        features = standard_scaler.transform([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes , Region]])
        prediction = ridge_model.predict(features)[0]
        return render_template('home.html', prediction_text=f"Prediction: {prediction}")
    else :
        return render_template('home.html')

if __name__ == "__main__":
    # Runs on port 5000 by default, change if needed
    app.run(host="0.0.0.0", port=5001, debug=True)

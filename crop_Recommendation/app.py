# -*- coding: utf-8 -*-
"""
Created on Thu May 20 21:01:58 2021

@author: hell0o_hell
"""

# Importing essential libraries
from flask import Flask, render_template, request
import pickle
import numpy as np
import config
import requests

# Load the Random Forest CLassifier model
filename = 'XGBoost.pkl'
classifier = pickle.load(open(filename, 'rb'))

app = Flask(__name__)


def weather_fetch(city_name):
    """
    Fetch and returns the temperature and humidity of a city
    :params: city_name
    :return: temperature, humidity
    """
    api_key = config.weather_api_key
    base_url = "http://api.openweathermap.org/data/2.5/weather?"

    complete_url = base_url + "appid=" + api_key + "&q=" + city_name
    response = requests.get(complete_url)
    x = response.json()

    if x["cod"] != "404":
        y = x["main"]

        temperature = round((y["temp"] - 273.15), 2)
        humidity = y["humidity"]
        return temperature, humidity
    else:
        return None




@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        n = int(request.form['nitrogen'])
        p = int(request.form['phosphorus'])
        k = int(request.form['potassium'])
        ph = float(request.form['ph'])
        rain = float(request.form['rainfall'])
        
        
        city = request.form.get("city")

        if weather_fetch(city) != None:
            temperature, humidity = weather_fetch(city)
            data = np.array([[n, p, k, temperature, humidity, ph, rain]])
            my_prediction = classifier.predict(data)
            final_prediction = my_prediction[0]

            return render_template('index.html', prediction=final_prediction)
        

        
          

if __name__ == '__main__':
	app.run(debug=True)
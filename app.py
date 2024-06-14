from flask import Flask,request,render_template
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline


application=Flask(__name__)

app=application

## Route for a home page

@app.route('/')
def index():
    return render_template('home.html') 

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            MinTemp=float(request.form.get('MinTemp')),
            MaxTemp=float(request.form.get('MaxTemp')),
            Rainfall=float(request.form.get('Rainfall')),
            Evaporation=float(request.form.get('Evaporation')),
            Sunshine=float(request.form.get('Sunshine')),
            WindGustSpeed=int(request.form.get('WindGustSpeed')),
            WindSpeed9am=int(request.form.get('WindSpeed9am')),
            WindSpeed3pm=int(request.form.get('WindSpeed3pm')),
            Humidity9am=int(request.form.get('Humidity9am')),
            Humidity3pm=int(request.form.get('Humidity3pm')),
            Pressure9am=float(request.form.get('Pressure9am')),
            Pressure3pm=float(request.form.get('Pressure3pm')),
            Cloud9am=int(request.form.get('Cloud9am')),
            Cloud3pm=int(request.form.get('Cloud3pm')),
            Temp9am=float(request.form.get('Temp9am')),
            Temp3pm=float(request.form.get('Temp3pm')),
            RainToday=int(request.form.get('RainToday')),
            Location=request.form.get('Location'),
            WindGustDir=request.form.get('WindGustDir'),
            WindDir9am=request.form.get('WindDir9am'),
            WindDir3pm=request.form.get('WindDir3pm'),
            season=request.form.get('season')
        )
        
        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline=PredictPipeline()
        print("Mid Prediction")
        results=predict_pipeline.predict(pred_df)
        print("after Prediction")
        return render_template('home.html',results=results[0])
    

if __name__=="__main__":
    app.run(host="0.0.0.0") 


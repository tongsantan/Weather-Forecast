import sys
import pandas as pd
from src.exception import CustomException
from src.utils.common import load_object
import os

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("output", "model_trainer", "model.pkl")
            preprocessor_path=os.path.join('output',"data_transformation", 'preprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)

class CustomData:
    def __init__(  self,
        MinTemp: float,
        MaxTemp: float,
        Rainfall: float,
        Evaporation: float,
        Sunshine: float,
        WindGustSpeed: int,
        WindSpeed9am: int, 
        WindSpeed3pm: int,
        Humidity9am: int,
        Humidity3pm: int,
        Pressure9am: float,
        Pressure3pm: float,
        Cloud9am: int,
        Cloud3pm: int,
        Temp9am: float,
        Temp3pm: float,
        RainToday: int,
        Location,
        WindGustDir,
        WindDir9am,
        WindDir3pm,
        season):

        self.MinTemp = MinTemp

        self.MaxTemp = MaxTemp

        self.Rainfall = Rainfall

        self.Evaporation = Evaporation

        self.Sunshine = Sunshine

        self.WindGustSpeed = WindGustSpeed

        self.WindSpeed9am = WindSpeed9am

        self.WindSpeed3pm = WindSpeed3pm

        self.Humidity9am = Humidity9am

        self.Humidity3pm = Humidity3pm

        self.Pressure9am = Pressure9am

        self.Pressure3pm = Pressure3pm

        self.Cloud9am = Cloud9am

        self.Cloud3pm = Cloud3pm

        self.Temp9am = Temp9am

        self.Temp3pm = Temp3pm

        self.RainToday = RainToday

        self.Location = Location

        self.WindGustDir = WindGustDir

        self.WindDir9am = WindDir9am

        self.WindDir3pm = WindDir3pm

        self.season = season

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "MinTemp": [self.MinTemp],
                "MaxTemp": [self.MaxTemp],
                "Rainfall": [self.Rainfall],
                "Evaporation": [self.Evaporation],
                "Sunshine": [self.Sunshine],
                "WindGustSpeed": [self.WindGustSpeed],
                "WindSpeed9am": [self.WindSpeed9am],
                "WindSpeed3pm": [self.WindSpeed3pm],
                "Humidity9am": [self.Humidity9am],
                "Humidity3pm": [self.Humidity3pm],
                "Pressure9am": [self.Pressure9am],
                "Pressure3pm": [self.Pressure3pm],
                "Cloud9am": [self.Cloud9am],
                "Cloud3pm": [self.Cloud3pm],
                "Temp9am": [self.Temp9am],
                "Temp3pm": [self.Temp3pm],
                "RainToday": [self.RainToday],
                "Location": [self.Location],
                "WindGustDir": [self.WindGustDir],
                "WindDir9am": [self.WindDir9am],
                "WindDir3pm": [self.WindDir3pm],
                "season": [self.season]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
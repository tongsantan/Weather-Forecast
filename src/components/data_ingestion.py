## 5. Update the components

import os
import sys
from src.exception import CustomException
from src import logger
import pandas as pd
from src.entity.config_entity import DataIngestionConfig
from sklearn.model_selection import StratifiedShuffleSplit
from dataclasses import dataclass

## 5. Update the components

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def data_preprocessing_feature_engineering(self):
        logger.info("Data preprocessing and feature engineering") 
        
        try:
            weather_dataset = pd.read_csv(self.config.input_data_path)
            
            weather_dataset = weather_dataset[weather_dataset["RainTomorrow"].notna()]

            spring = [3,4,5]
            summer = [6,7,8]
            autumn = [9,10,11]
            winter = [12,1,2]

            # create a user-defined function, month_to_season, that takes in a list of month as a parameter and return the season

            def month_to_season(month):
                """return the season of the year"""
                if month in spring: 
                    return 'spring'
                elif month in summer:
                    return 'summer'
                elif month in autumn:
                    return 'autumn'
                elif month in winter:
                    return 'winter'

            # convert the 'Date' column to a date-time datatype
            # create a new 'month' column by extracting the month of the 'Date' column using df[].dt.month
            # map the user-defined function, month_to_season, to the 'month' column of the dataset 
            # and assigned it to a new 'season' column

            weather_dataset['Date'] = pd.to_datetime(weather_dataset['Date'], format='%Y-%m-%d')
           
            weather_dataset['month'] = weather_dataset['Date'].dt.month
            
            weather_dataset['season'] = weather_dataset['month'].map(month_to_season)

            weather_dataset.drop(['Date', 'month'], axis=1, inplace=True)
            
            weather_dataset["RainToday"] = weather_dataset["RainToday"].map({'Yes': 1, 'No': 0})
            
            weather_dataset["RainTomorrow"] = weather_dataset["RainTomorrow"].map({'Yes': 1, 'No': 0})
           
            os.makedirs(os.path.dirname(self.config.processed_data_path),exist_ok=True)

            weather_dataset.to_csv(self.config.processed_data_path,index=False,header=True)

            return(
                self.config.processed_data_path
            )

        except Exception as e:
            raise CustomException(e,sys)    
        

    def complete_data_ingestion(self):
        logger.info("Resume data ingestion method or component") 

        try:  

            df=pd.read_csv(self.config.processed_data_path)
           
            logger.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.config.train_data_path),exist_ok=True)

            df.to_csv(self.config.raw_data_path,index=False,header=True)

            strat_shuff_split = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
        
            X = df.drop(columns=['RainTomorrow'],axis=1)
            
            y = df['RainTomorrow']
            
            train_idx, test_idx = next(strat_shuff_split.split(X, y))
            
            train_set = df.loc[train_idx]
            
            test_set = df.loc[test_idx]

            logger.info("Train test split initiated")

            train_set.to_csv(self.config.train_data_path,index=False,header=True)

            test_set.to_csv(self.config.test_data_path,index=False,header=True)

            logger.info("Ingestion of the data is completed")

            return(
                self.config.train_data_path,
                self.config.test_data_path

            )
        except Exception as e:
            raise CustomException(e,sys)
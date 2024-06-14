import os
import sys
from src.exception import CustomException
from src import logger
import pandas as pd

from sklearn.model_selection import StratifiedShuffleSplit
from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('output',"train.csv")
    test_data_path: str=os.path.join('output',"test.csv")
    raw_data_path: str=os.path.join('output',"data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logger.info("Entered the data ingestion method or component")
        try:
            df=pd.read_csv('notebook\processed_data\weather.csv')
            logger.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            strat_shuff_split = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
        
            X = df.drop(columns=['RainTomorrow'],axis=1)
            
            y = df['RainTomorrow']
            
            train_idx, test_idx = next(strat_shuff_split.split(X, y))
            
            train_set = df.loc[train_idx]
            
            test_set = df.loc[test_idx]

            logger.info("Train test split initiated")

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)

            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logger.info("Ingestion of the data is completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path

            )
        except Exception as e:
            raise CustomException(e,sys)




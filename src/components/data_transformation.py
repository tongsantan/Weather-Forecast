## 5. Update the components
import os
import sys
import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from feature_engine.encoding import OrdinalEncoder
from src.entity.config_entity import DataTransformationConfig
from src.exception import CustomException
from src import logger

from src.utils.common import save_object

## 5. Update the components

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config=config

    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation
        
        '''
        try:
            numerical_columns = ['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine',
                                'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am',
                                'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm',
                                'Temp9am', 'Temp3pm', 'RainToday']
            
            categorical_columns = ['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'season']
            
            num_pipeline= Pipeline(
                    steps=[
                            ("imputer",SimpleImputer(strategy="median")),
                            ("scaler",StandardScaler())
                    ]
                )

            logger.info(f"Numerical columns: {numerical_columns}")

            cat_pipeline=Pipeline(
                steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("ordinal_encoder", OrdinalEncoder(encoding_method='arbitrary')),
                        ("scaler", StandardScaler(with_mean=False))
                ]
            )  
            
            logger.info(f"Categorical columns: {categorical_columns}")

            preprocessor=ColumnTransformer(
                    [
                        ("num_pipeline",num_pipeline,numerical_columns),
                        ("cat_pipelines",cat_pipeline,categorical_columns)
                    ]
                )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self):

        try:
            train_df=pd.read_csv(self.config.train_data_path)
            test_df=pd.read_csv(self.config.test_data_path)

            logger.info("Read train and test data completed")

            logger.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_object()

            target_column_name="RainTomorrow"

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logger.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr=preprocessing_obj.fit(input_feature_train_df, target_feature_train_df)
            input_feature_train_arr=preprocessing_obj.transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            np.save(self.config.train_array_path, train_arr)
            np.save(self.config.test_array_path, test_arr)

            logger.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            return (

                self.config.train_array_path,
                self.config.test_array_path,
                self.config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise CustomException(e,sys)
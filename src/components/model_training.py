## 5. Update the components

import numpy as np
import os
import sys
from dataclasses import dataclass
from sklearn.metrics import confusion_matrix
from src.exception import CustomException
from src import logger
from src.utils.common import save_object,evaluate_models
from src.params import params
from src.models import models
import warnings
warnings.filterwarnings("ignore")
from src.entity.config_entity import ModelTrainerConfig

## 5. Update the components

class ModelTrainer:
    def __init__(self, config:ModelTrainerConfig):
        self.config=config


    def initiate_model_trainer(self):
        try:
            train_array=np.load(self.config.train_array_path)
            test_array=np.load(self.config.test_array_path)
            logger.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            model_df=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models(),param=params())
            
            ## To get best model score from dict
            best_model_score = model_df.loc[model_df['F1_Score'].idxmax(), 'F1_Score']

            ## To get best model name from dict
            best_model_name = model_df.loc[model_df['F1_Score'].idxmax(), 'Model_Name']

            best_model = models()[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            
            logger.info(f"Best found model on both training and testing dataset")

            best_model = best_model.fit(X_train, y_train)

            save_object(
                file_path=self.config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            cm = confusion_matrix(y_test, predicted)

            logger.info(f"best model: {best_model_name}; best F1 score: {best_model_score}; confusion matrix: {cm};")

            return best_model_name, best_model_score, cm
            

        except Exception as e:
            raise CustomException(e,sys)
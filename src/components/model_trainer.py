import os
import sys
from dataclasses import dataclass
from sklearn.metrics import confusion_matrix
from src.exception import CustomException
from src import logger
from src.utils import save_object,evaluate_models
from src.params import params
from src.models import models
import warnings
warnings.filterwarnings("ignore")

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("output","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,train_array,test_array):
        try:
            logger.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models(),param=params())
            
            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models()[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            
            logger.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            best_model = best_model.fit(X_train, y_train)

            predicted=best_model.predict(X_test)

            cm = confusion_matrix(y_test, predicted)

            return best_model_name, best_model_score, cm
            

        except Exception as e:
            raise CustomException(e,sys)
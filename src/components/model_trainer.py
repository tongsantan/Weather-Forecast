import os
import sys
from dataclasses import dataclass
from xgboost import XGBClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from src.exception import CustomException
from src import logger
from src.utils import save_object,evaluate_models
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
            models = {
                        "Logistic Regression": LogisticRegression(random_state=42),
                        "XGBClassifier": XGBClassifier(),
                        "HistGradientBoostingClassifier": HistGradientBoostingClassifier()

                    }
            
            params={
                        "Logistic Regression":{
                        'C': [10], 
                        'max_iter': [100], 
                        'solver': ['liblinear']
                                        },

                        "XGBClassifier":{
                        'learning_rate': [0.1], 
                        'n_estimators': [128]
                                        },

                        "HistGradientBoostingClassifier":{
                        'learning_rate': [0.1], 
                        'max_depth': [10]
                                }
                                            
                    }

            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models,param=params)
            
            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            
            logger.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            cm = confusion_matrix(y_test, predicted)

            return best_model_name, best_model_score, cm
            

        except Exception as e:
            raise CustomException(e,sys)
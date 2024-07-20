import os
import sys
import yaml
from src import logger
import pandas as pd
import dill
import pickle
from ensure import ensure_annotations
from box.exceptions import BoxValueError
from box import ConfigBox
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score)

from sklearn.model_selection import GridSearchCV, cross_validate, RepeatedStratifiedKFold

from src.exception import CustomException

@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """reads yaml file and returns

    Args:
        path_to_yaml (str): path like input

    Raises:
        ValueError: if yaml file is empty
        e: empty file

    Returns:
        ConfigBox: ConfigBox type
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e

@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """create list of directories

    Args:
        path_to_directories (list): list of path of directories
        ignore_log (bool, optional): ignore if multiple dirs is to be created. Defaults to False.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"created directory at: {path}")

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train,X_test,y_test,models,param):
    # record each set of results
    model_list = []
    params_list = []
    f1_list = []
    recall_list = []
    precision_list = []
    accuracy_list = []
    f1_percent_difference_list = []
    
    for i in range(len(list(models))):
        
        model = list(models.values())[i]
        para=param[list(models.keys())[i]]
        cv = RepeatedStratifiedKFold(n_splits= 5, n_repeats=2, random_state=0)
        gs = GridSearchCV(model,para, cv=cv, n_jobs = -1, scoring = 'roc_auc_ovo_weighted')
        gs.fit(X_train, y_train)
        params_list.append(gs.best_params_)
        model.set_params(**gs.best_params_)
        model.fit(X_train, y_train) # Train model
            
        # Make predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
    
        logger.info(list(models.keys())[i])
        logger.info(len(list(models.keys())[i]) * '-')
    
        logger.info('Model performance')
        train_test_full_error = pd.concat([measure_error(y_train, y_train_pred, 'train'),
                                           measure_error(y_test, y_test_pred, 'test'),
                                          ((measure_error(y_train, y_train_pred, 'train') - measure_error(y_test, y_test_pred,
                                            'validation'))/measure_error(y_test, y_test_pred, 'validation')) * 100],
                                           axis=1)
        train_test_full_error.rename(columns = {0:'difference_(%)'}, inplace = True)
        logger.info(train_test_full_error)
        model_list.append(list(models.keys())[i])
    
        accuracy_list.append(train_test_full_error['test'].values[0])
        precision_list.append(train_test_full_error['test'].values[1])
        recall_list.append(train_test_full_error['test'].values[2])
        f1_list.append(train_test_full_error['test'].values[3])
        f1_percent_difference_list.append(train_test_full_error['difference_(%)'].values[3])
    
        logger.info('='*40)
        logger.info('\n')
    
    model_df = pd.DataFrame(list(zip(model_list, f1_list, f1_percent_difference_list, recall_list, precision_list, accuracy_list, params_list)), columns=['Model_Name', 'F1_Score', "F1 Difference (%)", 'Recall_Score', 'Precision_Score', 'Accuracy_Score', 'Best_Params']).sort_values(by=["F1_Score", "F1 Difference (%)"],ascending=False).reset_index(drop=True)
    model_df.index += 1
    logger.info(model_df)
    return model_df

def measure_error(y_true, y_pred, label):
    return pd.Series({'accuracy': accuracy_score(y_true, y_pred),
                      'precision': precision_score(y_true, y_pred),
                      'recall': recall_score(y_true, y_pred),
                      'f1': f1_score(y_true, y_pred)},
                      name=label)

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
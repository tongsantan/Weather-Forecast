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
    
    report = {}

    for i in range(len(list(models))):

        model = list(models.values())[i]

        para=param[list(models.keys())[i]]

        gs = GridSearchCV(model,para, n_jobs = -1, scoring = 'roc_auc')
        gs.fit(X_train, y_train)

        cv = RepeatedStratifiedKFold(n_splits= 3, n_repeats=1, random_state=0)
        scores = cross_validate(gs, X_train, y_train, scoring=['roc_auc'], cv=cv, n_jobs=-1)

        model.set_params(**gs.best_params_)
        model.fit(X_train, y_train) # Train model

        # Make predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        train_test_full_error = pd.concat([measure_error(y_train, y_train_pred, 'train'),
                                           measure_error(y_test, y_test_pred, 'test')],
                                           axis=1)

        train_model_score = train_test_full_error['train'].values[3]

        test_model_score = train_test_full_error['test'].values[3]

        report[list(models.keys())[i]] = test_model_score
    
    return report

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
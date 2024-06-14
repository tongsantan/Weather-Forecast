import os
import sys

import pandas as pd
import dill
import pickle
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score)


from sklearn.model_selection import GridSearchCV, cross_validate, RepeatedStratifiedKFold


from src.exception import CustomException

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
def params():
    params={
              "k-NN": {"n_neighbors": [10, 20, 30 , 40]},
              
              "Decision Tree":{
                  'max_depth': [1, 3, 5, 7],
                  'max_features': [1, 10, 20 , 30]
                              },
      
              "BernoulliNB":{},
      
              "LogisticRegression":{
                  'solver': ['lbfgs', 'liblinear', 'newton-cholesky'],
                  'C': [0.01, 0.1, 1, 10],
                  'max_iter': [100, 1000, 10000]
                                  },
              
              "HistGradientBoostingClassifier":{
              'learning_rate':[.1,.01,.05,.001],
              'max_depth': [6,8,10]
                  },
      
              "XGBClassifier":{
              'learning_rate':[.1,.01,.001],
              'n_estimators': [8,16,32,64,128]
                              }
                                  
          }
    return params
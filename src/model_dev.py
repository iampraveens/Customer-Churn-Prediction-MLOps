from abc import ABC, abstractmethod
import logging
from typing import Sequence
import pandas as pd
import mlflow

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

class Model(ABC):
    
    @abstractmethod
    def train(self, X_train, y_train):
        pass
    
class DecisionTreeClassifier_Model(Model):
    
    def train(self, X_train, y_train, **kwargs):
        
        
        try:
            dt = DecisionTreeClassifier()
            dt.fit(X_train, y_train)
            logging.info(f"Model training completed")
            return dt
        except Exception as e:
            logging.error(f"Error in training model {e}")
            raise e
        
class RandomForestClassifier_Model(Model):
    
    def train(self, X_train, y_train, criterion: str, max_depth: int, 
              max_features: str, verbose: int, n_estimators: int, n_jobs: int, **kwargs):
        
        try:
            rf = RandomForestClassifier(criterion=criterion, 
                                        max_depth=max_depth, 
                                        max_features=max_features, 
                                        verbose=verbose, 
                                        n_estimators=n_estimators, n_jobs=n_jobs)
            rf.fit(X_train, y_train)
            logging.info(f"Model training completed")
            # fim = pd.Series(rf.feature_importances_, index= X_train.columns)
            # print(fim.sort_values(ascending=False))
            mlflow.log_param('criterion', criterion)
            return rf
        except Exception as e:
            logging.error(f"Error in training model {e}")
            raise e
        
class GridSearchCV_Model(Model):
    
    def train(self, X_train, y_train, estimator,
              scoring: str, cv: int, verbose: int, 
              n_jobs: int, **kwargs):

        try:
            param_grid_dict = {
                                "DecisionTreeClassifier": {
                                    'criterion':['entropy', 'log_loss', 'gini'],
                                },
                                "RandomForestClassifier":{
                                    'n_estimators': [8,16,32,64,128,256],                
                                    'criterion': ['gini', 'entropy'],               
                                    'max_depth': [None, 10, 20, 30],               
                                    'min_samples_split': [2, 5, 10],              
                                    'min_samples_leaf': [1, 2, 4],               
                                    'max_features': ['sqrt', 'log2', None]
                                },
                                "GradientBoostingClassifier":{
                                    'learning_rate':[.1,.01,.05,.001],
                                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                                    'n_estimators': [8,16,32,64,128,256]
                                },
                                "LinearRegression":{},
                                "XGBClassifier":{
                                    'learning_rate':[.1,.01,.05,.001],
                                    'n_estimators': [8,16,32,64,128,256]
                                },
                                "CatBoostRegressor":{
                                    'depth': [6,8,10],
                                    'learning_rate': [0.01, 0.05, 0.1],
                                    'iterations': [30, 50, 100]
                                },
                                "AdaBoostRegressor":{
                                    'learning_rate':[.1,.01,0.5,.001],
                                    'n_estimators': [8,16,32,64,128,256]
                                }
                            }
            estimator_name = estimator.__class__.__name__
            param_grid = param_grid_dict.get(estimator_name, {})
            
            gs = GridSearchCV(estimator=estimator, param_grid=param_grid, 
                              scoring=scoring, cv=cv, 
                              verbose=verbose, n_jobs=n_jobs)
            gs.fit(X_train, y_train)
            logging.info(f"Model training completed")
            # print(gs.best_score_)
            return gs
        except Exception as e:
            logging.error(f"Error in training model {e}")
            raise e
        
class GradientBoostingClassifier_Model(Model):
    
    def train(self, X_train, y_train, criterion: str, max_depth: int, 
              max_features: str, verbose: int, 
              n_estimators: int, learning_rate: float, 
              min_samples_leaf: int, max_leaf_nodes: int, 
              min_samples_split: int, **kwargs):
        
        try:
            gb = GradientBoostingClassifier(criterion=criterion, 
                                            max_depth=max_depth, 
                                            max_features=max_features, 
                                            verbose=verbose, 
                                            n_estimators=n_estimators, learning_rate=learning_rate, 
                                            min_samples_leaf=min_samples_leaf,
                                            max_leaf_nodes=max_leaf_nodes, 
                                            min_samples_split=min_samples_split)
            gb.fit(X_train, y_train)
            logging.info(f"Model training completed")
            # fim = pd.Series(gb.feature_importances_, index= X_train.columns)
            # print(fim.sort_values(ascending=False))
            return gb
        except Exception as e:
            logging.error(f"Error in training model {e}")
            raise e
        
class XGBoost_Model(Model):
    
    def train(self, X_train, y_train, **kwargs):

        try:
            dt = XGBClassifier()
            dt.fit(X_train, y_train)
            logging.info(f"Model training completed")
            return dt
        except Exception as e:
            logging.error(f"Error in training model {e}")
            raise e
            
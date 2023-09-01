import logging

import pandas as pd
import mlflow
from zenml import step
from zenml.client import Client
from sklearn.base import ClassifierMixin 

from src.model_dev import DecisionTreeClassifier_Model
from config import ModelNameConfig

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def train_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    config: ModelNameConfig,
) -> ClassifierMixin:
    """_summary_

    Args:
        df (pd.DataFrame): _description_
    """
    
    try:
        # config = ModelNameConfig
        model = None
        if config.model_name == 'DecisionTree':
            mlflow.sklearn.autolog()
            model = DecisionTreeClassifier_Model()
            trained_model = model.train(X_train, y_train)
            return trained_model
        else:
            raise ValueError(f"Model not supported {config.model_name}")
        
    except Exception as e:
        logging.error(f"Error in  training mdoel {e}")
        raise e
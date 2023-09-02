import logging

import pandas as pd
import mlflow
# from zenml import step
# from zenml.client import Client
from typing import Tuple
from typing_extensions import Annotated
from sklearn.base import ClassifierMixin

from src.evaluation import Accuracy, F1Score, Recall

# experiment_tracker = Client().active_stack.experiment_tracker

# @step(experiment_tracker=experiment_tracker.name)
def evaluate_model(model: ClassifierMixin,
        X_test: pd.DataFrame,
        y_test: pd.DataFrame
) -> Tuple[
    Annotated[float, "accuracy"],
    Annotated[float, "f1score"],
    Annotated[float, "recall"],
]:
    """_summary_

    Args:
        df (pd.DataFrame): _description_
    """
    try:
        prediction = model.predict(X_test)
        accuracy_class = Accuracy()
        accuracy = accuracy_class.calculate_scores(y_test, prediction)
        mlflow.log_metric("accuracy", accuracy)
        
        f1_class = F1Score()
        f1score = f1_class.calculate_scores(y_test, prediction)
        mlflow.log_metric("f1score", f1score)
        
        recall_class = Recall()
        recall = recall_class.calculate_scores(y_test, prediction)
        mlflow.log_metric("recall", recall)
        
        return accuracy, f1score, recall
    
    except Exception as e:
        logging.error(f"Error in evaluating the model {e}")
        raise e
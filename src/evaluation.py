import logging
from abc import ABC, abstractmethod

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score

class Evaluation(ABC):
    """
    Abstract base class for evaluation metrics.
    """
    @abstractmethod
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Calculates evaluation scores based on true and predicted labels.

        Args:
            y_true (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted labels.
        """
        pass
    
class Accuracy(Evaluation):
    """
    Class for calculating accuracy score as an evaluation metric.
    Inherits from the Evaluation abstract class.
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Calculates accuracy score based on true and predicted labels.

        Args:
            y_true (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted labels.

        Returns:
            float: Accuracy score.
        """
        try:
            logging.info("Calculating Accuracy Score")
            accuracy = accuracy_score(y_true, y_pred)
            logging.info(f"Accuracy Score: {accuracy}")
            return accuracy
        
        except Exception as e:
            logging.error(f"Error in calculating Accuracy Score {e}")
            raise e
        
class F1Score(Evaluation):
    """
    Class for calculating F1 score as an evaluation metric.
    Inherits from the Evaluation abstract class.
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Calculates F1 score based on true and predicted labels.

        Args:
            y_true (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted labels.

        Returns:
            float: F1 score.
        """
        try:
            logging.info("Calculating F1 Score")
            f1score = f1_score(y_true, y_pred)
            logging.info(f"F1 Score: {f1score}")
            return f1score
        
        except Exception as e:
            logging.error(f"Error in calculating F1 Score {e}")
            raise e
        
class Recall(Evaluation):
    """
    Class for calculating recall score as an evaluation metric.
    Inherits from the Evaluation abstract class.
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Calculates recall score based on true and predicted labels.

        Args:
            y_true (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted labels.

        Returns:
            float: Recall score.
        """
        try:
            logging.info("Calculating Recall Score")
            recall = recall_score(y_true, y_pred)
            logging.info(f"Recall Score: {recall}")
            return recall
        
        except Exception as e:
            logging.error(f"Error in calculating Recall Score {e}")
            raise e
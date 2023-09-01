import logging
from abc import ABC, abstractmethod

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score

class Evaluation(ABC):
    
    @abstractmethod
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        
        pass
    
class Accuracy(Evaluation):
    
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        
        try:
            logging.info("Calculating Accuracy Score")
            accuracy = accuracy_score(y_true, y_pred)
            logging.info(f"Accuracy Score: {accuracy}")
            return accuracy
        
        except Exception as e:
            logging.error(f"Error in calculating Accuracy Score {e}")
            raise e
        
class F1Score(Evaluation):
    
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        
        try:
            logging.info("Calculating F1 Score")
            f1score = f1_score(y_true, y_pred)
            logging.info(f"F1 Score: {f1score}")
            return f1score
        
        except Exception as e:
            logging.error(f"Error in calculating F1 Score {e}")
            raise e
        
class Recall(Evaluation):
    
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        
        try:
            logging.info("Calculating Recall Score")
            recall = recall_score(y_true, y_pred)
            logging.info(f"Recall Score: {recall}")
            return recall
        
        except Exception as e:
            logging.error(f"Error in calculating Recall Score {e}")
            raise e
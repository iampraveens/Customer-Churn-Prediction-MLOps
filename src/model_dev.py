from abc import ABC, abstractmethod
import logging

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

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
            
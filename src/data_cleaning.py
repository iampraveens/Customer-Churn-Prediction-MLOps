import logging
import sys
from typing import Union
from abc import ABC, abstractmethod
from imblearn.combine import SMOTEENN

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.append("..")

class DataStrategy(ABC):
    """
    Abstract class defining startegy for handing data
   
    """
    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass
    
class DataPreProcessStrategy(DataStrategy):
    
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        
        try:
            data = data.drop(columns=['customerID'], axis=1)
            data.drop(data[data['TotalCharges'] == " "].index, axis=0, inplace=True)
            data['TotalCharges'] = data['TotalCharges'].astype('float')
            
            columns_1 = ['Partner', 'Dependents', 'PaperlessBilling', 'Churn', 'PhoneService']
            for column in columns_1:
                data[column] = data[column].apply(lambda x: 1 if x == 'Yes' else 0)
            
            data['gender'] = data['gender'].apply(lambda x: 1 if x == 'Female' else 0)
            data['MultipleLines'] = data['MultipleLines'].map({'No phone service': 0, 'No': 0, 'Yes': 1})
            
            column_2 = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
            for column in column_2:
                data[column] = data[column].map({'No internet service': 0, 'No': 0, 'Yes': 1})

            data = pd.get_dummies(data, columns=['InternetService', 'Contract', 'PaymentMethod'], drop_first=True, dtype='int')
            return data
        
        except Exception as e:
            logging.error(f"Error in preprocessing the data {e}")
            raise e
        
class DataSplitStrategy(DataStrategy):
    
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        
        try:
            X = data.drop(columns=['Churn'], axis=1)
            y = data['Churn']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=data.Churn)
            return X_train, X_test, y_train, y_test
        
        except Exception as e:
            logging.error(f"Error in spliting the data {e}")
            raise e
        
class DataBalanceStrategy(DataStrategy):
    
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
    
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        
        try:
            smoteen = SMOTEENN()
            X_train, y_train = smoteen.fit_resample(self.X_train, self.y_train)
            logging.info(f"Data balancing completed")
            return X_train, y_train
        
        except Exception as e:
            logging.error(f"Errot in balancing the data {e}")
            raise e
        
class DataCleaning:
    
    def __init__(self, data: pd.DataFrame, strategy: DataStrategy):
        self.data = data
        self.strategy = strategy
        
    def handle_data(self) -> Union[pd.DataFrame, pd.Series]:
        
        try: 
            return self.strategy.handle_data(self.data)
        
        except Exception as e:
            logging.info(f"Error in handing the data {e}")
            raise e
        
        
        
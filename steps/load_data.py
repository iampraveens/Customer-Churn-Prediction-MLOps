import logging

import pandas as pd
# from zenml import step

class LoadData:
    """
    Loading data from data_path
    """
    def __init__(self, data_path: str):
        """
        Args:
            data_path (str): path to the data
        """
        self.data_path = data_path
        
    def get_data(self):
        """
        Loading data from data_path
        """
        logging.info(f"Loading data from {self.data_path}")
        return pd.read_csv(self.data_path)

# @step
def load_df(data_path: str) -> pd.DataFrame:
    """
    Load the data from data_path and convert to a dataframe
    
    Args:
        data_path (str): path to the data
        
    Returns:
        pd.DataFrame: the loaded data
    """
    try:
        load_data = LoadData(data_path)
        df = load_data.get_data()
        return df
    except Exception as e:
        logging.error(f"Error while loading data {e}")
        raise e
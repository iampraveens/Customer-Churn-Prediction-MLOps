import logging

import pandas as pd
from typing_extensions import Annotated
from typing import Tuple
# from zenml import step

from src.data_cleaning import DataCleaning, DataPreProcessStrategy, DataSplitStrategy, DataBalanceStrategy

# @step
def clean_df(df: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"]
    ]:
    """_summary_

    Args:
        df (pd.DataFrame): _description_
    """
    try:
        process_strategy = DataPreProcessStrategy()
        data_cleaning = DataCleaning(df, process_strategy)
        proccessed_data = data_cleaning.handle_data()
        
        split_strategy = DataSplitStrategy()
        data_cleaning = DataCleaning(proccessed_data, split_strategy)
        X_train, X_test, y_train, y_test = data_cleaning.handle_data()
        
        balance_strategy = DataBalanceStrategy(X_train, y_train)
        data_cleaning = DataCleaning(X_train, balance_strategy)
        X_train, y_train = data_cleaning.handle_data()
        
        logging.info("Data cleaning completed")
        return X_train, X_test, y_train, y_test
        
    except Exception as e:
        logging.error(f"Error in cleaning the data {e}")
        raise e
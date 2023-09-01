from zenml import pipeline

from steps.load_data import load_df
from steps.clean_data import clean_df
from steps.model_train import train_model
from steps.model_evaluate import evaluate_model
from steps.config import ModelNameConfig


@pipeline(enable_cache=True)
def train_pipeline(data_path: str):
    
    # config_instance = ModelNameConfig()
    df = load_df(data_path)
    X_train, X_test, y_train, y_test = clean_df(df)
    model = train_model(X_train, X_test, y_train, y_test)
    accuracy, f1score, recall = evaluate_model(model, X_test, y_test)
    
    
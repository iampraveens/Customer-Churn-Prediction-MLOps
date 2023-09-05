from pipelines.training_pipeline import train_pipeline
import mlflow
from urllib.parse import urlparse 

if __name__ == "__main__":
    
    with mlflow.start_run():
        remote_server_uri = "https://dagshub.com/iampraveens/mlflow-experiment-tracking.mlflow"
        mlflow.set_tracking_uri(remote_server_uri)
        tracking_url_store_type = urlparse(mlflow.get_tracking_uri()).scheme
        
        model = train_pipeline(r"C:\Users\sprav\Pictures\Customer Churn Prediction\data\telcoChurn.csv")
        
        if tracking_url_store_type != "file":
            mlflow.sklearn.log_model(model, "model", registered_model_name="RandomForest")
        else:
            mlflow.sklearn.log_model(model, "model")
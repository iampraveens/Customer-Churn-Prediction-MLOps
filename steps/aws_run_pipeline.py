from pipelines.training_pipeline import train_pipeline
import mlflow
from zenml.client import Client
from mlflow.client import MlflowClient
from urllib.parse import urlparse 

if __name__ == "__main__":
    mlflow.end_run()
    # mlflow.set_experiment("telco")
    # experiment = mlflow.get_experiment_by_name("telco")
    
    with mlflow.start_run():
        
        # run = mlflow.active_run()
        # print(run)
        remote_server_uri = "http://ec2-3-109-214-143.ap-south-1.compute.amazonaws.com:5000/"
        mlflow.set_tracking_uri(remote_server_uri)
        tracking_url_store_type = urlparse(mlflow.get_tracking_uri()).scheme
        
        model = train_pipeline(r"C:\Users\sprav\Pictures\Customer Churn Prediction\data\telcoChurn.csv")
        
        if tracking_url_store_type != "file":
            mlflow.sklearn.log_model(model, "model", registered_model_name="DecisionTreeModel")
        else:
            mlflow.sklearn.log_model(model, "model")
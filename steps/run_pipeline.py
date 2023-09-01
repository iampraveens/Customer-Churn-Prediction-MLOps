# import sys
 
from pipelines.training_pipeline import train_pipeline
# sys.path.append("../pipelines") 

if __name__ == "__main__":
    
    # Run the pipeline
    # print(mlflow.get_tracking_uri())
    # train_pipeline("C:/Users/sprav/Desktop/My Projects/Customer Churn Prediction/data/telcoChurn.csv")
    train_pipeline(r"C:\Users\sprav\Desktop\My Projects\Customer Churn Prediction\data\telcoChurn.csv")


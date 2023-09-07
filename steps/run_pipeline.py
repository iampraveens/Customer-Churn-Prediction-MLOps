# import sys
 
from pipelines.training_pipeline import train_pipeline
# sys.path.append("../pipelines") 

if __name__ == "__main__":
    
    # Run the pipeline
    train_pipeline(r"C:\Users\sprav\Pictures\Customer Churn Prediction\data\telcoChurn.csv")


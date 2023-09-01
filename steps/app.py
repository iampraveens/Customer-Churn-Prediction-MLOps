import sys
 
sys.path.append("..")

from load_data import load_df
from clean_data import clean_df
from model_train import train_model
from model_evaluate import evaluate_model
from config import ModelNameConfig

df = load_df(r"C:\Users\sprav\Desktop\My Projects\Customer Churn Prediction\data\telcoChurn.csv")
X_train, X_test, y_train, y_test = clean_df(df)
model = train_model(X_train, X_test, y_train, y_test, ModelNameConfig())
accuracy, f1score, recall = evaluate_model(model, X_test, y_test)

print(f"Accuracy Score: {accuracy} \nF1 Score: {f1score} \nRecall Score: {recall}")
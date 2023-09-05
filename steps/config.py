from zenml.steps import BaseParameters
# from pydantic import BaseModel

class ModelNameConfig(BaseParameters):
    
    model_name: str = 'RandomForest'
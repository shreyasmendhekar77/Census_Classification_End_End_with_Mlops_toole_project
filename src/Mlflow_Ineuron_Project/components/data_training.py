import pandas as pd
import os
from Mlflow_Ineuron_Project import logger
# from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import joblib
from Mlflow_Ineuron_Project.entity.config_entity import ModelTrainerConfig

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    
    def train(self):
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)


        train_x = train_data.drop([self.config.target_column], axis=1)
        test_x = test_data.drop([self.config.target_column], axis=1)
        train_y = train_data[[self.config.target_column]]
        test_y = test_data[[self.config.target_column]]


        classification_model=DecisionTreeClassifier()
        classification_model.fit(train_x,train_y)
        y_pred=classification_model.predict(test_x)
        report=classification_report(test_y,y_pred)
        print(report)

        joblib.dump(classification_model, os.path.join(self.config.root_dir, self.config.model_name))


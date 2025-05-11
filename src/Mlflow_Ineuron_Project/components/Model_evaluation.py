import os
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score,precision_score,f1_score,recall_score,confusion_matrix
import matplotlib.pyplot as plt
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
import numpy as np
import joblib
from Mlflow_Ineuron_Project.entity.config_entity import ModelEvaluationConfig
from Mlflow_Ineuron_Project.utils.common import save_json
from pathlib import Path

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    
    def eval_metrics(self,actual, pred):
        accuracy = accuracy_score(actual, pred)
        precision = precision_score(actual, pred, average='weighted')
        recall = recall_score(actual, pred, average='weighted')
        f1 = f1_score(actual, pred, average='weighted')
        return accuracy, precision, recall, f1

    


    def log_into_mlflow(self):

        test_data = pd.read_csv(self.config.test_data_path)
        model = joblib.load(self.config.model_path)

        test_x = test_data.drop([self.config.target_column], axis=1)
        test_y = test_data[[self.config.target_column]]


        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():
            predicted_qualities = model.predict(test_x)

            # Calculate metrics
            (accuracy, precision, recall, f1) = self.eval_metrics(test_y, predicted_qualities)

             # Saving metrics as local
            scores = {"Accuracy": accuracy, "Precision": precision, "Recall": recall,"F1":f1}
            save_json(path=Path(self.config.metric_file_name), data=scores)

            mlflow.log_params(self.config.all_params)
                # Log metrics
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)

                # Confusion matrix
            # cm = confusion_matrix(test_y, predicted_qualities)
            # plt.figure(figsize=(8, 6))
            # sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
            # plt.title("Confusion Matrix")
            # plt.xlabel("Predicted")
            # plt.ylabel("Actual")

            #     # Save confusion matrix as an image
            # cm_path = "model_evaluation/confusion_matrix.png"
            # plt.savefig(cm_path)
            # plt.close()

            #     # Log confusion matrix as an artifact
            # mlflow.log_artifact(cm_path)

                # Model registry does not work with file store
            if tracking_url_type_store != "file":

                    # Register the model
                    # There are other ways to use the Model Registry, which depends on the use case,
                    # please refer to the doc for more information:
                    # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                mlflow.sklearn.log_model(model, "model", registered_model_name="Decision_tree")
            else:
                mlflow.sklearn.log_model(model, "model")

        

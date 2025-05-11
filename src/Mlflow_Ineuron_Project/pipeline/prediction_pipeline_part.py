import joblib
import numpy as np
import pandas as pd
from pathlib import Path




class PredictionPipeline:
    def __init__(self):
        self.model = joblib.load(Path('artifacts/model_trainer/model.joblib'))

    
    def predict(self, data):
        prediction = self.model.predict(data)

        return prediction

class CustomData:
    def __init__(
        self,
        age: int,
        workclass: str,
        fnlwgt: int,
        education: str,
        marital_status: str,
        occupation: str,
        sex: str,
        capital_gain: int,
        capital_loss: int,
        hours_per_week: int,
        country: str
    ):
        self.age = age
        self.workclass = workclass
        self.fnlwgt = fnlwgt
        self.education = education
        self.marital_status = marital_status
        self.occupation = occupation
        self.sex = sex
        self.capital_gain = capital_gain
        self.capital_loss = capital_loss
        self.hours_per_week = hours_per_week
        self.country = country

    def get_data_as_data_frame(self):
        try:
            # custom_data_input_dict = {
            #     "age": [self.age],
            #     "workclass": [self.workclass],
            #     "fnlwgt": [self.fnlwgt],
            #     "education": [self.education],
            #     "marital-status": [self.marital_status],
            #     "occupation": [self.occupation],
            #     "sex": [self.sex],
            #     "capital-gain": [self.capital_gain],
            #     "capital-loss": [self.capital_loss],
            #     "hours-per-week": [self.hours_per_week],
            #     "country": [self.country],
            # }
            custom_data_input_dict = {
                "age": [self.age],
                "fnlwgt": [self.fnlwgt],
                "capital-gain": [self.capital_gain],
                "capital-loss": [self.capital_loss],
                "hours-per-week": [self.hours_per_week],
                "workclass": [self.workclass],
                "education": [self.education],
                "marital-status": [self.marital_status],
                "occupation": [self.occupation],
                "sex": [self.sex],
                "country": [self.country]
            }

      

           
            df=pd.DataFrame(custom_data_input_dict)
            # text_columns = ['workclass', 'education', 'marital-status', 'occupation', 'sex', 'country']
            # for col in text_columns:
            #     if col in df.columns:
            #         df[col] = df[col].str.lower()
            
            return df

        except Exception as e:
            raise e
    



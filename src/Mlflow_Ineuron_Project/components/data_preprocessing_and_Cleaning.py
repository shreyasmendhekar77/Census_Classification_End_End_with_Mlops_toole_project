from Mlflow_Ineuron_Project.entity.config_entity import Data_preprocessing_ValidationConfig
import os
from Mlflow_Ineuron_Project import logger
from  sklearn.preprocessing import LabelEncoder,OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import joblib
import pandas as pd

class Data_preprocessing_Validation:
    def __init__(self, config: Data_preprocessing_ValidationConfig):
        self.config = config


    def data_transformer_object(self):
        try:
            numerical_col=['age','fnlwgt', 'capital-gain', 'capital-loss', 'hours-per-week']

            categorical_col=['workclass', 'education', 'marital-status', 'occupation', 'sex', 'country']
            
       
            # pipeline for numerical columns
            # handling the 
            Numeric_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="mean"))
                ]
            )

            Categorical_pipeline=Pipeline(
               steps= [
                    ("imputer",SimpleImputer(strategy="most_frequent")), 
                    # ("lowercase", lowercase_transformer),  # Convert text to lowercase
                    ("onehot_encoder",OrdinalEncoder())
                    ]
            )

            # logging.info(f"categorical Columns: {categorical_col}")
            # logging.info("pipeline is created for the Column and numeric column transformation")


            preprocessor=ColumnTransformer(
                [
                    ("Numeric_pipeline",Numeric_pipeline,numerical_col),
                    ("Categorical_pipeline",Categorical_pipeline,categorical_col),
                    # ("target_pipeline",target_pipeline,target_col)

                ]
            )

            return preprocessor
        except Exception as e:
            raise e   

    def validate_all_columns(self)-> bool:
        try:
            validation_status = None

            data = pd.read_csv(self.config.unzip_data_dir)
       # remove the extra space around the columns 
            data.columns=data.columns.str.strip()
            categorical_col = ['workclass', 'education', 'marital-status', 'occupation', 'sex', 'country', 'salary']
            for col in categorical_col:
                if col in data.columns:
                    data[col] = data[col].str.strip()

            numerical_col=['age','fnlwgt', 'capital-gain', 'capital-loss', 'hours-per-week']

            # categorical_col=['workclass', 'education', 'marital-status', 'occupation', 'sex', 'country','salary']
            # # for col in categorical_col:
            #     if col in data.columns:
            #         data[col] = data[col].str.lower()
            #     if col in data.columns:
            #         data[col] = data[col].str.lower() 
            # print(data.head())
            data.drop(columns=['relationship', 'race','education-num'], inplace=True)
            preprocessor_object=self.data_transformer_object()
            output=data
            output=preprocessor_object.fit_transform(data)
            # label encoding 
            for i in categorical_col:
                le=LabelEncoder()
                data[i]=le.fit_transform(data[i])
            print(data.head(1))

            all_cols = list(data.columns)

            all_schema = self.config.all_schema.keys()

            print(all_schema)
            for col in all_cols:
                # print(col)
                if col not in all_schema:
                    validation_status = False
                    with open(self.config.STATUS_FILE, 'w') as f:
                        f.write(f"Validation status: {validation_status}")
                else:
                    validation_status = True
                    with open(self.config.STATUS_FILE, 'w') as f:
                        f.write(f"Validation status: {validation_status}")

            joblib.dump(preprocessor_object,os.path.join(self.config.Preprocess_data,"Preprocess_model.joblib"))
            data.to_csv(os.path.join(self.config.Preprocess_data,"preprocessed_data.csv"),index=False)

            return validation_status
        
        except Exception as e:
            raise e




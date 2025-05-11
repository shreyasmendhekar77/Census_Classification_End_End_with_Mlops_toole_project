# import os
# import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
# from src.pipeline.perdiction_pipeline import CustomData,predict_salary
# try:
    # print("Module imported successfully!")
# except Exception as e:
    # print(f"Error importing module: {e}")

from Mlflow_Ineuron_Project.pipeline.prediction_pipeline_part import PredictionPipeline,CustomData

import pandas as pd
from flask import Flask, request, render_template,jsonify
import pickle
import numpy as np
import joblib

application = Flask(__name__)
app = application

@app.route('/')
def home():
    return render_template('index.html')

# API of this model
@app.route('/predictdata.api',methods=['POST'])
def predict_api():
    data=request.json['data']
    print(np.array(list(data.values())).reshape(1,-1))
    data=np.array(list(data.values())).reshape(1,-1)
    model=pd.read_pickle('Model_ML_notebook\\final_model.pkl')
    prediction=model.predict(data)
    return jsonify({'output':int(prediction[0])})

# @app.route('/predictdata', methods=['GET', 'POST'])
# def predict_datapoint():
#     if request.method == 'POST':
#         # Get data from the form
#         age = request.form.get('age')
#         fnlwgt = request.form.get('fnlwgt')s
#         country = request.form.get('country')
#         workclass = request.form.get('workclass')
#         education = request.form.get('education')
#         marital_status = request.form.get('marital-status')
#         occupation = request.form.get('occupation')
#         sex = request.form.get('sex')
#         capital_gain = request.form.get('capital-gain')
#         capital_loss = request.form.get('capital-loss')
#         hours_per_week = request.form.get('hours-per-week')

#         # Apply conditional logic for country and marital-status
#         # country = 1 if country == 'United-States' else 0
#         # marital_status = 1 if marital_status == 'Married' else 0

#         # Create a DataFrame from the form data
#         data = {
#             'age': [age],
#             'workclass': [workclass],
#             'fnlwgt': [fnlwgt],
#             'education': [education],
#             'marital-status': [marital_status],
#             'occupation': [occupation],
#             'sex': [sex],
#             'capital-gain': [capital_gain],
#             'capital-loss': [capital_loss],
#             'hours-per-week': [hours_per_week],
#             'country': [country]

#         }
#         df = pd.DataFrame(data)
#         # df=df.astype('int32')

#         print(df.head())
#         # #CONVERTING THE DATAFRAME TO THE ARRAY
#         # # Creating a sample DataFrame
#         # vector=df.values.ravel()
#         # # Print the DataFrame to verify
#         # # print(df.info())
#         # print(vector)
#         # model=pd.read_pickle('Model_ML_notebook\\final_model.pkl')
#         # prediction=model.predict(vector)
#         # print("prediction is ",prediction,type(prediction))
#         prediction=1
#         out=''
#         if prediction==1:
#             out='Salary Greater than 50'
#         else:
#             out='Salary lesser than 50'

#         # Render the prediction page (you can pass the DataFrame to the template if needed)
#         return render_template('index.html', salary=out)
#     else:
#         return render_template('index.html')


# Using the pipeline for model 
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'POST':
        # Get data from the form
        data=CustomData( 
        age = request.form.get('age'),
        fnlwgt = request.form.get('fnlwgt'),
        country = request.form.get('country'),
        workclass = request.form.get('workclass'),
        education = request.form.get('education'),
        marital_status = request.form.get('marital-status'),
        occupation = request.form.get('occupation'),
        sex = request.form.get('sex'),
        capital_gain = request.form.get('capital-gain'),
        capital_loss = request.form.get('capital-loss'),
        hours_per_week = request.form.get('hours-per-week'),
        )

       
        df = data.get_data_as_data_frame()

        print(df)
        # preprocessor_obj=pd.read_pickle('artifacts\data_preprocessed\preprocessor_object.pkl')
        # df=preprocessor_obj.transform(df)
        preprocessor_obj=joblib.load('artifacts\data_preprocessed\Preprocess_model.joblib')
        df=preprocessor_obj.transform(df)
        print(df)
        model=PredictionPipeline()
        prediction=model.predict(df)
        print(prediction)
        
        out=''
        if prediction==1:
            out='Salary Greater than 50 ..'
        else:
            out='Salary lesser than 50 ..'

        # Render the prediction page (you can pass the DataFrame to the template if needed)
        return render_template('index.html', salary=out)
    else:
        return render_template('index.html')



if __name__ == "__main__":
    app.run(host="0.0.0.0", port = 8080)
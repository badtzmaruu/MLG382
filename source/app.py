import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import xgboost as xgb
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
from dash import Dash, html, dcc, Input, Output
from functools import lru_cache
import dash_bootstrap_components as dbc

#Loading models
@lru_cache(maxsize=1)
def get_DLmodel():
    print("Loading deep learning model...")
    return load_model(r'../artifacts/deep_learning_model.h5')

DL_model = get_DLmodel()

with open(r'../artifacts/features.pkl', 'rb') as f:
    features = pickle.load(f)

with open(r'../artifacts/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open(r'../artifacts/randomforest_model.pkl', 'rb') as f:
    randomforest_model = pickle.load(f)

with open(r'../artifacts/regression_model.pkl', 'rb') as f:
    regression_model = pickle.load(f)

with open(r'../artifacts/xgboost_model.pkl', 'rb') as f:
    xgboost_model = pickle.load(f)

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

# Sample input layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Student Grade Predictor", className="text-center mb-4"), width=12)
    ]),
    
    dbc.Row([
        dbc.Col(
            dbc.Card([
                dbc.CardBody([

                    dbc.Label("Age", className="text-center w-100"),
                    dbc.Input(id='age', type='number', value=18, className="mb-3"),

                    dbc.Label("Gender", className="text-center w-100"),
                    dcc.Dropdown(
                        id='gender',
                        options=[{'label': 'Male', 'value': 0}, {'label': 'Female', 'value': 1}],
                        value=0,
                        className="mb-3"
                    ),

                    dbc.Label("Study Time Weekly", className="text-center w-100"),
                    dbc.Input(id='study_time', type='number', value=15, className="mb-3"),

                    dbc.Label("Absences", className="text-center w-100"),
                    dbc.Input(id='absences', type='number', value=5, className="mb-3"),

                    dbc.Label("Ethnicity", className="text-center w-100"),
                    dcc.Dropdown(
                        id='ethnicity',
                        options=[
                            {'label': 'Caucasian', 'value': 0},
                            {'label': 'African American', 'value': 1},
                            {'label': 'Asian', 'value': 2},
                            {'label': 'Other', 'value': 3},
                        ],
                        value=0,
                        className="mb-3"
                    ),

                    dbc.Label("Parental Education", className="text-center w-100"),
                    dcc.Dropdown(
                        id='parental_education',
                        options=[
                            {'label': 'None', 'value': 0},
                            {'label': 'High School', 'value': 1},
                            {'label': 'Some College', 'value': 2},
                            {'label': 'Bachelors', 'value': 3},
                            {'label': 'Higher Study', 'value': 4}
                        ],
                        value=0,
                        className="mb-3"
                    ),

                    dbc.Label("Parental Support", className="text-center w-100"),
                    dcc.Dropdown(
                        id='parental_support',
                        options=[
                            {'label': 'None', 'value': 0},
                            {'label': 'Low', 'value': 1},
                            {'label': 'Moderate', 'value': 2},
                            {'label': 'High', 'value': 3},
                            {'label': 'Very High', 'value': 4}
                        ],
                        value=0,
                        className="mb-3"
                    ),

                    html.Div([
                        dbc.Label("Activities"),
                        dbc.Checklist(
                            options=[{'label': 'Tutoring', 'value': 1}],
                            value=[], id='tutoring', inline=True
                        ),
                        dbc.Checklist(
                            options=[{'label': 'Extracurricular', 'value': 1}],
                            value=[], id='extracurricular', inline=True
                        ),
                        dbc.Checklist(
                            options=[{'label': 'Sports', 'value': 1}],
                            value=[], id='sports', inline=True
                        ),
                        dbc.Checklist(
                            options=[{'label': 'Music', 'value': 1}],
                            value=[], id='music', inline=True
                        ),
                        dbc.Checklist(
                            options=[{'label': 'Volunteering', 'value': 1}],
                            value=[], id='volunteering', inline=True
                        ),
                    ], className="mb-4"),

                    dbc.Button("Predict", id='predict_button', color="primary", className="mb-3 w-100"),

                    html.Div(id='prediction-output', className="mt-3 text-center")

                ])
            ], className="p-4 shadow-sm border rounded bg-white"),  
            width=4,
            className="mx-auto"
        )
    ])
], fluid=True)

@app.callback(
    Output('prediction-output', 'children'),
    [Input('predict_button', 'n_clicks'),
     Input('age', 'value'),
     Input('gender', 'value'),
     Input('study_time', 'value'),
     Input('absences', 'value'),
     Input('tutoring', 'value'),
     Input('extracurricular', 'value'),
     Input('sports', 'value'),
     Input('music', 'value'),
     Input('volunteering', 'value'),
     Input('ethnicity', 'value'),
     Input('parental_education', 'value'),
     Input('parental_support', 'value')]
)

def predict_grade(n_clicks, age, gender,study_time, absences, tutoring,extracurricular, sports, music, volunteering, ethnicity, parental_education, parental_support):
    if (n_clicks or 0) > 0 and None not in (age, gender, study_time, absences, ethnicity, parental_education, parental_support):
         input_data = {
            'Age': [age],
            'Gender': [gender],
            'Ethnicity': [ethnicity],
            'ParentalEducation': [parental_education],
            'StudyTimeWeekly': [study_time],
            'Absences': [absences],
            'Tutoring': [1 if tutoring and 1 in tutoring else 0],
            'ParentalSupport': [parental_support],
            'Extracurricular': [1 if extracurricular and 1 in extracurricular else 0],
            'Sports': [1 if sports and  1 in sports else 0],
            'Music': [1 if music and 1 in music else 0],
            'Volunteering': [1 if volunteering and 1 in volunteering else 0]
        }
         
         input_df = pd.DataFrame(input_data)


         input_df = input_df[features]
         
         #Scaling input values
         num_features = ['Age', 'StudyTimeWeekly', 'Absences']
         input_df[num_features] = scaler.transform(input_df[num_features])
         
         grade_mapping = {
            0: "A",
            1: "B",
            2: "C",
            3: "D",
            4: "F"
         }
         
         #Predicting with models
         dl_prediction = DL_model.predict(input_df)
         rf_prediction = randomforest_model.predict(input_df)
         logreg_prediction = regression_model.predict(input_df)
         xgboost_prediction = xgboost_model.predict(input_df)
         print("Prediction done")

         if len(dl_prediction.shape) == 2 and dl_prediction.shape[1] > 1:
             class_prediction = np.argmax(dl_prediction)
             probability = np.max(dl_prediction)
         else:
             class_prediction = int(round(float(dl_prediction[0][0])))
             probability = float(dl_prediction[0][0]) if class_prediction == 1 else 1 - float(dl_prediction[0][0])

         probability_percent = probability * 100
         
        

         return dbc.Card([
            dbc.CardHeader("Model Predictions", className="bg-success text-white text-center"),
            dbc.CardBody([
                dbc.Table([
                    html.Thead(html.Tr([
                        html.Th("Model", className="text-center"),
                        html.Th("Prediction", className="text-center")
                    ])),
                html.Tbody([
                    html.Tr([html.Td("Deep Learning", className="text-center"), html.Td(f"{grade_mapping.get(class_prediction)} (Confidence: {probability_percent:.2f}%)", className="text-center")]),
                    html.Tr([html.Td("Random Forest", className="text-center"), html.Td(f"{grade_mapping.get(int(rf_prediction[0]))}", className="text-center")]),
                    html.Tr([html.Td("Logistic Regression", className="text-center"), html.Td(f"{grade_mapping.get(int(logreg_prediction[0]))}", className="text-center")]),
                    html.Tr([html.Td("XGBoost", className="text-center"), html.Td(f"{grade_mapping.get(int(xgboost_prediction[0]))}", className="text-center")])
                ])
                ], bordered=True, striped=True, hover=True, responsive=True)
            ])
        ], className="mt-4 shadow-sm")

    return "Please fill in all fields."
        
if __name__ == "__main__":
    print("Launching Dash app...")
    try:
        app.run(debug=True, host='127.0.0.1', port=8050)
    except Exception as e:
        print("Failed to start server:", e)
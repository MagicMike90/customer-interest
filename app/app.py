from flask import Flask, jsonify ,request
import json
from sklearn.externals import joblib
import pandas as pd
import numpy as np
from labelEncoder import MultiColumnLabelEncoder

app = Flask(__name__)

def transform_cols(df):
    df.columns = map(str.lower, df.columns)
    df.columns = df.columns.str.replace('_', ' ')
    return df

def cleanFeatures(data) :
    for col in model_columns: 
        if col not in data.columns:
            data[col] = 0
            
    for col in model_columns: 
        if col not in data.columns:
            data[col] = 0
    return data

def transform(df): 
    df = transform_cols(df)
    
    if 'loan amount' in df.columns: 
        df['loan amount'] = df['loan amount'].astype('float')
        
    if 'enquired' in df.columns: 
        df['enquired'] = pd.DatetimeIndex(df['enquired'])
        df['month'] = df['enquired'].dt.month
        df['day'] = df['enquired'].dt.day
        df['hour'] = df['enquired'].dt.hour
        df['weekday'] = df['enquired'].dt.dayofweek
    
    if 'post code' in df.columns: 
        df['post code'] = df['post code'].astype('int')
    
    if 'enquired'in df.columns:
        df.drop(['enquired'], axis = 1, inplace = True)
    
    return df.replace(encoder)

def classify(data):
    # label = {0: 'accept', 1: 'reject'}
    X = transform(data)
    y = clf.predict(X)[0]
    proba = np.max(clf.predict_proba(X))
    return y, proba
  
@app.route('/predict', methods=['POST'])
def predict():
  if request.is_json:
    data = pd.DataFrame(request.get_json(),index=[0])

    y, proba = classify(data)
    return jsonify({'prediction': y,'proba': proba})


if __name__ == '__main__':
     clf = joblib.load('models/classifier.pkl')
     model_columns = joblib.load('models/model_columns.pkl')
     encoder = joblib.load('models/encoder.pkl')
     app.run(debug=True)

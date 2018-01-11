from flask import Flask, jsonify ,request
import json
from sklearn.externals import joblib
import pandas as pd
import numpy as np
from labelEncoder import MultiColumnLabelEncoder

app = Flask(__name__)

def changeDateType(df): 
  if 'enquired' in df.columns:
    df['enquired'] = pd.DatetimeIndex(df['enquired'])
  df['loan amount'] = df['loan amount'].astype(int)
  df['post code'] = df['post code'].astype(int)
  return df

def getDetailDate(df):
    # remove Year feature since it is not important (show below random forest)
    # data_set['Year'] = data_set['Enquired'].dt.year
    if 'enquired' in df.columns:
      df['month'] = df['enquired'].dt.month
      df['day'] = df['enquired'].dt.day
      df['hour'] = df['enquired'].dt.hour
      df['weekday'] = df['enquired'].dt.weekday_name
      df = df.loc[:,df.columns != 'enquired']
    return df

def transform(data):
  data.columns = map(str.lower, data.columns)
  data.columns = data.columns.str.replace('_', ' ')
  changeDateType(data)
  data = getDetailDate(data)
  return data

def classify(data):
    X = transform(data)
    print(X)
    y = clf.predict(X)[0]
    proba = np.max(clf.predict_proba(X))
    return y, proba
  
@app.route('/predict', methods=['POST'])
def predict():
  if request.is_json:
    data = pd.DataFrame(request.get_json(),index=[0])
    data = transform(data)
    y, proba = classify(data)

    print(y)
    print(proba)

    return jsonify({'prediction': 'b','proba': "a"})
  # try:
  #   if request.is_json:
  #     # print(request.get_json())
  #     data = pd.DataFrame(request.get_json(),index=[0])
  #     # data = json.dumps(request.get_json())
  #     y, proba = classify(data)
  #     return jsonify({'prediction': list(y),'proba': list(proba)})
  # except Exception as inst:
  #   return jsonify({'error': inst})


if __name__ == '__main__':
     clf = joblib.load('models/lrpipeline.pkl')
     model_columns = joblib.load('models/model_columns.pkl')
     app.run(debug=True)

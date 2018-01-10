from flask import Flask, jsonify ,request
import json
from sklearn.externals import joblib
import pandas as pd
import numpy as np


app = Flask(__name__)

def changeDateType(df): 
    df['enquired'] = pd.DatetimeIndex(df['enquired'])
    df['loan amount'] = df['loan amount'].astype(int)
    df['post code'] = df['post code'].astype(int)
    return df

def getDetailDate(df):
    # remove Year feature since it is not important (show below random forest)
    # data_set['Year'] = data_set['Enquired'].dt.year
    df['month'] = df['enquired'].dt.month
    df['day'] = df['enquired'].dt.day
    df['hour'] = df['enquired'].dt.hour
    df['weekday'] = df['enquired'].dt.weekday_name

    # if 'enquired' in df.columns:
    #   df = df.loc[:,df.columns != 'enquired']
    return df

def transform(data):
  data.columns = map(str.lower, data.columns)
  data = data.applymap(lambda x: x if not '_' in str(x) else x.replace('_',''))
  changeDateType(data)
  getDetailDate(data)
  return data

def classify(data):
    label = {0: 'Accpeted', 1: 'Rejected'}
    X = transform(data)

    print("X {}".format(X))

    y = clf.predict(X)

    print("y {}".format(y))

    # labels = list(map(lambda x: label[x], y))
    # proba = list(map(lambda x: np.max(x), clf.predict_proba(X)))

    return labels, proba
  
@app.route('/predict', methods=['POST'])
def predict():
  if request.is_json:
    data = pd.DataFrame(request.get_json(),index=[0])
    data = transform(data)
    y, proba = classify(data)
    return jsonify('worked!')
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

from flask import Flask, jsonify ,request
import json
from sklearn.externals import joblib
import pandas as pd
import numpy as np


app = Flask(__name__)

def transform(data):
  test = pd.read_json(data, orient='records')
  print(test)
  query = pd.get_dummies(test)
  print(query)
  for col in model_columns:
    if col not in query.columns:
      query[col] = 0

  query = query.fillna(0)

  return query[model_columns]

def classify(data):
    label = {1: 'Accpeted', 3: 'Rejected'}
    X = transform(data)
    y = clf.predict(X)
    labels = list(map(lambda x: label[x], y))
    proba = list(map(lambda x: np.max(x), clf.predict_proba(X)))

    return labels, proba
  
@app.route('/predict', methods=['POST'])
def predict():
  try:
    if request.is_json:
      data = json.dumps(request.get_json())

      y, proba = classify(data)
      return jsonify({'prediction': list(y),'proba': list(proba)})
  except Exception as inst:
    return jsonify({'error': inst})


if __name__ == '__main__':
     clf = joblib.load('models/lrpipeline.pkl')
     model_columns = joblib.load('models/model_columns.pkl')
     app.run(port=8080)

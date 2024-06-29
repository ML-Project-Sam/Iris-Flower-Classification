from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained models
log_reg_model = joblib.load('log_reg_model.pkl')
knn_model = joblib.load('knn_model.pkl')

@app.route('/')
def home():
    return "Welcome to the Iris Classification API!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = pd.DataFrame([data])
    log_reg_prediction = log_reg_model.predict(features)[0]
    knn_prediction = knn_model.predict(features)[0]
    return jsonify({
        'logistic_regression_prediction': log_reg_prediction,
        'knn_prediction': knn_prediction
    })

if __name__ == '__main__':
    app.run(debug=True)

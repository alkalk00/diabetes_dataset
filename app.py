from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import os

app = Flask(__name__, static_folder='.', static_url_path='', template_folder='templates')

# Load model
with open('diabetes_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = np.array([[data['pregnancy'], data['glucose'], data['bloodPressure'],
                          data['skinThickness'], data['insulin'], data['bmi'],
                          data['diabetesPedigree'], data['age']]])
    prediction = model.predict(features)
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)

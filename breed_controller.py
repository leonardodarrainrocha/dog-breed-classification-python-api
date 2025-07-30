import breed_predictor
from flask import Flask, request

app = Flask(__name__)

@app.route('/api/predict', methods=['POST'])
def predict():
    if not request.files:
        return "No image part", 400

    file = request.files['dog']
    if file.filename == '':
        return "No selected file", 400

    aux = breed_predictor.get_prediction(file)
    return aux

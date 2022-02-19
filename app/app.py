from flask import Flask, request
from requests import request
from model.models import load_model

app = Flask(__name__)

@app.route("/predict", methods=["GET","POST"])
def predict():
    # Load the Input
    data = request.headers['x']

    # Load the model
    model = load_model()

    # Make predictions on input data
    model.predict(data)


app.run(host='0.0.0.0', port = 80)
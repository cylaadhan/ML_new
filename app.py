import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# Flask App
app = Flask(__name__)

# Load model
model = pickle.load(open("linear_regression_model.pkl", "rb"))

# Home Page
@app.route("/")
def home():
    return render_template("index.html")

# Prediction Route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        float_features = [float(x) for x in request.form.values()]
        features = [np.array(float_features)]
        prediction = model.predict(features)

        return render_template(
            "index.html",
            prediction_text="The flower species is {}".format(prediction[0])
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction_text="Error: {}".format(str(e))
        )

# Run Local
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

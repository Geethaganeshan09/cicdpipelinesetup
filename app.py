from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load("my_model.joblib")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    features = data.get("features")

    if features is None:
        return jsonify({"error": "Missing 'features' in request body"}), 400

    features_array = np.array(features).reshape(1, -1)
    prediction = model.predict(features_array)

    return jsonify({"prediction": prediction.tolist()})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

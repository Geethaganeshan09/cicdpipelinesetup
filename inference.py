import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify

# Load the trained model
model = joblib.load("my_model.joblib")

# Initialize Flask app
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Parse incoming JSON request
        data = request.get_json()
        
        # Convert input data into a DataFrame
        features = pd.DataFrame([data])

        # Ensure correct feature order
        feature_columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
        features = features[feature_columns]

        # Make prediction
        prediction = model.predict(features)
        
        # Return JSON response
        return jsonify({"prediction": prediction.tolist()})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Run the application
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

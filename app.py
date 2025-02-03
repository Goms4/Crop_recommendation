from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model and label encoder
model = joblib.load("crop_prediction_model.pkl")  # Your trained model file
label_encoder = joblib.load("label_encoder.pkl")  # Encoder to decode class labels

@app.route('/')
def home():
    return render_template("index.html")  # Serve your HTML file

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract input values from form
        features = [float(request.form[key]) for key in ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]]
        features = np.array(features).reshape(1, -1)

        # Make prediction
        prediction = model.predict(features)
        predicted_label = label_encoder.inverse_transform(prediction)[0]

        return jsonify({"predicted_label": predicted_label})
    
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
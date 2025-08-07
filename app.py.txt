from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

app = Flask(__name__)

# Load the trained LSTM Autoencoder model
model = load_model("lstm_autoencoder.h5")

# Set your anomaly threshold (match your training)
THRESHOLD = 0.015

@app.route('/')
def home():
    return "✅ LSTM Autoencoder API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json.get("sequence")
        
        if not data:
            return jsonify({"error": "Missing 'sequence' in request"}), 400

        # Convert input to padded array
        padded = pad_sequences([data], padding='post')
        prediction = model.predict(padded)

        # Calculate reconstruction error
        X_flat = np.array(data).reshape(1, -1)
        X_pred_flat = prediction.reshape(1, -1)
        error = np.mean(np.square(X_flat - X_pred_flat))

        is_anomaly = int(error > THRESHOLD)

        return jsonify({
            "reconstruction_error": float(error),
            "anomaly": is_anomaly
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

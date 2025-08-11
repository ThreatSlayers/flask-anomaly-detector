from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow import not_equal
import tensorflow as tf
import numpy as np
import os

app = Flask(__name__)

# Load model with custom objects
try:
    from keras.utils import custom_object_scope
except ImportError:
    from tensorflow.keras.utils import custom_object_scope

try:
    with custom_object_scope({'NotEqual': not_equal}):
        model = load_model("lstm_autoencoder.h5")
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    model = None

# Set your anomaly threshold
THRESHOLD = 0.015

@app.route('/')
def home():
    if model:
        return "✅ LSTM Autoencoder API is running!"
    else:
        return "❌ Model failed to load. Check server logs."

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded on server"}), 500

    try:
        data = request.json.get("sequence")

        if not data:
            return jsonify({"error": "Missing 'sequence' in request"}), 400

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
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)

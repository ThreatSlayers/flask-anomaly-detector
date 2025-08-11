from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import numpy as np

app = Flask(__name__)

# Add all your custom objects here:
custom_objects = {
    "NotEqual": tf.not_equal,
    # Example: Add your custom layers or functions if any, e.g.
    # "MyCustomLayer": MyCustomLayerClass,
    # "custom_activation": custom_activation_function,
}

# Load the trained LSTM Autoencoder model
try:
    model = load_model("lstm_autoencoder.h5", custom_objects=custom_objects)
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    model = None

# Set your anomaly threshold (match your training)
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

        # Pad sequence if needed (adjust maxlen if you know your model input length)
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
    import os
    port = int(os.environ.get("PORT", 5000))  # Render sets PORT dynamically
    app.run(host="0.0.0.0", port=port, debug=False)

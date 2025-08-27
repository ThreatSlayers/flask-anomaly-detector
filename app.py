from flask import Flask, request, jsonify, render_template_string
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import os
import re

app = Flask(__name__)

# --- Model Loading ---
try:
    model = load_model("threatslayers_lstm_autoencoder.keras", compile=False)
    print("✅ Model loaded successfully from threatslayers_lstm_autoencoder.keras")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    print("Ensure 'threatslayers_lstm_autoencoder.keras' is in the same directory as app.py")
    model = None

THRESHOLD = 0.015

# --- HTML Template for the User Interface ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Log Anomaly Detector</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        body { font-family: 'Inter', sans-serif; }
        .loader {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #3498db;
            border-radius: 50%;
            width: 24px;
            height: 24px;
            animation: spin 1s linear infinite;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        /* Style the file input button */
        input[type="file"] {
            display: none;
        }
        .custom-file-upload {
            border: 1px solid #ccc;
            display: inline-block;
            padding: 8px 12px;
            cursor: pointer;
            background-color: #f9f9f9;
            border-radius: 8px;
            border-color: #d1d5db;
        }
    </style>
</head>
<body class="bg-gray-100 flex items-center justify-center min-h-screen p-4">
    <div class="bg-white p-8 rounded-xl shadow-lg w-full max-w-3xl">
        <h1 class="text-3xl font-bold text-gray-800 mb-2">Log Anomaly Detector</h1>
        <p class="text-gray-600 mb-6">Upload a log file, paste raw text, or enter a manual sequence to check for anomalies.</p>

        <form id="anomaly-form">
            <div class="mb-4">
                 <label for="file-upload" class="custom-file-upload font-medium text-gray-700 hover:bg-gray-50">
                    📂 Upload Log File
                </label>
                <input id="file-upload" type="file" accept=".txt,.log"/>
                <span id="file-name" class="ml-3 text-sm text-gray-600">No file selected</span>
            </div>

            <div class="mb-4">
                <label for="log_data" class="block text-sm font-medium text-gray-700 mb-1">Or Paste Raw Log Text Here</label>
                <textarea id="log_data" name="log_data" rows="8" class="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500" placeholder="Paste content from a file like System_log.txt..."></textarea>
            </div>
            <div class="text-center my-4 text-gray-500 font-semibold">OR</div>
            <div class="mb-6">
                <label for="sequence" class="block text-sm font-medium text-gray-700 mb-1">Manual Sequence Input</label>
                <input type="text" id="sequence" name="sequence" class="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500" placeholder="e.g., 10, 25, 10, 3, 18, 5">
            </div>
            <button type="submit" class="w-full bg-blue-600 text-white font-bold py-3 px-4 rounded-lg hover:bg-blue-700 transition duration-300 flex items-center justify-center">
                <span id="button-text">Check for Anomaly</span>
                <div id="loader" class="loader hidden ml-3"></div>
            </button>
        </form>

        <div id="result-container" class="mt-8 p-6 border-t border-gray-200 hidden">
            <h2 class="text-xl font-semibold text-gray-800 mb-4">Analysis Result</h2>
            <div id="result-content"></div>
        </div>
    </div>

    <script>
        // Make the Python THRESHOLD variable available to JavaScript
        const THRESHOLD = {{ THRESHOLD }};

        // --- File Upload Handler ---
        document.getElementById('file-upload').addEventListener('change', function(event) {
            const file = event.target.files[0];
            if (file) {
                document.getElementById('file-name').textContent = file.name;
                const reader = new FileReader();
                reader.onload = function(e) {
                    // When the file is read, put its content into the textarea
                    document.getElementById('log_data').value = e.target.result;
                };
                reader.readAsText(file);
            }
        });


        // --- Form Submission Handler ---
        document.getElementById('anomaly-form').addEventListener('submit', async function(event) {
            event.preventDefault();

            const logData = document.getElementById('log_data').value;
            const sequenceInput = document.getElementById('sequence').value;
            const resultContainer = document.getElementById('result-container');
            const resultContent = document.getElementById('result-content');
            const loader = document.getElementById('loader');
            const buttonText = document.getElementById('button-text');

            // Show loader and disable button
            loader.classList.remove('hidden');
            buttonText.innerText = 'Analyzing...';
            event.target.querySelector('button').disabled = true;
            
            resultContainer.classList.add('hidden');
            resultContent.innerHTML = '';

            let payload = {};
            // Prioritize raw log data (from file upload or paste) if it exists
            if (logData.trim().length > 0) {
                payload = { log_text: logData };
            } else if (sequenceInput.trim().length > 0) {
                const sequenceArray = sequenceInput.split(',').map(item => parseInt(item.trim())).filter(item => !isNaN(item));
                if (sequenceArray.length === 0) {
                    showError('Manual sequence is invalid. Please enter numbers separated by commas.');
                    return;
                }
                payload = { sequence: sequenceArray };
            } else {
                showError('Please upload a file, paste log text, or enter a manual sequence.');
                return;
            }

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });

                if (!response.ok) {
                    const errorData = await response.json();
                    throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
                }

                const data = await response.json();
                
                let resultHTML = '';
                if (data.is_anomaly) {
                    resultHTML += '<p class="text-2xl font-bold text-red-600">Status: Anomaly Detected!</p>';
                } else {
                    resultHTML += '<p class="text-2xl font-bold text-green-600">Status: Normal</p>';
                }
                resultHTML += `<p class="text-gray-700 mt-2">Reconstruction Error: <span class="font-mono bg-gray-200 px-2 py-1 rounded">${data.reconstruction_error.toFixed(5)}</span></p>`;
                resultHTML += `<p class="text-gray-500 text-sm mt-1">Threshold: <span class="font-mono">${THRESHOLD.toFixed(5)}</span></p>`;
                if (data.extracted_sequence) {
                    resultHTML += `<p class="text-gray-600 mt-3">Extracted Event IDs: <span class="font-mono text-sm">${data.extracted_sequence.join(', ')}</span></p>`;
                }


                resultContent.innerHTML = resultHTML;
                resultContainer.classList.remove('hidden');

            } catch (error) {
                showError(`An error occurred: ${error.message}`);
            } finally {
                resetButton();
            }
        });

        function showError(message) {
            const resultContainer = document.getElementById('result-container');
            const resultContent = document.getElementById('result-content');
            resultContent.innerHTML = `<p class="text-red-600 font-semibold">${message}</p>`;
            resultContainer.classList.remove('hidden');
            resetButton();
        }

        function resetButton() {
            document.getElementById('loader').classList.add('hidden');
            document.getElementById('button-text').innerText = 'Check for Anomaly';
            document.querySelector('#anomaly-form button').disabled = false;
        }
    </script>
</body>
</html>
"""

def parse_event_ids_from_log(log_text):
    """
    Parses a block of Windows Event Log text to extract Event IDs.
    It looks for a number that follows the 'Source' column.
    Example line: 'Information   8/13/2025 1:25:51 PM    Netwtw14    7021    None'
    The regex will find '7021'.
    """
    # This regex looks for lines containing a date and time, followed by a source,
    # and captures the number (Event ID) that comes after the source name.
    # It's designed to be robust against variations in spacing.
    regex = r"\d{1,2}/\d{1,2}/\d{4}\s+\d{1,2}:\d{2}:\d{2}\s+(?:AM|PM)\s+.*?(\d+)\s"
    
    event_ids = re.findall(regex, log_text)
    
    # Convert found strings to integers
    return [int(eid) for eid in event_ids]


@app.route('/')
def home():
    if model is None:
        return "<h1>❌ Model failed to load.</h1><p>Check server logs for details and ensure 'lstm_autoencoder.keras' is present.</p>", 500
    return render_template_string(HTML_TEMPLATE, THRESHOLD=THRESHOLD)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model is not loaded on the server"}), 500

    try:
        json_data = request.get_json()
        sequence = []
        extracted_sequence_for_response = None

        if 'log_text' in json_data:
            log_text = json_data['log_text']
            sequence = parse_event_ids_from_log(log_text)
            extracted_sequence_for_response = sequence # Save for response
            if not sequence:
                return jsonify({"error": "Could not find any valid Event IDs in the provided log text."}), 400
        elif 'sequence' in json_data:
            sequence = json_data['sequence']
            if not isinstance(sequence, list):
                 return jsonify({"error": "Invalid 'sequence' format, must be a list."}), 400
        else:
            return jsonify({"error": "Request must contain either 'log_text' or 'sequence'."}), 400

        if not sequence:
             return jsonify({"error": "No sequence to process."}), 400

        padded = pad_sequences([sequence], padding='post', maxlen=6)
        prediction = model.predict(padded)
        error = np.mean(np.square(padded - prediction))
        is_anomaly = bool(error > THRESHOLD)

        response_data = {
            "reconstruction_error": float(error),
            "is_anomaly": is_anomaly
        }
        if extracted_sequence_for_response:
            response_data["extracted_sequence"] = extracted_sequence_for_response

        return jsonify(response_data)

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({"error": f"An internal error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)


* This repo was set up as an example for student to understand about deploying AI model in local machines then hosting it for others to use. 

# AI Log Anomaly Detector

A full-stack web application that uses a pre-trained LSTM Autoencoder model to detect anomalies in sequences of log event IDs.

## Overview

This project provides a simple and interactive web interface to an AI model trained for anomaly detection. Users can upload a log file, paste raw log text, or input a sequence of numerical log IDs directly. The application will automatically parse the Event IDs and use a Keras/TensorFlow model to determine if the sequence is normal or anomalous based on its reconstruction error.

The entire application, including the Python backend (Flask) and the HTML/CSS/JS frontend, is contained within a single `app.py` file for simplicity and ease of deployment.

## Features

* **Interactive Web UI:** A clean user interface for submitting log data.
* **Flexible Input:** Supports file uploads, raw text pasting, and manual sequence entry.
* **Automatic Log Parsing:** Intelligently extracts Event IDs from raw log text.
* **Real-time Analysis:** Get instant feedback on whether a sequence is normal or an anomaly.
* **Detailed Results:** Shows the model's reconstruction error and the threshold used for classification.
* **Self-Contained:** The entire frontend and backend are in one Python file.
* **Pre-trained Model:** Comes with a `lstm_autoencoder.keras` model ready for use.

## Prerequisites

Before you begin, ensure you have the following installed:
* Python 3.7+
* `pip` (Python package installer)
* Visual Studio Code
* Go to this drive https://drive.google.com/drive/folders/19o20TH4Nol-QlZVSoVQqqDrloz4e-Rhw   and download `lstm_autoencoder.keras` model to save in your root dir

## Setup & Installation

1.  **Clone the repository or download the project files.**

2.  **Navigate to the project directory:**
    ```bash
    cd path/to/your/project
    ```

3.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

4.  **Install the required Python packages from `requirements.txt`:**
    ```bash
    pip install -r requirements.txt
    ```

5.  **Ensure the model is present:** Make sure the pre-trained model file `lstm_autoencoder.keras` is in the same directory as `app.py`.

## Running the Application

1.  **Start the Flask server from your terminal:**
    ```bash
    python app.py
    ```

2.  You should see output indicating the model has loaded and the server is running, like this:
    ```
    ✅ Model loaded successfully from lstm_autoencoder.keras
     * Running on http://127.0.0.1:5000
    ```

## How to Use the Application

1.  **Open your web browser** and navigate to:
    http://127.0.0.1:5000

2.  You will see the **Log Anomaly Detector** user interface. You have three ways to input data:
    * **Upload a File:** Click the "Upload Log File" button and select a `.txt` or `.log` file.
    * **Paste Raw Text:** Copy and paste content from a log file into the text area.
    * **Enter Manually:** Type a comma-separated sequence of numbers into the manual input box.

3.  Click the **"Check for Anomaly"** button. The result will be displayed on the page.

## Sharing for Testing (Port Forwarding with VS Code)

To allow others to test your application over the internet, you can use the built-in port forwarding feature in VS Code.

1.  **Run the Application:** Make sure your app is running via `python app.py` in the VS Code terminal.

2.  **Open the Ports View:**
    * Go to the **"Ports"** tab in the bottom panel of VS Code (next to Terminal, Debug Console, etc.).
    * If you don't see it, open the Command Palette (`Ctrl+Shift+P` or `Cmd+Shift+P`) and type `Ports: Focus on Ports View`.

3.  **Forward the Port:**
    * In the Ports view, click the **"Forward a Port"** button.
    * When prompted, enter the port number `5000` and press Enter.

4.  **Authenticate (First Time Only):**
    * VS Code will ask you to sign in with a **GitHub** or **Microsoft** account. This is required to create the secure tunnel. Follow the login prompts.

5.  **Set Port to Public:**
    * A new entry for port 5000 will appear in the list. By default, it is "Private".
    * Right-click on the forwarded port, select **"Port Visibility"**, and choose **"Public"**.

6.  **Share the URL:**
    * The "Forwarded Address" column will now show a public URL (e.g., `https://random-words-5000.usw2.devtunnels.ms/`).
    * Click the "Copy" icon next to the URL and share it with your testers. They can now access your running application from their own browsers.

## (Optional) Retraining the Model

If you wish to retrain the model with new data:

1.  **Place your dataset** in the `dataset/Dataset-ThreatSlayers/` directory. The training script expects `Event_traces.csv` and `anomaly_label.csv`.

2.  **Run the training script:**
    ```bash
    python modelTrainingCode/ai_train_lstm_encoder.py
    ```
    *(Note: You may need to escape the parentheses in some shells, as shown above)*

3.  This will generate a new `lstm_autoencoder.keras` file in the project's root directory, which the application will use the next time it is started.

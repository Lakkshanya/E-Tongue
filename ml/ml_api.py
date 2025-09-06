from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import numpy as np
import joblib
import logging
import os

# === Optional: CORS for local dev ===
try:
    from flask_cors import CORS
    CORS_ENABLED = True
except Exception:
    CORS_ENABLED = False

# === Setup Logging ===
os.makedirs('logs', exist_ok=True)
logging.basicConfig(filename='logs/api_errors.log', level=logging.ERROR,
                    format='%(asctime)s %(levelname)s %(message)s')

# === Define Model Architecture (UNCHANGED) ===
class WaterQualityNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(9, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.model(x)

# === Load Model and Scaler (UNCHANGED) ===
try:
    model = WaterQualityNN()
    model.load_state_dict(torch.load('models/model.pth', map_location='cpu'))
    model.eval()
    scaler = joblib.load('models/scaler.pkl')
except Exception as e:
    logging.error(f"Model loading error: {str(e)}")
    raise RuntimeError("Failed to load model or scaler")

# === Initialize Flask App ===
app = Flask(__name__)
if CORS_ENABLED:
    CORS(app)

FEATURE_KEYS = [
    'Hardness', 'Solids_TDS', 'Sulphate', 'Chloramine', 'Conductivity',
    'Organic_Carbon', 'Trihalomethane', 'Turbidity', 'pH'
]

@app.get("/health")
def health():
    return jsonify({"ok": True, "model_loaded": True, "expected_keys": FEATURE_KEYS})

# === Preprocess Input (UNCHANGED) ===
def preprocess_input(data):
    try:
        if not all(k in data for k in FEATURE_KEYS):
            missing = [k for k in FEATURE_KEYS if k not in data]
            raise ValueError(f"Missing input fields: {missing}")

        input_vals = [float(data[k]) for k in FEATURE_KEYS]
        input_scaled = scaler.transform([input_vals])
        input_tensor = torch.tensor(input_scaled, dtype=torch.float32)
        return input_tensor
    except Exception as e:
        logging.error(f"Preprocessing error: {str(e)}")
        raise

# === Prediction Route (UNCHANGED LOGIC) ===
@app.post('/predict')
def predict():
    try:
        data = request.get_json(force=True)
        input_tensor = preprocess_input(data)

        with torch.no_grad():
            output = model(input_tensor).item()
            confidence = round(output, 3)
            adulterated = output < 0.5
            safe_to_use = output >= 0.5
            borderline = 0.4 <= output <= 0.6

            return jsonify({
                "confidence": confidence,
                "adulterated": adulterated,
                "safe_to_use": safe_to_use,
                "borderline": borderline
            })
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        return jsonify({"error": "Prediction failed", "details": str(e)}), 500

if __name__ == '__main__':
    # Run on a fixed port so Node can call it easily
    app.run(host='0.0.0.0', port=5001, debug=True)

from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import joblib
import os
import logging

# Optional CORS
try:
    from flask_cors import CORS
    CORS_ENABLED = True
except:
    CORS_ENABLED = False

os.makedirs('logs', exist_ok=True)
logging.basicConfig(filename='logs/api_errors.log', level=logging.ERROR,
                    format='%(asctime)s %(levelname)s %(message)s')

# === Model definition ===
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

# === Load model and preprocessing ===
model = WaterQualityNN()
model.load_state_dict(torch.load('models/model.pth', map_location='cpu'))
model.eval()

scaler = joblib.load('models/scaler.pkl')
feature_medians = joblib.load('models/feature_medians.pkl')

FEATURE_KEYS = [
    'Hardness', 'Solids_TDS', 'Sulphate', 'Chloramine', 'Conductivity',
    'Organic_Carbon', 'Trihalomethane', 'Turbidity', 'pH'
]

app = Flask(__name__)
if CORS_ENABLED:
    CORS(app)

@app.get("/health")
def health():
    return jsonify({"ok": True, "model_loaded": True, "expected_keys": FEATURE_KEYS})

# === Preprocessing ===
def preprocess_input(data):
    vals = []
    for k in FEATURE_KEYS:
        v = data.get(k)
        if v is None or v == 0 or str(v).strip() == "":
            vals.append(float(feature_medians[k]))
        else:
            vals.append(float(v))
    input_scaled = scaler.transform([vals])
    input_tensor = torch.tensor(input_scaled, dtype=torch.float32)
    return input_tensor, vals

# === Prediction API ===
@app.post("/predict")
def predict():
    try:
        data = request.get_json(force=True)
        input_tensor, input_vals = preprocess_input(data)

        with torch.no_grad():
            output = model(input_tensor).item()

        confidence = round(output, 3)
        adulterated = output < 0.5
        safe_to_use = output >= 0.5
        borderline = 0.45 <= output <= 0.55

        # Domain rule: TDS override
        tds_value = float(data.get("Solids_TDS", feature_medians["Solids_TDS"]))
        if tds_value > 500 or tds_value < 50:
            adulterated = True
            safe_to_use = False

        return jsonify({
            "confidence": confidence,
            "adulterated": adulterated,
            "safe_to_use": safe_to_use,
            "borderline": borderline,
            "inputs_used": dict(zip(FEATURE_KEYS, input_vals))
        })
    except Exception as e:
        logging.error(str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)

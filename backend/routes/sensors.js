// backend/routes/sensors.js
const express = require('express');
const axios = require('axios');
const SensorData = require('../models/sensorSchema');
const Prediction = require('../models/resultSchema');

const router = express.Router();

const FEATURE_KEYS = [
  'Hardness','Solids_TDS','Sulphate','Chloramine','Conductivity',
  'Organic_Carbon','Trihalomethane','Turbidity','pH'
];

// Convert body to ML payload, using 0 for any missing/empty values
function toMlPayload(body) {
  const payload = {};
  FEATURE_KEYS.forEach(k => {
    // if missing, null, empty string -> use 0
    let raw = body[k];
    if (raw === undefined || raw === null || raw === '') {
      payload[k] = 0;
    } else {
      const n = Number(raw);
      if (Number.isNaN(n)) {
        const err = new Error(`Invalid number for ${k}`);
        err.status = 400;
        throw err;
      }
      payload[k] = n;
    }
  });
  return payload;
}

// POST /api/sensors
router.post('/', async (req, res, next) => {
  try {
    // Build payload (zeros for missing)
    const features = toMlPayload(req.body);

    // Save sensor doc — features object will include zeros where missing
    const sensorDoc = await SensorData.create({
      features,
      temperature: req.body.temperature ?? undefined,
      deviceId: req.body.deviceId ?? undefined,
      meta: req.body.meta ?? undefined,
      source: req.body.source ?? 'arduino'
    });

    // Call Flask ML service
    const { data: ml } = await axios.post(`${process.env.FLASK_URL}/predict`, features, {
      timeout: 10_000,
      headers: { 'Content-Type': 'application/json' }
    });

    if (ml.error) {
      const err = new Error(`ML service error: ${ml.details || ml.error}`);
      err.status = 502;
      throw err;
    }

    // Save prediction
    const resultDoc = await Prediction.create({
      sensorRef: sensorDoc._id,
      adulterated: !!ml.adulterated,
      safe_to_use: !!ml.safe_to_use,
      borderline: !!ml.borderline,
      confidence: Number(ml.confidence),
      raw: ml
    });

    res.status(201).json({
      sensor: sensorDoc,
      prediction: resultDoc
    });
  } catch (err) {
    next(err);
  }
});

// GET endpoints unchanged...
// (keep the rest of file as you had it)
module.exports = router;

const mongoose = require('mongoose');

const resultSchema = new mongoose.Schema({
  sensorRef: { type: mongoose.Schema.Types.ObjectId, ref: 'SensorData', required: true },
  adulterated: { type: Boolean, required: true },
  safe_to_use: { type: Boolean, required: true },
  borderline: { type: Boolean, required: true },
  confidence: { type: Number, required: true },
  raw: { type: Object },                   // store full ML response for audit
}, { timestamps: true });

module.exports = mongoose.model('Prediction', resultSchema);

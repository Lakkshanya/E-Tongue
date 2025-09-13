const mongoose = require('mongoose');

const resultSchema = new mongoose.Schema({
  sensorRef: { type: mongoose.Schema.Types.ObjectId, ref: 'SensorData', required: true },
  adulterated: { type: Boolean },
  safe_to_use: { type: Boolean },
  borderline: { type: Boolean },
  confidence: { type: Number },
  raw: { type: Object }
}, { timestamps: true });

module.exports = mongoose.model('Prediction', resultSchema);

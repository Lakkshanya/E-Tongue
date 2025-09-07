const mongoose = require('mongoose');

const featureSchema = new mongoose.Schema({
  Hardness: { type: Number, required: true, default: 0 },
  Solids_TDS: { type: Number, required: true, default: 0 },
  Sulphate: { type: Number, required: true, default: 0 },
  Chloramine: { type: Number, required: true, default: 0 },
  Conductivity: { type: Number, required: true, default: 0 },
  Organic_Carbon: { type: Number, required: true, default: 0 },
  Trihalomethane: { type: Number, required: true, default: 0 },
  Turbidity: { type: Number, required: true, default: 0 },
  pH: { type: Number, required: true, default: 0 }
}, { _id: false });

const sensorSchema = new mongoose.Schema({
  features: { type: featureSchema, required: true },
  temperature: { type: Number },
  deviceId: { type: String },
  source: { type: String, default: 'arduino' },
  meta: { type: Object },
}, { timestamps: true });

module.exports = mongoose.model('SensorData', sensorSchema);

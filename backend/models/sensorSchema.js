const mongoose = require('mongoose');

const featureSchema = new mongoose.Schema({
  Hardness: { type: Number, default: null },
  Solids_TDS: { type: Number, required: true },
  Sulphate: { type: Number, default: null },
  Chloramine: { type: Number, default: null },
  Conductivity: { type: Number, default: null },
  Organic_Carbon: { type: Number, default: null },
  Trihalomethane: { type: Number, default: null },
  Turbidity: { type: Number, default: null },
  pH: { type: Number, required: true }
}, { _id: false });

const sensorSchema = new mongoose.Schema({
  features: { type: featureSchema, required: true },
  temperature: { type: Number },
  deviceId: { type: String },
  source: { type: String, default: 'arduino' },
  meta: { type: Object }
}, { timestamps: true });

module.exports = mongoose.model('SensorData', sensorSchema);

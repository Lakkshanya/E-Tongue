require('dotenv').config();
const express = require('express');
const mongoose = require('mongoose');
const cors = require('cors');
const morgan = require('morgan');
const axios = require('axios');
const { SerialPort } = require('serialport');
const { ReadlineParser } = require('@serialport/parser-readline');

// Routes
const sensorsRouter = require('./routes/sensors');
const resultsRouter = require('./routes/results');

// Models
const SensorModel = require('./models/sensorSchema');
const PredictionModel = require('./models/resultSchema');

const app = express();
app.use(cors());
app.use(express.json());
app.use(morgan('dev'));

const {
  MONGO_URI,
  PORT = 4000,
  FLASK_URL = 'http://localhost:5001',
  SERIAL_PORT = 'COM10',
  SERIAL_BAUD = 9600
} = process.env;

mongoose.connect(MONGO_URI)
  .then(() => {
    console.log('✅ MongoDB connected');
    startSerialListener();
  })
  .catch(err => {
    console.error('❌ MongoDB connection error:', err.message);
    process.exit(1);
  });

app.use('/api/sensors', sensorsRouter);
app.use('/api/results', resultsRouter);

app.get('/api/health', async (_req, res) => {
  try {
    const { data } = await axios.get(`${FLASK_URL}/health`, { timeout: 3000 });
    res.json({ ok: true, ml: data });
  } catch (e) {
    res.status(200).json({ ok: true, ml: { ok: false, error: e.message } });
  }
});

app.use((req, res) => res.status(404).json({ error: 'Route not found' }));
app.use((err, _req, res, _next) => {
  console.error(err);
  res.status(err.status || 500).json({ error: err.message || 'Internal Server Error' });
});

app.listen(PORT, () => {
  console.log(`🚀 Node server running on http://localhost:${PORT}`);
});

// ---- SERIAL LISTENER ----
function startSerialListener() {
  console.log(`🔌 Opening serial port: ${SERIAL_PORT} @ ${SERIAL_BAUD}`);

  let port;
  try {
    port = new SerialPort({
      path: SERIAL_PORT,
      baudRate: Number(SERIAL_BAUD),
      autoOpen: false
    });
  } catch (err) {
    console.error('Failed to construct SerialPort:', err.message);
    return;
  }

  port.open(err => {
    if (err) {
      console.error(`❌ Failed to open serial port ${SERIAL_PORT}:`, err.message);
      return;
    }
    console.log(`✅ Serial port ${SERIAL_PORT} opened.`);
  });

  const parser = port.pipe(new ReadlineParser({ delimiter: '\n' }));

  parser.on('data', async (data) => {
    try {
      console.log('📥 From Arduino:', data.trim());
      const raw = JSON.parse(data);

      // ✅ Only include values actually sent by Arduino
      const features = {};
      if (raw.Hardness !== undefined) features.Hardness = raw.Hardness;
      if (raw.Solids_TDS !== undefined) features.Solids_TDS = raw.Solids_TDS;
      if (raw.Sulphate !== undefined) features.Sulphate = raw.Sulphate;
      if (raw.Chloramine !== undefined) features.Chloramine = raw.Chloramine;
      if (raw.Conductivity !== undefined) features.Conductivity = raw.Conductivity;
      if (raw.Organic_Carbon !== undefined) features.Organic_Carbon = raw.Organic_Carbon;
      if (raw.Trihalomethane !== undefined) features.Trihalomethane = raw.Trihalomethane;
      if (raw.Turbidity !== undefined) features.Turbidity = raw.Turbidity;
      if (raw.pH !== undefined) features.pH = raw.pH;

      const sensorDoc = await SensorModel.create({
        features,
        temperature: raw.temperature ?? undefined,
        source: 'arduino'
      });

      console.log('✅ Saved sensor data:', sensorDoc._id);

      // Call ML API → missing keys will get filled with medians in ml_api.py
      const response = await axios.post(`${FLASK_URL}/predict`, features);
      console.log('🤖 ML Response:', response.data);

      // Save prediction
      const predictionDoc = await PredictionModel.create({
        sensorRef: sensorDoc._id,
        adulterated: response.data.adulterated,
        safe_to_use: response.data.safe_to_use,
        borderline: response.data.borderline,
        confidence: response.data.confidence,
        raw: response.data
      });

      console.log(`✅ Prediction saved: ${predictionDoc._id}`);

    } catch (err) {
      console.error('⚠️ Serial data error:', err.message);
    }
  });

  port.on('close', () => {
    console.warn(`⚠️ Serial port ${SERIAL_PORT} closed.`);
  });
}

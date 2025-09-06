// server.js (Unified Express Server + Serial Reader for Node.js 20+/22 & serialport v10+)

require('dotenv').config();
const express = require('express');
const mongoose = require('mongoose');
const cors = require('cors');
const morgan = require('morgan');
const axios = require('axios');

// ✅ Updated imports for serialport v10+ (Node 20/22 compatible)
const { SerialPort } = require('serialport');
const { ReadlineParser } = require('@serialport/parser-readline');

const sensorsRouter = require('./routes/sensors');
const resultsRouter = require('./routes/results');

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
  SERIAL_PORT = 'COM5',
  SERIAL_BAUD = 9600
} = process.env;

// ✅ Connect to MongoDB first
mongoose.connect(MONGO_URI, {
  useNewUrlParser: true,
  useUnifiedTopology: true
})
  .then(() => {
    console.log('✅ MongoDB connected');
    startSerialListener(); // Start reading serial only after DB is connected
  })
  .catch(err => {
    console.error('❌ MongoDB connection error:', err.message);
    process.exit(1);
  });

// Health check endpoint
app.get('/api/health', async (_req, res) => {
  try {
    const { data } = await axios.get(`${FLASK_URL}/health`, { timeout: 3000 });
    res.json({ ok: true, ml: data });
  } catch (e) {
    res.status(200).json({ ok: true, ml: { ok: false, error: e.message } });
  }
});

app.use('/api/sensors', sensorsRouter);
app.use('/api/results', resultsRouter);

// 404 handler
app.use((req, res) => res.status(404).json({ error: 'Route not found' }));

// Error handler
app.use((err, _req, res, _next) => {
  console.error(err);
  res.status(err.status || 500).json({ error: err.message || 'Internal Server Error' });
});

app.listen(PORT, () => {
  console.log(`🚀 Node server running on http://localhost:${PORT}`);
});

/**
 * Serial Listener
 * Reads JSON from Arduino, saves to DB, calls ML API, saves predictions.
 */
function startSerialListener() {
  console.log(`🔌 Attempting to open serial port: ${SERIAL_PORT} @ ${SERIAL_BAUD} baud...`);

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
      console.log('Serial listener will not run until the port is available.');
      return;
    }
    console.log(`✅ Serial port ${SERIAL_PORT} opened at ${SERIAL_BAUD} baud.`);
  });

  port.on('error', (err) => {
    console.error('Serial port error:', err.message);
  });

  const parser = port.pipe(new ReadlineParser({ delimiter: '\n' }));

  parser.on('data', async (data) => {
    try {
      console.log('📥 Received from Arduino:', data);
      const sensorData = JSON.parse(data);

      // 1️⃣ Save raw sensor data
      const sensorDoc = await SensorModel.create(sensorData);

      // 2️⃣ Call ML API
      const response = await axios.post(`${FLASK_URL}/predict`, sensorData);
      console.log('🤖 ML Response:', response.data);

      // 3️⃣ Save prediction
      const predictionDoc = await PredictionModel.create({
        sensorRef: sensorDoc._id,
        adulterated: response.data.adulterated,
        safe_to_use: response.data.safe_to_use,
        borderline: response.data.borderline,
        confidence: response.data.confidence,
        raw: response.data
      });

      console.log('✅ Prediction saved:', predictionDoc._id);
    } catch (err) {
      console.error('⚠️ Error processing serial data:', err.message);
    }
  });

  port.on('close', () => {
    console.warn(`⚠️ Serial port ${SERIAL_PORT} closed.`);
  });
}

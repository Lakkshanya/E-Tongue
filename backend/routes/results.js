const express = require('express');
const Prediction = require('../models/resultSchema');

const router = express.Router();

// GET /api/results?limit=50 — latest predictions
router.get('/', async (req, res, next) => {
  try {
    const limit = Math.min(Number(req.query.limit) || 50, 200);
    const docs = await Prediction.find()
      .populate('sensorRef')
      .sort({ createdAt: -1 })
      .limit(limit);
    res.json(docs);
  } catch (err) {
    next(err);
  }
});

// GET /api/results/by-sensor/:sensorId — predictions for a given sensor record
router.get('/by-sensor/:sensorId', async (req, res, next) => {
  try {
    const docs = await Prediction.find({ sensorRef: req.params.sensorId }).sort({ createdAt: -1 });
    res.json(docs);
  } catch (err) {
    next(err);
  }
});

// GET /api/results/:id — a single prediction
router.get('/:id', async (req, res, next) => {
  try {
    const doc = await Prediction.findById(req.params.id).populate('sensorRef');
    if (!doc) return res.status(404).json({ error: 'Not found' });
    res.json(doc);
  } catch (err) {
    next(err);
  }
});

module.exports = router;

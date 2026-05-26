'use strict';
/**
 * /api/sync — import CSV / JSON into PostgreSQL
 *
 * GET  /api/sync/status              → DB row counts + default file paths
 * GET  /api/sync/sources             → which source files exist on disk
 * POST /api/sync/trajectories        → sync index + points (+ optional properties)
 * POST /api/sync/results             → sync experiment_results
 * POST /api/sync/all                 → sync everything
 *
 * POST body (all optional — defaults to project CSV paths):
 * {
 *   "indexFile": "data/processed/trajectories_index.csv",
 *   "pointsFile": "data/processed/trajectories_points.csv",
 *   "propertiesFile": "data/processed/trajectory_properties.csv",
 *   "resultsFile": "results/experiment_results.csv",
 *   "includeProperties": true,
 *   "index":  [ { trajectory_id, user_id, file_id, num_points, ... } ],
 *   "points": [ { trajectory_id, user_id, point_index, lat, lon, alt, timestamp } ],
 *   "rows":   [ { algorithm, trajectory_id, compression_ratio, ... } ]
 * }
 */

const express = require('express');
const {
  getSources,
  getStatus,
  syncTrajectories,
  syncResults,
  syncAll,
  DEFAULTS,
} = require('../services/syncService');

const router = express.Router();

function err(res, err) {
  console.error('[sync]', err.message);
  res.status(500).json({ ok: false, error: err.message });
}

// ── GET /status ───────────────────────────────────────────────────────────────
router.get('/status', async (_req, res) => {
  try {
    res.json(await getStatus());
  } catch (e) { err(res, e); }
});

// ── GET /sources ──────────────────────────────────────────────────────────────
router.get('/sources', (req, res) => {
  res.json({
    defaults: DEFAULTS,
    files: getSources(req.query),
    description: {
      trajectoriesIndex: '→ users, trajectory_index',
      trajectoriesPoints: '→ trajectory_points',
      trajectoryProperties: '→ enriches trajectory_index (duration, speed, stops, …)',
      experimentResults: '→ experiment_results',
    },
  });
});

// ── POST /trajectories ────────────────────────────────────────────────────────
router.post('/trajectories', async (req, res) => {
  try {
    const result = await syncTrajectories({ ...req.query, ...req.body });
    res.json(result);
  } catch (e) { err(res, e); }
});

// ── POST /results ─────────────────────────────────────────────────────────────
router.post('/results', async (req, res) => {
  try {
    const result = await syncResults({ ...req.query, ...req.body });
    res.json(result);
  } catch (e) { err(res, e); }
});

// ── POST /all ─────────────────────────────────────────────────────────────────
router.post('/all', async (req, res) => {
  try {
    const result = await syncAll({ ...req.query, ...req.body });
    res.json(result);
  } catch (e) { err(res, e); }
});

module.exports = router;

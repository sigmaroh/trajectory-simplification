'use strict';
/**
 * /api/trajectories — metadata and GPS points (PostgreSQL + CSV fallback)
 *
 * GET /api/trajectories/list
 * GET /api/trajectories/points?trajectory_id=0&algorithm=proposed&compression_ratio=5
 *     → { original, simplified, meta }  (map.js compatible)
 */

const fs = require('fs');
const { parse: parseSync } = require('csv-parse/sync');
const express = require('express');
const { query, queryOne } = require('../db');
const { DEFAULTS } = require('../lib/paths');
const { simplifyPoints } = require('../lib/simplify');
const router = express.Router();

function padUserId(uid) {
  if (uid == null || uid === '') return null;
  return String(uid).padStart(3, '0');
}

function errHandler(res, err) {
  console.error('[trajectories]', err.message);
  res.status(500).json({ error: err.message });
}

// ── CSV fallback loaders ───────────────────────────────────────────────────────

function loadIndexFromCsv() {
  const p = DEFAULTS.trajectoriesIndex;
  if (!fs.existsSync(p)) return [];
  return parseSync(fs.readFileSync(p, 'utf8'), { columns: true, skip_empty_lines: true });
}

function loadPointsFromCsv(userId, trajectoryId) {
  const p = DEFAULTS.trajectoriesPoints;
  if (!fs.existsSync(p)) return [];
  const uid = padUserId(userId);
  const tid = parseInt(trajectoryId);
  const raw = fs.readFileSync(p, 'utf8');
  const rows = parseSync(raw, { columns: true, skip_empty_lines: true });
  return rows
    .filter(r => padUserId(r.user_id) === uid && parseInt(r.trajectory_id) === tid)
    .map(r => ({
      lat: parseFloat(r.lat),
      lon: parseFloat(r.lon),
      alt: r.alt != null ? parseFloat(r.alt) : null,
      timestamp: r.timestamp || null,
    }));
}

async function loadIndexFromDb(userId, limit, offset) {
  const conditions = [];
  const params = [];
  if (userId) {
    params.push(padUserId(userId));
    conditions.push(`user_id = $${params.length}`);
  }
  const where = conditions.length ? `WHERE ${conditions.join(' AND ')}` : '';
  params.push(limit, offset);
  return query(
    `SELECT user_id, trajectory_id, file_id, num_points,
            start_time, end_time, mean_lat, mean_lon,
            duration, total_distance, mean_speed, max_speed,
            stop_ratio, turn_ratio, cv_interval
     FROM trajectory_index ${where}
     ORDER BY user_id, trajectory_id
     LIMIT $${params.length - 1} OFFSET $${params.length}`,
    params
  );
}

async function getTrajectoryList(userId, limit, offset) {
  try {
    const rows = await loadIndexFromDb(userId, limit, offset);
    if (rows.length > 0) return rows;
  } catch (err) {
    console.warn('[trajectories] DB list failed, using CSV:', err.message);
  }
  let rows = loadIndexFromCsv().map(r => ({
    user_id: padUserId(r.user_id),
    trajectory_id: parseInt(r.trajectory_id),
    file_id: r.file_id,
    num_points: parseInt(r.num_points),
  }));
  if (userId) rows = rows.filter(r => r.user_id === padUserId(userId));
  return rows.slice(offset, offset + limit);
}

async function resolveMeta(trajectoryId, userId) {
  const tid = parseInt(trajectoryId);
  let uid = userId ? padUserId(userId) : null;

  if (uid) {
    try {
      const meta = await queryOne(
        `SELECT * FROM trajectory_index WHERE user_id = $1 AND trajectory_id = $2`,
        [uid, tid]
      );
      if (meta) return meta;
    } catch (_) { /* fall through */ }
  }

  try {
    const meta = await queryOne(
      `SELECT * FROM trajectory_index WHERE trajectory_id = $1 LIMIT 1`,
      [tid]
    );
    if (meta) return meta;
  } catch (_) { /* fall through */ }

  const csv = loadIndexFromCsv().find(r => parseInt(r.trajectory_id) === tid);
  if (csv) {
    return {
      user_id: padUserId(csv.user_id),
      trajectory_id: tid,
      file_id: csv.file_id,
      num_points: parseInt(csv.num_points),
    };
  }
  return null;
}

async function loadAllPoints(userId, trajectoryId) {
  const uid = padUserId(userId);
  const tid = parseInt(trajectoryId);

  try {
    const rows = await query(
      `SELECT lat, lon, alt, recorded_at AS timestamp
       FROM trajectory_points
       WHERE user_id = $1 AND trajectory_id = $2
       ORDER BY point_index`,
      [uid, tid]
    );
    if (rows.length > 0) {
      return rows.map(r => ({
        lat: parseFloat(r.lat),
        lon: parseFloat(r.lon),
        alt: r.alt != null ? parseFloat(r.alt) : null,
        timestamp: r.timestamp,
      }));
    }
  } catch (err) {
    console.warn('[trajectories] DB points failed, using CSV:', err.message);
  }

  return loadPointsFromCsv(uid, tid);
}

// ── GET /list ─────────────────────────────────────────────────────────────────
router.get('/list', async (req, res) => {
  try {
    const limit = parseInt(req.query.limit || req.query.max || '200');
    const offset = parseInt(req.query.offset || '0');
    const rows = await getTrajectoryList(req.query.user_id, limit, offset);
    res.json(rows);
  } catch (err) { errHandler(res, err); }
});

// ── GET /users ────────────────────────────────────────────────────────────────
router.get('/users', async (_req, res) => {
  try {
    try {
      const rows = await query(`
        SELECT user_id, COUNT(*)::int AS trajectory_count, SUM(num_points) AS total_points
        FROM trajectory_index GROUP BY user_id ORDER BY user_id`);
      if (rows.length) return res.json(rows);
    } catch (_) { /* CSV fallback */ }

    const index = loadIndexFromCsv();
    const map = {};
    for (const r of index) {
      const u = padUserId(r.user_id);
      if (!map[u]) map[u] = { user_id: u, trajectory_count: 0, total_points: 0 };
      map[u].trajectory_count++;
      map[u].total_points += parseInt(r.num_points) || 0;
    }
    res.json(Object.values(map).sort((a, b) => a.user_id.localeCompare(b.user_id)));
  } catch (err) { errHandler(res, err); }
});

// ── GET /points ───────────────────────────────────────────────────────────────
async function handlePoints(req, res) {
  try {
    const { trajectory_id, user_id, algorithm, compression_ratio = 5 } = req.query;
    if (trajectory_id == null) {
      return res.status(400).json({ error: 'trajectory_id is required' });
    }

    const meta = await resolveMeta(trajectory_id, user_id);
    if (!meta) {
      return res.status(404).json({ error: `Trajectory ${trajectory_id} not found` });
    }

    const uid = meta.user_id;
    const tid = meta.trajectory_id;
    const cr = parseFloat(compression_ratio);
    const algo = algorithm || 'original';

    const allPoints = await loadAllPoints(uid, tid);
    if (allPoints.length < 2) {
      return res.status(404).json({ error: 'No GPS points found for this trajectory' });
    }

    const original = allPoints.map(p => ({ lat: p.lat, lon: p.lon }));

    let simplified = original;
    let budget = original.length;

    if (algo !== 'original' && algo !== 'none') {
      const result = simplifyPoints(allPoints, algo, cr);
      simplified = result.simplified;
      budget = result.budget;
    }

    res.json({
      original,
      simplified,
      meta: {
        trajectory_id: tid,
        user_id: uid,
        file_id: meta.file_id,
        n_original: original.length,
        n_simplified: simplified.length,
        compression_ratio: cr,
        budget,
      },
      points: original,
    });
  } catch (err) { errHandler(res, err); }
}

router.get('/points', handlePoints);

// ── GET /:user_id/:trajectory_id ──────────────────────────────────────────────
router.get('/:user_id/:trajectory_id', (req, res) => {
  req.query = {
    ...req.query,
    user_id: req.params.user_id,
    trajectory_id: req.params.trajectory_id,
  };
  return handlePoints(req, res);
});

module.exports = router;

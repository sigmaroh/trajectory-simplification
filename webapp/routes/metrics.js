'use strict';
/**
 * /api/metrics — experiment results from PostgreSQL, CSV fallback if DB empty
 */

const express = require('express');
const { getAllRows, filterRows, toNum } = require('../lib/metricsStore');
const router = express.Router();

const ALL_METRICS = [
  { key: 'hausdorff_distance',      label: 'Hausdorff Distance (m)',  log: true  },
  { key: 'average_pte',             label: 'Average PTE (m)',          log: true  },
  { key: 'frechet_distance',        label: 'Fréchet Distance (m)',     log: true  },
  { key: 'ped',                     label: 'PED (m)',                  log: true  },
  { key: 'sed',                     label: 'SED (m)',                  log: true  },
  { key: 'dad',                     label: 'DAD (degrees)',            log: false },
  { key: 'sad',                     label: 'SAD (m/s)',                log: false },
  { key: 'issd',                    label: 'ISSD (m·s)',               log: true  },
  { key: 'turn_preservation',       label: 'Turn Preservation (0–1)',  log: false },
  { key: 'stop_preservation',       label: 'Stop Preservation (0–1)',  log: false },
  { key: 'runtime_seconds',         label: 'Runtime (seconds)',        log: false },
  { key: 'memory_mb',               label: 'Memory (MB)',              log: false },
  { key: 'throughput_traj_per_sec', label: 'Throughput (traj/s)',      log: false },
];

const TARGET_CRS = [2, 5, 10, 20];
const CR_TOL = 0.6;

function snapCR(v) {
  const n = parseFloat(v);
  const candidates = TARGET_CRS.filter(t => Math.abs(n - t) <= CR_TOL);
  if (!candidates.length) return null;
  return candidates.reduce((a, b) => (Math.abs(n - a) <= Math.abs(n - b) ? a : b));
}

function errHandler(res, err) {
  console.error('[metrics]', err.message);
  res.status(500).json({ error: err.message });
}

// ── GET /filters ──────────────────────────────────────────────────────────────
router.get('/filters', async (_req, res) => {
  try {
    const { rows } = await getAllRows();
    const algorithms = [...new Set(rows.map(r => r.algorithm))].filter(Boolean).sort();
    const users = [...new Set(rows.map(r => r.user_id).filter(Boolean))].sort();
    const rawCRs = rows.map(r => parseFloat(r.compression_ratio)).filter(n => !isNaN(n));
    const snapped = [...new Set(rawCRs.map(snapCR).filter(n => n != null))].sort((a, b) => a - b);
    const compressionRatios = snapped.length ? snapped : [...new Set(rawCRs)].sort((a, b) => a - b);
    res.json({ algorithms, users, compressionRatios });
  } catch (err) { errHandler(res, err); }
});

// ── GET /results ────────────────────────────────────────────────────────────────
router.get('/results', async (req, res) => {
  try {
    const { rows } = await getAllRows();
    let out = filterRows(rows, req.query);
    const limit = parseInt(req.query.limit || req.query.max || '5000');
    const offset = parseInt(req.query.offset || '0');
    out = out.slice(offset, offset + limit);
    res.json(out);
  } catch (err) { errHandler(res, err); }
});

// ── GET /summary ────────────────────────────────────────────────────────────────
router.get('/summary', async (req, res) => {
  try {
    const { rows } = await getAllRows();
    const filtered = filterRows(rows, req.query);

    const groups = {};
    for (const row of filtered) {
      const cr = parseFloat(row.compression_ratio);
      if (isNaN(cr)) continue;
      const key = `${row.algorithm}__${cr}`;
      if (!groups[key]) groups[key] = { algorithm: row.algorithm, compression_ratio: cr, rows: [] };
      groups[key].rows.push(row);
    }

    const metricKeys = [
      'hausdorff_distance', 'average_pte', 'frechet_distance',
      'runtime_seconds', 'memory_mb', 'throughput_traj_per_sec',
      'turn_preservation', 'stop_preservation',
      'input_points', 'output_points', 'actual_compression_ratio',
    ];

    const summary = Object.values(groups).map(g => {
      const out = { algorithm: g.algorithm, compression_ratio: g.compression_ratio, count: g.rows.length };
      for (const col of metricKeys) {
        const vals = g.rows.map(r => toNum(r[col])).filter(v => v !== null);
        out[`${col}_mean`] = vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : null;
      }
      return out;
    });

    summary.sort((a, b) => a.algorithm.localeCompare(b.algorithm) || a.compression_ratio - b.compression_ratio);
    res.json(summary);
  } catch (err) { errHandler(res, err); }
});

// ── GET /comparison ───────────────────────────────────────────────────────────
router.get('/comparison', async (req, res) => {
  try {
    const metric = req.query.metric || 'hausdorff_distance';
    const { rows } = await getAllRows();
    const filtered = filterRows(rows, req.query);

    const algoList = req.query.algorithms
      ? req.query.algorithms.split(',').map(s => s.trim())
      : [...new Set(filtered.map(r => r.algorithm))].filter(Boolean).sort();

    const series = {};
    for (const algo of algoList) {
      series[algo] = TARGET_CRS.map(targetCR => {
        const bucket = filtered.filter(r =>
          r.algorithm === algo && snapCR(r.compression_ratio) === targetCR
        );
        const vals = bucket.map(r => toNum(r[metric])).filter(v => v !== null);
        return vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : null;
      });
    }

    res.json({ compressionRatios: TARGET_CRS, series, metric });
  } catch (err) { errHandler(res, err); }
});

// ── GET /plotdata ─────────────────────────────────────────────────────────────
router.get('/plotdata', async (req, res) => {
  try {
    const { rows } = await getAllRows();
    const filtered = filterRows(rows, req.query);

    const algorithms = [...new Set(filtered.map(r => r.algorithm))].filter(Boolean).sort();
    const tagged = filtered
      .map(r => ({ ...r, _cr: snapCR(r.compression_ratio) }))
      .filter(r => r._cr !== null);

    const pivot = {};
    for (const m of ALL_METRICS) {
      if (!tagged.some(r => r[m.key] != null && r[m.key] !== '')) continue;
      pivot[m.key] = {};
      for (const algo of algorithms) {
        pivot[m.key][algo] = {};
        for (const cr of TARGET_CRS) {
          const bucket = tagged.filter(r => r.algorithm === algo && r._cr === cr);
          const vals = bucket.map(r => toNum(r[m.key])).filter(v => v !== null);
          if (!vals.length) continue;
          const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
          const std = vals.length > 1
            ? Math.sqrt(vals.reduce((s, v) => s + (v - mean) ** 2, 0) / (vals.length - 1))
            : 0;
          pivot[m.key][algo][cr] = { mean: +mean.toFixed(6), std: +std.toFixed(6), n: vals.length };
        }
      }
    }

    res.json({
      algorithms,
      compressionRatios: TARGET_CRS,
      metrics: ALL_METRICS.filter(m => pivot[m.key]),
      pivot,
    });
  } catch (err) { errHandler(res, err); }
});

module.exports = router;

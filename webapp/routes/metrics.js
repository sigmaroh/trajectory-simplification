/**
 * /api/metrics  — serves experiment_results.csv with filtering
 *
 * GET /api/metrics/filters          → { algorithms, users, compressionRatios }
 * GET /api/metrics/results          → filtered rows
 * GET /api/metrics/summary          → mean by algorithm × compression_ratio
 * GET /api/metrics/comparison       → pivot for bar/line charts
 * GET /api/metrics/plotdata         → full pivot ready for generate_plots-style charts
 */

const express = require('express');
const fs      = require('fs');
const path    = require('path');
const { parse } = require('csv-parse/sync');

const router = express.Router();

// ── path resolution ──────────────────────────────────────────────────────────
const ROOT       = path.join(__dirname, '..', '..');
const RESULTS_CSV = path.join(ROOT, 'results', 'experiment_results.csv');
const SUMMARY_CSV = path.join(ROOT, 'results', 'summary_table.csv');

function loadCSV(filepath) {
  if (!fs.existsSync(filepath)) return [];
  const raw = fs.readFileSync(filepath, 'utf8');
  return parse(raw, { columns: true, skip_empty_lines: true, cast: true });
}

function toNum(v) {
  const n = parseFloat(v);
  return isNaN(n) ? null : n;
}

// ── filters ──────────────────────────────────────────────────────────────────
router.get('/filters', (_req, res) => {
  const rows = loadCSV(RESULTS_CSV);
  const algorithms       = [...new Set(rows.map(r => r.algorithm))].filter(Boolean).sort();
  const users            = [...new Set(rows.map(r => r.user_id))].filter(Boolean).sort();
  const compressionRatios = [...new Set(rows.map(r => parseFloat(r.compression_ratio)).filter(n => !isNaN(n)))].sort((a,b) => a-b);
  res.json({ algorithms, users, compressionRatios });
});

// ── raw results with optional filters ────────────────────────────────────────
router.get('/results', (req, res) => {
  let rows = loadCSV(RESULTS_CSV);

  const { algorithm, user_id, compression_ratio } = req.query;

  if (algorithm)
    rows = rows.filter(r => r.algorithm === algorithm);
  if (user_id)
    rows = rows.filter(r => String(r.user_id) === String(user_id));
  if (compression_ratio) {
    const cr = parseFloat(compression_ratio);
    rows = rows.filter(r => Math.abs(parseFloat(r.compression_ratio) - cr) < cr * 0.05);
  }

  res.json(rows);
});

// ── aggregated summary ────────────────────────────────────────────────────────
const METRIC_COLS = [
  'hausdorff_distance','average_pte','frechet_distance',
  'runtime_seconds','memory_mb','throughput_traj_per_sec',
  'turn_preservation','stop_preservation',
  'input_points','output_points','actual_compression_ratio',
];

router.get('/summary', (req, res) => {
  let rows = loadCSV(RESULTS_CSV);

  const { algorithm, user_id } = req.query;
  if (algorithm) rows = rows.filter(r => r.algorithm === algorithm);
  if (user_id)   rows = rows.filter(r => String(r.user_id) === String(user_id));

  // group by algorithm × compression_ratio
  const groups = {};
  for (const row of rows) {
    const cr  = parseFloat(row.compression_ratio);
    if (isNaN(cr)) continue;
    const key = `${row.algorithm}__${cr}`;
    if (!groups[key]) groups[key] = { algorithm: row.algorithm, compression_ratio: cr, rows: [] };
    groups[key].rows.push(row);
  }

  const summary = Object.values(groups).map(g => {
    const out = { algorithm: g.algorithm, compression_ratio: g.compression_ratio, count: g.rows.length };
    for (const col of METRIC_COLS) {
      const vals = g.rows.map(r => toNum(r[col])).filter(v => v !== null);
      out[col + '_mean'] = vals.length ? vals.reduce((a,b)=>a+b,0)/vals.length : null;
    }
    return out;
  });

  summary.sort((a,b) => a.algorithm.localeCompare(b.algorithm) || a.compression_ratio - b.compression_ratio);
  res.json(summary);
});

// ── comparison pivot for charts ───────────────────────────────────────────────
// Returns { compressionRatios: [...], series: { algo: { metric: [...values] } } }
router.get('/comparison', (req, res) => {
  const { metric = 'hausdorff_distance', algorithms, user_id } = req.query;
  let rows = loadCSV(RESULTS_CSV);

  if (user_id) rows = rows.filter(r => String(r.user_id) === String(user_id));

  const algoList = algorithms
    ? algorithms.split(',').map(s => s.trim())
    : [...new Set(rows.map(r => r.algorithm))].filter(Boolean).sort();

  const crSet = [...new Set(
    rows.map(r => parseFloat(r.compression_ratio)).filter(n => !isNaN(n))
  )].sort((a,b) => a-b);

  const roundCR = cr => Math.round(cr);
  const targetCRs = [...new Set(crSet.map(roundCR))].sort((a,b) => a-b);

  const series = {};
  for (const algo of algoList) {
    series[algo] = targetCRs.map(targetCR => {
      const bucket = rows.filter(
        r => r.algorithm === algo && Math.abs(roundCR(parseFloat(r.compression_ratio)) - targetCR) < 1
      );
      const vals = bucket.map(r => toNum(r[metric])).filter(v => v !== null);
      return vals.length ? vals.reduce((a,b)=>a+b,0)/vals.length : null;
    });
  }

  res.json({ compressionRatios: targetCRs, series, metric });
});

// ── plotdata: full pivot for all metrics × algorithms × CRs ──────────────────
// Matches generate_plots.py logic: snap each row's CR to nearest target CR
const ALL_METRICS = [
  { key: 'hausdorff_distance',     label: 'Hausdorff Distance (m)',   log: true  },
  { key: 'average_pte',            label: 'Average PTE (m)',           log: true  },
  { key: 'frechet_distance',       label: 'Fréchet Distance (m)',      log: true  },
  { key: 'ped',                    label: 'PED (m)',                    log: true  },
  { key: 'sed',                    label: 'SED (m)',                    log: true  },
  { key: 'dad',                    label: 'DAD (degrees)',              log: false },
  { key: 'sad',                    label: 'SAD (m/s)',                  log: false },
  { key: 'issd',                   label: 'ISSD (m·s)',                 log: true  },
  { key: 'turn_preservation',      label: 'Turn Preservation (0–1)',    log: false },
  { key: 'stop_preservation',      label: 'Stop Preservation (0–1)',    log: false },
  { key: 'runtime_seconds',        label: 'Runtime (seconds)',          log: false },
  { key: 'memory_mb',              label: 'Memory (MB)',                log: false },
  { key: 'throughput_traj_per_sec',label: 'Throughput (traj/s)',        log: false },
];

router.get('/plotdata', (req, res) => {
  let rows = loadCSV(RESULTS_CSV);
  const { algorithm, user_id } = req.query;
  if (algorithm) rows = rows.filter(r => r.algorithm === algorithm);
  if (user_id)   rows = rows.filter(r => String(r.user_id) === String(user_id));

  const TARGET_CRS = [2, 5, 10, 20];
  const TOL        = 0.6;

  function snapCR(v) {
    const n = parseFloat(v);
    const candidates = TARGET_CRS.filter(t => Math.abs(n - t) <= TOL);
    if (!candidates.length) return null;
    return candidates.reduce((a, b) => Math.abs(n - a) <= Math.abs(n - b) ? a : b);
  }

  // tag each row with snapped CR
  const tagged = rows.map(r => ({ ...r, _cr: snapCR(r.compression_ratio) })).filter(r => r._cr !== null);

  const algorithms = [...new Set(tagged.map(r => r.algorithm))].filter(Boolean).sort();

  // build pivot: { metric → { algo → { cr → { mean, std, n } } } }
  const pivot = {};
  for (const m of ALL_METRICS) {
    if (!tagged.some(r => r[m.key] != null && r[m.key] !== '')) continue;
    pivot[m.key] = {};
    for (const algo of algorithms) {
      pivot[m.key][algo] = {};
      for (const cr of TARGET_CRS) {
        const bucket = tagged.filter(r => r.algorithm === algo && r._cr === cr);
        const vals   = bucket.map(r => toNum(r[m.key])).filter(v => v !== null);
        if (!vals.length) continue;
        const mean = vals.reduce((a,b)=>a+b,0) / vals.length;
        const std  = vals.length > 1
          ? Math.sqrt(vals.reduce((s,v)=>s+(v-mean)**2,0)/(vals.length-1))
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
});

module.exports = router;

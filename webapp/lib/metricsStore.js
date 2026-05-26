'use strict';
/**
 * Load experiment results from PostgreSQL, with CSV fallback.
 */
const fs = require('fs');
const { parse: parseSync } = require('csv-parse/sync');
const { query } = require('../db');
const { DEFAULTS } = require('./paths');

const RESULTS_CSV = DEFAULTS.experimentResults;

function toNum(v) {
  const n = parseFloat(v);
  return isNaN(n) ? null : n;
}

function loadCsvRows() {
  if (!fs.existsSync(RESULTS_CSV)) return [];
  const raw = fs.readFileSync(RESULTS_CSV, 'utf8');
  return parseSync(raw, { columns: true, skip_empty_lines: true, cast: false });
}

async function loadDbRows() {
  const rows = await query(`SELECT * FROM experiment_results ORDER BY algorithm, compression_ratio, trajectory_id`);
  return rows;
}

async function getAllRows() {
  try {
    const dbRows = await loadDbRows();
    if (dbRows.length > 0) return { rows: dbRows, source: 'db' };
  } catch (err) {
    console.warn('[metricsStore] DB unavailable, using CSV:', err.message);
  }
  const csvRows = loadCsvRows();
  return { rows: csvRows, source: 'csv' };
}

function filterRows(rows, { algorithm, user_id, compression_ratio } = {}) {
  let out = rows;
  if (algorithm) out = out.filter(r => r.algorithm === algorithm);
  if (user_id) out = out.filter(r => String(r.user_id || '').padStart(3, '0') === String(user_id).padStart(3, '0'));
  if (compression_ratio) {
    const cr = parseFloat(compression_ratio);
    out = out.filter(r => Math.abs(parseFloat(r.compression_ratio) - cr) < cr * 0.05);
  }
  return out;
}

module.exports = { getAllRows, filterRows, loadCsvRows, toNum, RESULTS_CSV };

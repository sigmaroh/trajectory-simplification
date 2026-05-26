'use strict';
/**
 * Sync CSV / JSON data into PostgreSQL (replaces src/utils/db_export.py).
 *
 * Tables populated:
 *   users, trajectory_index, trajectory_points  ← trajectories_index.csv
 *                                                 + trajectories_points.csv
 *                                                 + trajectory_properties.csv (optional)
 *   experiment_results                          ← experiment_results.csv
 */

const fs = require('fs');
const { parse: parseSync } = require('csv-parse/sync');
const { parse } = require('csv-parse');
const { pool } = require('../db');
const { DEFAULTS, resolvePath } = require('../lib/paths');

// ── helpers ───────────────────────────────────────────────────────────────────

function padUserId(uid) {
  if (uid == null || uid === '') return 'UNK';
  return String(uid).padStart(3, '0');
}

function toNum(v) {
  if (v == null || v === '') return null;
  const n = parseFloat(v);
  return Number.isFinite(n) ? n : null;
}

function toInt(v) {
  if (v == null || v === '') return null;
  const n = parseInt(v, 10);
  return Number.isFinite(n) ? n : null;
}

function fileInfo(filePath) {
  if (!filePath || !fs.existsSync(filePath)) {
    return { path: filePath, exists: false };
  }
  const st = fs.statSync(filePath);
  return {
    path: filePath,
    exists: true,
    sizeBytes: st.size,
    modifiedAt: st.mtime.toISOString(),
  };
}

async function tableCounts(client) {
  const tables = ['users', 'trajectory_index', 'trajectory_points', 'experiment_results'];
  const counts = {};
  for (const t of tables) {
    const res = await client.query(`SELECT COUNT(*)::int AS n FROM ${t}`);
    counts[t] = res.rows[0].n;
  }
  return counts;
}

/** Normalise a value for stable conflict-key comparison. */
function normConflictValue(col, val) {
  if (val == null || val === '') return '';
  if (col === 'user_id') return padUserId(val);
  if (col === 'trajectory_id') return String(parseInt(val, 10));
  if (col === 'compression_ratio') return Number(val).toFixed(6);
  return String(val);
}

/** Keep one row per conflict key (last wins). Prevents PG error 21000. */
function dedupeByConflictKey(rows, conflictCols) {
  const map = new Map();
  for (const row of rows) {
    const key = conflictCols.map(c => normConflictValue(c, row[c])).join('\0');
    map.set(key, row);
  }
  return [...map.values()];
}

async function upsertBatch(client, table, cols, rows, conflictCols, updateCols) {
  if (!rows.length) return 0;

  const unique = dedupeByConflictKey(rows, conflictCols);
  if (unique.length < rows.length) {
    console.warn(`[sync] deduped ${rows.length - unique.length} duplicate row(s) in ${table}`);
  }

  const values = [];
  const tuples = [];
  let idx = 1;

  for (const row of unique) {
    const ph = cols.map(() => `$${idx++}`);
    tuples.push(`(${ph.join(',')})`);
    for (const c of cols) values.push(row[c] ?? null);
  }

  const updates = [
    ...updateCols.map(c => `${c} = EXCLUDED.${c}`),
    ...(table === 'trajectory_index' ? ['updated_at = now()'] : []),
  ].join(', ');

  const sql = `
    INSERT INTO ${table} (${cols.join(', ')})
    VALUES ${tuples.join(', ')}
    ON CONFLICT (${conflictCols.join(', ')})
    DO UPDATE SET ${updates}
  `;

  await client.query(sql, values);
  return unique.length;
}

async function upsertUsers(client, userIds) {
  const unique = [...new Set(userIds.map(padUserId))];
  if (!unique.length) return 0;

  await client.query(
    `INSERT INTO users (user_id)
     SELECT unnest($1::char(3)[])
     ON CONFLICT (user_id) DO NOTHING`,
    [unique]
  );
  return unique.length;
}

function indexRowFromCsv(row, props = null) {
  const uid = padUserId(row.user_id);
  const tid = toInt(row.trajectory_id);
  const p = props || row;
  return {
    user_id: uid,
    trajectory_id: tid,
    file_id: row.file_id != null ? String(row.file_id) : null,
    num_points: toInt(p.num_points ?? row.num_points),
    start_time: row.start_time || null,
    end_time: row.end_time || null,
    mean_lat: toNum(row.mean_lat),
    mean_lon: toNum(row.mean_lon),
    duration: toNum(p.duration),
    total_distance: toNum(p.total_distance),
    mean_interval: toNum(p.mean_interval),
    std_interval: toNum(p.std_interval),
    cv_interval: toNum(p.cv_interval),
    mean_speed: toNum(p.mean_speed),
    max_speed: toNum(p.max_speed),
    mean_direction_change: toNum(p.mean_direction_change),
    num_stops: toInt(p.num_stops),
    stop_ratio: toNum(p.stop_ratio),
    num_turns: toInt(p.num_turns),
    turn_ratio: toNum(p.turn_ratio),
  };
}

async function upsertTrajectoryIndex(client, rows) {
  const cols = [
    'user_id', 'trajectory_id', 'file_id', 'num_points',
    'start_time', 'end_time', 'mean_lat', 'mean_lon',
    'duration', 'total_distance', 'mean_interval', 'std_interval',
    'cv_interval', 'mean_speed', 'max_speed', 'mean_direction_change',
    'num_stops', 'stop_ratio', 'num_turns', 'turn_ratio',
  ];
  const updateCols = cols.filter(c => !['user_id', 'trajectory_id'].includes(c));
  return upsertBatch(client, 'trajectory_index', cols, rows,
    ['user_id', 'trajectory_id'], updateCols);
}

async function upsertTrajectoryPoints(client, rows) {
  const cols = ['user_id', 'trajectory_id', 'point_index', 'lat', 'lon', 'alt', 'recorded_at'];
  const updateCols = ['lat', 'lon', 'alt', 'recorded_at'];
  return upsertBatch(client, 'trajectory_points', cols, rows,
    ['user_id', 'trajectory_id', 'point_index'], updateCols);
}

async function upsertExperimentResults(client, rows) {
  const cols = [
    'algorithm', 'trajectory_id', 'user_id', 'file_id', 'compression_ratio',
    'input_points', 'output_points', 'budget', 'actual_compression_ratio',
    'runtime_seconds', 'memory_mb', 'throughput_traj_per_sec',
    'hausdorff_distance', 'average_pte', 'frechet_distance',
    'turn_preservation', 'stop_preservation',
  ];
  const pk = ['algorithm', 'trajectory_id', 'user_id', 'compression_ratio'];
  const updateCols = cols.filter(c => !pk.includes(c));
  return upsertBatch(client, 'experiment_results', cols, rows, pk, updateCols);
}

function readCsvSync(filePath) {
  const raw = fs.readFileSync(filePath, 'utf8');
  return parseSync(raw, { columns: true, skip_empty_lines: true, cast: false });
}

function streamCsvBatches(filePath, batchSize, onBatch) {
  return new Promise((resolve, reject) => {
    let batch = [];
    let chain = Promise.resolve();

    const parser = fs.createReadStream(filePath)
      .pipe(parse({ columns: true, skip_empty_lines: true, cast: false }));

    parser.on('data', (row) => {
      batch.push(row);
      if (batch.length >= batchSize) {
        parser.pause();
        const chunk = batch;
        batch = [];
        chain = chain
          .then(() => onBatch(chunk))
          .then(() => parser.resume())
          .catch((err) => { parser.destroy(); reject(err); });
      }
    });

    parser.on('end', () => {
      chain
        .then(() => (batch.length ? onBatch(batch) : null))
        .then(resolve)
        .catch(reject);
    });

    parser.on('error', reject);
  });
}

async function updateIndexAggregates(client) {
  await client.query(`
    UPDATE trajectory_index ti SET
      mean_lat   = sub.mean_lat,
      mean_lon   = sub.mean_lon,
      start_time = sub.start_time,
      end_time   = sub.end_time,
      num_points = sub.cnt,
      updated_at = now()
    FROM (
      SELECT user_id, trajectory_id,
             AVG(lat)::float8 AS mean_lat,
             AVG(lon)::float8 AS mean_lon,
             MIN(recorded_at) AS start_time,
             MAX(recorded_at) AS end_time,
             COUNT(*)::int    AS cnt
      FROM trajectory_points
      GROUP BY user_id, trajectory_id
    ) sub
    WHERE ti.user_id = sub.user_id AND ti.trajectory_id = sub.trajectory_id
  `);
}

async function loadTrajectoryUserMap(client) {
  const res = await client.query(
    `SELECT trajectory_id, user_id, file_id FROM trajectory_index`
  );
  const map = new Map();
  for (const r of res.rows) {
    map.set(r.trajectory_id, { user_id: r.user_id, file_id: r.file_id });
  }
  return map;
}

// ── public API ────────────────────────────────────────────────────────────────

function getSources(custom = {}) {
  return {
    trajectoriesIndex: fileInfo(resolvePath(custom.indexFile) || DEFAULTS.trajectoriesIndex),
    trajectoriesPoints: fileInfo(resolvePath(custom.pointsFile) || DEFAULTS.trajectoriesPoints),
    trajectoryProperties: fileInfo(resolvePath(custom.propertiesFile) || DEFAULTS.trajectoryProperties),
    experimentResults: fileInfo(resolvePath(custom.resultsFile) || DEFAULTS.experimentResults),
  };
}

async function getStatus() {
  const client = await pool.connect();
  try {
    return { tables: await tableCounts(client), sources: getSources() };
  } finally {
    client.release();
  }
}

/**
 * Sync trajectories from CSV files and/or JSON arrays in options.
 * options: { indexFile, pointsFile, propertiesFile, index[], points[], includeProperties }
 */
async function syncTrajectories(options = {}) {
  const indexPath = resolvePath(options.indexFile) || DEFAULTS.trajectoriesIndex;
  const pointsPath = resolvePath(options.pointsFile) || DEFAULTS.trajectoriesPoints;
  const propsPath = resolvePath(options.propertiesFile) || DEFAULTS.trajectoryProperties;
  const includeProperties = options.includeProperties !== false;
  const batchSize = options.batchSize || 2000;

  const client = await pool.connect();
  const stats = { users: 0, indexRows: 0, pointRows: 0, propertiesRows: 0 };

  try {
    await client.query('BEGIN');

    // ── index rows (CSV or JSON) ──
    let indexRows = [];
    if (Array.isArray(options.index) && options.index.length) {
      indexRows = options.index.map(r => indexRowFromCsv(r));
    } else if (fs.existsSync(indexPath)) {
      indexRows = readCsvSync(indexPath).map(r => indexRowFromCsv(r));
    } else {
      throw new Error(`Index file not found: ${indexPath}`);
    }

    stats.users = await upsertUsers(client, indexRows.map(r => r.user_id));

    // ── optional properties merge ──
    if (includeProperties && Array.isArray(options.properties)) {
      const propMap = new Map(
        options.properties.map(p => [`${padUserId(p.user_id)}:${toInt(p.trajectory_id)}`, p])
      );
      indexRows = indexRows.map(r => {
        const p = propMap.get(`${r.user_id}:${r.trajectory_id}`);
        return p ? indexRowFromCsv(r, p) : r;
      });
      stats.propertiesRows = options.properties.length;
    } else if (includeProperties && fs.existsSync(propsPath)) {
      const propMap = new Map();
      await streamCsvBatches(propsPath, 500, async (batch) => {
        for (const row of batch) {
          const key = `${padUserId(row.user_id)}:${toInt(row.trajectory_id)}`;
          propMap.set(key, row);
        }
        stats.propertiesRows += batch.length;
      });
      indexRows = indexRows.map(r => {
        const p = propMap.get(`${r.user_id}:${r.trajectory_id}`);
        return p ? indexRowFromCsv(r, p) : r;
      });
    }

    // upsert index in chunks (dedupe each chunk)
    indexRows = dedupeByConflictKey(indexRows, ['user_id', 'trajectory_id']);
    for (let i = 0; i < indexRows.length; i += 500) {
      const chunk = dedupeByConflictKey(indexRows.slice(i, i + 500), ['user_id', 'trajectory_id']);
      stats.indexRows += await upsertTrajectoryIndex(client, chunk);
    }

    // ── GPS points (CSV stream or JSON array) ──
    const pointMapper = (row) => ({
      user_id: padUserId(row.user_id),
      trajectory_id: toInt(row.trajectory_id),
      point_index: toInt(row.point_index),
      lat: toNum(row.lat),
      lon: toNum(row.lon),
      alt: toNum(row.alt),
      recorded_at: row.timestamp || row.recorded_at || null,
    });

    if (Array.isArray(options.points) && options.points.length) {
      for (let i = 0; i < options.points.length; i += batchSize) {
        const batch = dedupeByConflictKey(
          options.points.slice(i, i + batchSize).map(pointMapper).filter(p => p.lat != null && p.lon != null),
          ['user_id', 'trajectory_id', 'point_index']
        );
        stats.pointRows += await upsertTrajectoryPoints(client, batch);
      }
    } else if (fs.existsSync(pointsPath)) {
      await streamCsvBatches(pointsPath, batchSize, async (batch) => {
        const mapped = dedupeByConflictKey(
          batch.map(pointMapper).filter(p => p.lat != null && p.lon != null),
          ['user_id', 'trajectory_id', 'point_index']
        );
        stats.pointRows += await upsertTrajectoryPoints(client, mapped);
      });
    } else if (!options.points) {
      throw new Error(`Points file not found: ${pointsPath}`);
    }

    await updateIndexAggregates(client);
    await client.query('COMMIT');

    return { ok: true, ...stats, tables: await tableCounts(client) };
  } catch (err) {
    await client.query('ROLLBACK');
    throw err;
  } finally {
    client.release();
  }
}

/**
 * Sync experiment results from CSV or JSON.
 * Resolves user_id from trajectory_index when missing in source row.
 */
async function syncResults(options = {}) {
  const resultsPath = resolvePath(options.resultsFile) || DEFAULTS.experimentResults;
  const batchSize = options.batchSize || 500;

  const client = await pool.connect();
  const stats = { resultRows: 0, skipped: 0 };

  try {
    await client.query('BEGIN');

    const userMap = await loadTrajectoryUserMap(client);

    let rows = [];
    if (Array.isArray(options.rows) && options.rows.length) {
      rows = options.rows;
    } else if (fs.existsSync(resultsPath)) {
      rows = readCsvSync(resultsPath);
    } else {
      throw new Error(`Results file not found: ${resultsPath}`);
    }

    const mapped = [];
    for (const row of rows) {
      const tid = toInt(row.trajectory_id);
      const cr = toNum(row.compression_ratio);
      const algo = row.algorithm;
      if (!algo || tid == null || cr == null) {
        stats.skipped++;
        continue;
      }

      let uid = row.user_id ? padUserId(row.user_id) : null;
      let fid = row.file_id || null;
      if (!uid) {
        const meta = userMap.get(tid);
        if (!meta) { stats.skipped++; continue; }
        uid = meta.user_id;
        fid = fid || meta.file_id;
      }

      mapped.push({
        algorithm: String(algo),
        trajectory_id: tid,
        user_id: uid,
        file_id: fid ? String(fid) : null,
        compression_ratio: cr,
        input_points: toInt(row.input_points),
        output_points: toInt(row.output_points),
        budget: toInt(row.budget),
        actual_compression_ratio: toNum(row.actual_compression_ratio),
        runtime_seconds: toNum(row.runtime_seconds),
        memory_mb: toNum(row.memory_mb),
        throughput_traj_per_sec: toNum(row.throughput_traj_per_sec),
        hausdorff_distance: toNum(row.hausdorff_distance),
        average_pte: toNum(row.average_pte),
        frechet_distance: toNum(row.frechet_distance),
        turn_preservation: toNum(row.turn_preservation),
        stop_preservation: toNum(row.stop_preservation),
      });
    }

    const pk = ['algorithm', 'trajectory_id', 'user_id', 'compression_ratio'];
    const deduped = dedupeByConflictKey(mapped, pk);
    if (deduped.length < mapped.length) {
      console.warn(`[sync] deduped ${mapped.length - deduped.length} duplicate experiment result(s) total`);
    }

    for (let i = 0; i < deduped.length; i += batchSize) {
      const chunk = dedupeByConflictKey(deduped.slice(i, i + batchSize), pk);
      stats.resultRows += await upsertExperimentResults(client, chunk);
    }

    await client.query('COMMIT');
    return { ok: true, ...stats, tables: await tableCounts(client) };
  } catch (err) {
    await client.query('ROLLBACK');
    throw err;
  } finally {
    client.release();
  }
}

async function syncAll(options = {}) {
  const traj = await syncTrajectories(options);
  const results = await syncResults(options);
  return { ok: true, trajectories: traj, results };
}

module.exports = {
  DEFAULTS,
  getSources,
  getStatus,
  syncTrajectories,
  syncResults,
  syncAll,
};

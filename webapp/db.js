'use strict';
/**
 * Shared PostgreSQL connection pool.
 * Reads credentials from .env (loaded by server.js / migrate.js before this).
 */
const { Pool } = require('pg');

const pool = new Pool({
  host:     process.env.PGHOST     || 'localhost',
  port:     parseInt(process.env.PGPORT || '5432'),
  user:     process.env.PGUSER     || 'postgres',
  password: process.env.PGPASSWORD || '',
  database: process.env.PGDATABASE || 'trajectory_db',
  max: 10,                // max pool connections
  idleTimeoutMillis: 30000,
  connectionTimeoutMillis: 5000,
});

pool.on('error', (err) => {
  console.error('[db] idle client error:', err.message);
});

/** Run a parameterised query and return rows. */
async function query(sql, params = []) {
  const client = await pool.connect();
  try {
    const result = await client.query(sql, params);
    return result.rows;
  } finally {
    client.release();
  }
}

/** Run a query and return the first row (or null). */
async function queryOne(sql, params = []) {
  const rows = await query(sql, params);
  return rows[0] || null;
}

module.exports = { pool, query, queryOne };

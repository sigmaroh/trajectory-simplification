#!/usr/bin/env node
/**
 * migrate.js — Lightweight SQL migration runner for PostgreSQL
 *
 * Usage:
 *   node webapp/migrations/migrate.js            # run pending migrations
 *   node webapp/migrations/migrate.js --status   # show applied / pending
 *   node webapp/migrations/migrate.js --rollback # not supported (append-only)
 *
 * Connection via env vars (or .env file if dotenv is installed):
 *   PGHOST     (default: localhost)
 *   PGPORT     (default: 5432)
 *   PGUSER     (default: postgres)
 *   PGPASSWORD (default: empty)
 *   PGDATABASE (default: trajectory_db)
 *
 * Migration files must be named NNN_description.sql (e.g. 001_initial_schema.sql)
 * and live in the same directory as this script.
 * They are applied in ascending numeric order, exactly once each.
 */

'use strict';

const fs   = require('fs');
const path = require('path');
const { Client } = require('pg');

// ── Try to load dotenv if available ──────────────────────────────────────────
try { require('dotenv').config(); } catch (_) { /* optional */ }

// ── DB connection ─────────────────────────────────────────────────────────────
function makeClient() {
  return new Client({
    host:     process.env.PGHOST     || 'localhost',
    port:     parseInt(process.env.PGPORT || '5432'),
    user:     process.env.PGUSER     || 'postgres',
    password: process.env.PGPASSWORD || 'root',
    database: process.env.PGDATABASE || 'trajectory_db',
  });
}

// ── Schema-migrations tracking table ─────────────────────────────────────────
const ENSURE_TABLE = `
  CREATE TABLE IF NOT EXISTS schema_migrations (
    version    VARCHAR(10)  PRIMARY KEY,   -- e.g. '001'
    filename   VARCHAR(256) NOT NULL,
    applied_at TIMESTAMPTZ  DEFAULT now()
  );
`;

// ── Helpers ───────────────────────────────────────────────────────────────────
function getMigrationFiles(dir) {
  return fs.readdirSync(dir)
    .filter(f => /^\d{3}_.*\.sql$/.test(f))
    .sort();
}

function getVersion(filename) {
  return filename.slice(0, 3);   // first three chars: '001', '002', …
}

async function getApplied(client) {
  const res = await client.query('SELECT version FROM schema_migrations ORDER BY version');
  return new Set(res.rows.map(r => r.version));
}

// ── Commands ──────────────────────────────────────────────────────────────────
async function runStatus() {
  const client = makeClient();
  await client.connect();
  await client.query(ENSURE_TABLE);

  const dir     = __dirname;
  const files   = getMigrationFiles(dir);
  const applied = await getApplied(client);

  console.log('\n  Version  Status     File');
  console.log('  ───────  ─────────  ────────────────────────────────────');
  for (const f of files) {
    const v      = getVersion(f);
    const status = applied.has(v) ? '✓ applied ' : '○ pending ';
    console.log(`  ${v}      ${status}  ${f}`);
  }
  console.log();

  await client.end();
}

async function runMigrate() {
  const client = makeClient();
  await client.connect();
  console.log(`\n  Connected to ${process.env.PGDATABASE || 'trajectory_db'} @ ${process.env.PGHOST || 'localhost'}:${process.env.PGPORT || 5432}`);

  await client.query(ENSURE_TABLE);

  const dir     = __dirname;
  const files   = getMigrationFiles(dir);
  const applied = await getApplied(client);

  const pending = files.filter(f => !applied.has(getVersion(f)));

  if (pending.length === 0) {
    console.log('  Nothing to migrate — all migrations already applied.\n');
    await client.end();
    return;
  }

  console.log(`  Found ${pending.length} pending migration(s):\n`);

  for (const f of pending) {
    const v    = getVersion(f);
    const sql  = fs.readFileSync(path.join(dir, f), 'utf8');
    const desc = f.replace(/\.sql$/, '');

    process.stdout.write(`  ▶ [${v}] ${desc} … `);
    try {
      await client.query('BEGIN');
      await client.query(sql);
      await client.query(
        'INSERT INTO schema_migrations (version, filename) VALUES ($1, $2)',
        [v, f]
      );
      await client.query('COMMIT');
      console.log('done');
    } catch (err) {
      await client.query('ROLLBACK');
      console.error(`\n  ✗ Migration ${f} failed:\n  ${err.message}\n`);
      await client.end();
      process.exit(1);
    }
  }

  console.log(`\n  ✓ ${pending.length} migration(s) applied successfully.\n`);
  await client.end();
}

// ── Entry point ───────────────────────────────────────────────────────────────
(async () => {
  const args = process.argv.slice(2);
  try {
    if (args.includes('--status')) {
      await runStatus();
    } else {
      await runMigrate();
    }
  } catch (err) {
    console.error('\n  Connection error:', err.message);
    console.error('  Make sure PostgreSQL is running and PGHOST/PGUSER/PGPASSWORD/PGDATABASE are set.\n');
    process.exit(1);
  }
})();

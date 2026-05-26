// Load .env before anything else (silently ignored if dotenv not installed)
try { require('dotenv').config(); } catch (_) {}

const express = require('express');
const path    = require('path');
const cors    = require('cors');

// Warm up the DB pool on startup
const { pool } = require('./db');
pool.query('SELECT 1').catch(err =>
  console.warn('  [db] PostgreSQL not reachable:', err.message)
);

const metricsRouter      = require('./routes/metrics');
const trajectoriesRouter = require('./routes/trajectories');
const pagesRouter        = require('./routes/pages');

const app  = express();

// Accept --port=XXXX or --port XXXX or PORT env var
const portArg = process.argv.find(a => a.startsWith('--port'));
const PORT = portArg
  ? parseInt(portArg.includes('=') ? portArg.split('=')[1] : process.argv[process.argv.indexOf(portArg) + 1])
  : parseInt(process.env.PORT || '3000');

app.set('view engine', 'ejs');
app.set('views', path.join(__dirname, 'views'));

app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

app.use('/api/metrics',      metricsRouter);
app.use('/api/trajectories', trajectoriesRouter);
app.use('/api/sync',         require('./routes/sync'));
app.use('/',                 pagesRouter);

const server = app.listen(PORT, () => {
  console.log(`\n  Trajectory Dashboard running at http://localhost:${PORT}`);
  console.log('  Dashboard : http://localhost:' + PORT + '/dashboard');
  console.log('  Plots     : http://localhost:' + PORT + '/plots');
  console.log('  Map       : http://localhost:' + PORT + '/map\n');
});

server.on('error', err => {
  if (err.code === 'EADDRINUSE') {
    console.error(`\n  Port ${PORT} is already in use.`);
    console.error(`  Run with a different port:  node server.js --port=3300\n`);
  } else {
    console.error(err);
  }
  process.exit(1);
});

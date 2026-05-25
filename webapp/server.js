const express = require('express');
const path    = require('path');
const cors    = require('cors');

const metricsRouter      = require('./routes/metrics');
const trajectoriesRouter = require('./routes/trajectories');

const app  = express();

// Accept --port=XXXX or --port XXXX or PORT env var
const portArg = process.argv.find(a => a.startsWith('--port'));
const PORT = portArg
  ? parseInt(portArg.includes('=') ? portArg.split('=')[1] : process.argv[process.argv.indexOf(portArg) + 1])
  : parseInt(process.env.PORT || '3000');

app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

app.use('/api/metrics',      metricsRouter);
app.use('/api/trajectories', trajectoriesRouter);

// root → dashboard
app.get('/', (_req, res) =>
  res.sendFile(path.join(__dirname, 'public', 'dashboard.html'))
);

const server = app.listen(PORT, () => {
  console.log(`\n  Trajectory Dashboard running at http://localhost:${PORT}`);
  console.log('  Dashboard : http://localhost:' + PORT + '/dashboard.html');
  console.log('  Plots     : http://localhost:' + PORT + '/plots.html');
  console.log('  Map       : http://localhost:' + PORT + '/map.html\n');
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

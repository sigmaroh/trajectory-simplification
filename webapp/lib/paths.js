'use strict';
const path = require('path');

/** Project root (parent of webapp/) */
const ROOT = path.join(__dirname, '..', '..');

const DEFAULTS = {
  trajectoriesIndex: path.join(ROOT, 'data', 'processed', 'trajectories_index.csv'),
  trajectoriesPoints: path.join(ROOT, 'data', 'processed', 'trajectories_points.csv'),
  trajectoryProperties: path.join(ROOT, 'data', 'processed', 'trajectory_properties.csv'),
  experimentResults: path.join(ROOT, 'results', 'experiment_results.csv'),
};

function resolvePath(filePath) {
  if (!filePath) return null;
  return path.isAbsolute(filePath) ? filePath : path.join(ROOT, filePath);
}

module.exports = { ROOT, DEFAULTS, resolvePath };

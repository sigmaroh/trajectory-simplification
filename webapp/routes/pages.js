'use strict';
/**
 * Page routes — render EJS views (replaces static .html files).
 */
const express = require('express');
const router = express.Router();

const DEFAULT_ALGOS = [
  { id: 'original',       label: 'Original',           color: '#6b7280' },
  { id: 'dp',             label: 'DP',                 color: '#3b82f6' },
  { id: 'vw',             label: 'VW',                 color: '#f97316' },
  { id: 'squish',         label: 'SQUISH',             color: '#a855f7' },
  { id: 'rw',             label: 'RW',                 color: '#ef4444' },
  { id: 'greedy_policy',  label: 'Greedy Policy (RL)', color: '#7c3aed' },
  { id: 'proposed',       label: 'Proposed',           color: '#111827' },
];

const COMPRESSION_RATIOS = [
  { value: 2,  label: '2.00x' },
  { value: 5,  label: '5.00x' },
  { value: 10, label: '10.00x' },
];

const BASE_MAPS = [
  { id: 'osm',   label: 'OpenStreetMap', default: true },
  { id: 'carto', label: 'Carto Light' },
  { id: 'dark',  label: 'Dark' },
  { id: 'esri',  label: 'Esri Grey' },
];

router.get('/', (_req, res) => res.redirect('/dashboard'));

router.get('/dashboard', (_req, res) => {
  res.render('dashboard', { title: 'Trajectory Metrics Dashboard' });
});

router.get('/plots', (_req, res) => {
  res.render('plots', { title: 'Plots — Trajectory Simplification' });
});

router.get('/map', (_req, res) => {
  res.render('map', {
    title: 'OSM Trajectory Comparison',
    defaultAlgorithms: DEFAULT_ALGOS,
    compressionRatios: COMPRESSION_RATIOS,
    baseMaps: BASE_MAPS,
  });
});

// Redirect legacy .html URLs
router.get('/dashboard.html', (_req, res) => res.redirect(301, '/dashboard'));
router.get('/plots.html', (_req, res) => res.redirect(301, '/plots'));
router.get('/map.html', (_req, res) => res.redirect(301, '/map'));

module.exports = router;

'use strict';
/**
 * OSM trajectory comparison map — dynamic version of trajectories_osm_comparison.html
 */

const CONFIG = window.MAP_CONFIG || {};
const DEFAULT_ALGOS = CONFIG.defaultAlgorithms || [
  { id: 'original', label: 'Original', color: '#6b7280' },
  { id: 'dp', label: 'DP', color: '#3b82f6' },
  { id: 'vw', label: 'VW', color: '#f97316' },
  { id: 'squish', label: 'SQUISH', color: '#a855f7' },
  { id: 'rw', label: 'RW', color: '#ef4444' },
  { id: 'greedy_policy', label: 'Greedy Policy (RL)', color: '#7c3aed' },
  { id: 'proposed', label: 'Proposed', color: '#111827' },
];

function trajLabel(tid) {
  return `T${String(tid).padStart(4, '0')}`;
}

function fmt(v) {
  if (v == null || v === '' || !Number.isFinite(Number(v))) return '—';
  const n = Number(v);
  if (Math.abs(n) >= 1000) return n.toFixed(1);
  if (Math.abs(n) >= 100) return n.toFixed(2);
  return n.toFixed(3);
}

// ─── map init (deferred until DOM ready) ─────────────────────────────────────
let map = null;
let currentTileLayer = null;
const tileLayers = {};

function buildTileLayers() {
  tileLayers.osm = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '© OpenStreetMap contributors', maxZoom: 19,
  });
  tileLayers.carto = L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
    attribution: '© OpenStreetMap & CARTO', maxZoom: 19,
  });
  tileLayers.dark = L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
    attribution: '© OpenStreetMap & CARTO', maxZoom: 19,
  });
  tileLayers.esri = L.tileLayer(
    'https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}',
    { attribution: 'Tiles © Esri', maxZoom: 16 }
  );
}

function setBasemap(key) {
  if (!map || !tileLayers[key]) return;
  if (currentTileLayer) map.removeLayer(currentTileLayer);
  currentTileLayer = tileLayers[key];
  currentTileLayer.addTo(map);
}

function initMap() {
  if (typeof L === 'undefined') {
    throw new Error('Leaflet failed to load — check network / CDN');
  }

  map = L.map('map', { zoomControl: false }).setView([39.9, 116.4], 11);
  L.control.zoom({ position: 'bottomright' }).addTo(map);
  L.control.scale().addTo(map);

  buildTileLayers();
  const defaultKey = (CONFIG.baseMaps || []).find(b => b.default)?.id || 'osm';
  setBasemap(defaultKey);

  const basemapSel = document.getElementById('basemap-select');
  if (basemapSel) {
    basemapSel.value = defaultKey;
    basemapSel.addEventListener('change', () => setBasemap(basemapSel.value));
  }

  requestAnimationFrame(() => map.invalidateSize());
  window.addEventListener('resize', () => map.invalidateSize());
}

// ─── state ────────────────────────────────────────────────────────────────────
let trajectories = [];
let layerRegistry = [];
let metricsCache = [];
let loading = false;
let loadToken = 0;

function setStatus(msg) {
  document.getElementById('map-status').textContent = msg || '';
}

function clearLayers() {
  if (!map) return;
  layerRegistry.forEach(reg => {
    map.removeLayer(reg.line);
    reg.markers.forEach(m => map.removeLayer(m));
  });
  layerRegistry = [];
}

function layerKey(ratioLbl, trajLbl, algo) {
  return `${ratioLbl} | ${trajLbl} - ${algo}`;
}

// ─── data loading ─────────────────────────────────────────────────────────────
async function loadTrajectories() {
  const res = await fetch('/api/trajectories/list?max=200');
  if (!res.ok) throw new Error(`Trajectory list failed (${res.status})`);
  trajectories = await res.json();
  const sel = document.getElementById('trajectory-select');
  sel.innerHTML = '';
  if (!trajectories.length) {
    sel.add(new Option('No trajectories', ''));
    return;
  }
  trajectories.forEach(t => {
    sel.add(new Option(`${trajLabel(t.trajectory_id)} — ${t.num_points} pts`, t.trajectory_id));
  });
  sel.value = String(trajectories[0].trajectory_id);
}

async function fetchTrajectoryPaths(tid, cr, algo) {
  const params = new URLSearchParams({
    trajectory_id: tid,
    algorithm: algo,
    compression_ratio: cr,
  });
  const res = await fetch(`/api/trajectories/points?${params}`);
  const data = await res.json();
  if (!res.ok || data.error) throw new Error(data.error || `HTTP ${res.status}`);
  return data;
}

async function fetchMetricsForTrajectory(tid) {
  const res = await fetch('/api/metrics/results');
  if (!res.ok) return [];
  const rows = await res.json();
  return rows.filter(r => String(r.trajectory_id) === String(tid));
}

function addLayerFromData(algoDef, data, ratioLbl, trajLbl, cr) {
  const algo = algoDef.id;
  const { original, simplified } = data;
  const pts = algo === 'original' ? original : simplified;
  if (!pts || pts.length < 2) return null;

  const latlngs = pts.map(p => [p.lat, p.lon]);
  const label = layerKey(ratioLbl, trajLbl, algo);

  const line = L.polyline(latlngs, {
    color: algoDef.color,
    weight: algo === 'original' ? 3 : 2,
    opacity: algo === 'original' ? 0.85 : 0.8,
    dashArray: algo === 'proposed' ? '6 4' : null,
  }).bindTooltip(`${algoDef.label} (${ratioLbl}) — ${pts.length} pts`);

  const markers = [
    L.circleMarker(latlngs[0], { radius: 4, color: '#fff', weight: 1, fillColor: algoDef.color, fillOpacity: 1 }),
    L.circleMarker(latlngs[latlngs.length - 1], { radius: 4, color: '#fff', weight: 1, fillColor: '#111', fillOpacity: 0.8 }),
  ];

  const match = metricsCache.find(r =>
    r.algorithm === algo && Math.abs(parseFloat(r.compression_ratio) - cr) < cr * 0.08
  );

  const reg = {
    key: label,
    algo,
    algoDef,
    line,
    markers,
    visible: true,
    metrics: match || null,
    ratioLbl,
    trajLbl,
    latlngs,
  };

  line.addTo(map);
  markers.forEach(m => m.addTo(map));
  layerRegistry.push(reg);
  return reg;
}

function fitToLayers() {
  const latlngs = layerRegistry.flatMap(r => r.latlngs);
  if (!latlngs.length) return;
  map.fitBounds(L.latLngBounds(latlngs), { padding: [40, 40] });
}

// ─── render layers for current selection ──────────────────────────────────────
async function loadComparison() {
  if (!map || loading) return;

  const tid = document.getElementById('trajectory-select').value;
  const ratioLbl = document.getElementById('ratio-select').value;
  const cr = parseFloat(ratioLbl);

  if (!tid) {
    setStatus('No trajectory selected.');
    return;
  }

  const token = ++loadToken;
  loading = true;
  setStatus('Loading original path…');
  clearLayers();

  const trajLbl = trajLabel(tid);

  try {
    metricsCache = await fetchMetricsForTrajectory(tid);
    if (token !== loadToken) return;

    // 1) Load original immediately (no Python) so map shows something fast
    const originalDef = DEFAULT_ALGOS.find(a => a.id === 'original') || DEFAULT_ALGOS[0];
    try {
      const origData = await fetchTrajectoryPaths(tid, cr, originalDef.id);
      if (token !== loadToken) return;
      addLayerFromData(originalDef, origData, ratioLbl, trajLbl, cr);
      fitToLayers();
      buildMetricsTable();
      refreshLayerList();
      setStatus(`Loaded original — simplifying other algorithms…`);
    } catch (err) {
      console.warn('Original path failed:', err.message);
    }

    // 2) Load remaining algorithms in parallel
    const others = DEFAULT_ALGOS.filter(a => a.id !== 'original');
    const results = await Promise.allSettled(
      others.map(async (algoDef) => {
        const data = await fetchTrajectoryPaths(tid, cr, algoDef.id);
        return { algoDef, data };
      })
    );

    if (token !== loadToken) return;

    for (const result of results) {
      if (result.status !== 'fulfilled') {
        console.warn('Algo load failed:', result.reason?.message || result.reason);
        continue;
      }
      addLayerFromData(result.value.algoDef, result.value.data, ratioLbl, trajLbl, cr);
    }

    fitToLayers();
    buildMetricsTable();
    refreshLayerList();
    applyVisibility();

    const n = layerRegistry.length;
    setStatus(n
      ? `Loaded ${n} algorithm${n > 1 ? 's' : ''} for ${trajLbl} at ${ratioLbl}`
      : 'No paths loaded — check server logs / Python venv');
  } catch (err) {
    setStatus(`Error: ${err.message}`);
    console.error(err);
  } finally {
    if (token === loadToken) loading = false;
  }
}

// ─── metrics overlay table ────────────────────────────────────────────────────
function buildMetricsTable() {
  const tbody = document.getElementById('metrics-overlay-body');
  tbody.innerHTML = '';

  const tid = document.getElementById('trajectory-select').value;
  if (!tid) return;
  const trajLbl = trajLabel(tid);

  const ratios = [...new Set((CONFIG.compressionRatios || []).map(r => r.label))];
  if (!ratios.length) ratios.push('2.00x', '5.00x', '10.00x');

  for (const rl of ratios) {
    const crVal = parseFloat(rl);
    for (const algoDef of DEFAULT_ALGOS) {
      const match = metricsCache.find(r =>
        r.algorithm === algoDef.id && Math.abs(parseFloat(r.compression_ratio) - crVal) < crVal * 0.08
      );
      if (!match) continue;

      const tr = document.createElement('tr');
      tr.dataset.ratio = rl;
      tr.dataset.traj = trajLbl;
      tr.innerHTML = `
        <td>${rl}</td>
        <td>${trajLbl}</td>
        <td style="font-weight:600">${algoDef.label}</td>
        <td class="num">${fmt(match.hausdorff_distance)}</td>
        <td class="num">${fmt(match.frechet_distance)}</td>
        <td class="num">${fmt(match.average_pte)}</td>`;
      tbody.appendChild(tr);
    }
  }

  if (!tbody.children.length) {
    tbody.innerHTML = '<tr><td colspan="6" class="osm-muted">No stored metrics for this trajectory.</td></tr>';
  }
}

// ─── layer list + visibility ──────────────────────────────────────────────────
function refreshLayerList() {
  const container = document.getElementById('layer-list-card');
  const ratioLbl = document.getElementById('ratio-select').value;
  const tid = document.getElementById('trajectory-select').value;
  if (!tid) {
    container.innerHTML = '<div class="osm-muted">Select a trajectory.</div>';
    return;
  }
  const trajLbl = trajLabel(tid);

  const items = layerRegistry.filter(r => r.ratioLbl === ratioLbl && r.trajLbl === trajLbl);
  if (!items.length) {
    container.innerHTML = '<div class="osm-muted">No layers yet — loading…</div>';
    return;
  }

  container.innerHTML = items.map(reg => `
    <label>
      <input type="checkbox" class="layer-card-checkbox" data-key="${reg.key.replace(/"/g, '&quot;')}" ${reg.visible ? 'checked' : ''}>
      <span class="layer-swatch" style="background:${reg.algoDef.color}"></span>
      <span>${reg.algoDef.label}</span>
    </label>`).join('');

  container.querySelectorAll('.layer-card-checkbox').forEach(cb => {
    cb.addEventListener('change', () => {
      const reg = layerRegistry.find(r => r.key === cb.dataset.key);
      if (!reg) return;
      reg.visible = cb.checked;
      if (reg.visible) {
        reg.line.addTo(map);
        reg.markers.forEach(m => m.addTo(map));
      } else {
        map.removeLayer(reg.line);
        reg.markers.forEach(m => map.removeLayer(m));
      }
    });
  });
}

function applyVisibility() {
  const ratioLbl = document.getElementById('ratio-select').value;
  const tid = document.getElementById('trajectory-select').value;
  if (!tid) return;
  const trajLbl = trajLabel(tid);

  layerRegistry.forEach(reg => {
    const show = reg.ratioLbl === ratioLbl && reg.trajLbl === trajLbl && reg.visible;
    if (show) {
      if (!map.hasLayer(reg.line)) reg.line.addTo(map);
      reg.markers.forEach(m => { if (!map.hasLayer(m)) m.addTo(map); });
    } else {
      if (map.hasLayer(reg.line)) map.removeLayer(reg.line);
      reg.markers.forEach(m => { if (map.hasLayer(m)) map.removeLayer(m); });
    }
  });

  document.querySelectorAll('#metrics-overlay-body tr').forEach(tr => {
    if (!tr.dataset.ratio) return;
    const show = tr.dataset.ratio === ratioLbl && tr.dataset.traj === trajLbl;
    tr.classList.toggle('hidden-row', !show);
  });
}

// ─── init ─────────────────────────────────────────────────────────────────────
async function init() {
  try {
    initMap();
    await loadTrajectories();
    document.getElementById('ratio-select').addEventListener('change', () => loadComparison());
    document.getElementById('trajectory-select').addEventListener('change', () => loadComparison());
    await loadComparison();
  } catch (err) {
    setStatus(`Init failed: ${err.message}`);
    console.error(err);
  }
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}

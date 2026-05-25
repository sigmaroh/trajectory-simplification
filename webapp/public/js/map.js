// ─── algorithm colours (matches export_osm_json_map.py) ──────────────────────
const ALGO_COLOR = {
  original:      '#6b7280',
  dp:            '#3b82f6',
  vw:            '#f97316',
  squish:        '#a855f7',
  rw:            '#ef4444',
  greedy_policy: '#7c3aed',
  rl_dqn:        '#94a3b8',
  proposed:      '#111827',
  us:            '#14b8a6',
  at:            '#64748b',
};
const ALGO_LABEL = {
  original:'Original', dp:'DP', vw:'VW', squish:'SQUISH',
  rw:'RW', greedy_policy:'Greedy Policy', proposed:'Proposed',
  us:'Uniform', at:'Adaptive Threshold', rl_dqn:'RL DQN',
};
function algoColor(a) { return ALGO_COLOR[a] || '#000'; }
function algoLabel(a) { return ALGO_LABEL[a] || a; }

// ─── tile layer definitions ───────────────────────────────────────────────────
const TILE_DEFS = {
  osm:   { url:'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
            opts:{ attribution:'© OpenStreetMap contributors', maxZoom:19 } },
  carto: { url:'https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png',
            opts:{ attribution:'© OpenStreetMap & CARTO', maxZoom:19 } },
  dark:  { url:'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png',
            opts:{ attribution:'© OpenStreetMap & CARTO', maxZoom:19 } },
  esri:  { url:'https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}',
            opts:{ attribution:'Tiles © Esri', maxZoom:16 } },
};

// ─── map init ─────────────────────────────────────────────────────────────────
const map = L.map('map').setView([39.9, 116.4], 11);
let currentTileLayer = L.tileLayer(TILE_DEFS.osm.url, TILE_DEFS.osm.opts).addTo(map);

function setTile(key, btn) {
  map.removeLayer(currentTileLayer);
  const def = TILE_DEFS[key] || TILE_DEFS.osm;
  currentTileLayer = L.tileLayer(def.url, def.opts).addTo(map);
  document.querySelectorAll('.tile-btn').forEach(b => b.classList.remove('active'));
  if (btn) btn.classList.add('active');
}

// ─── mode toggle ──────────────────────────────────────────────────────────────
let currentMode = 'simple';

function switchMode(mode) {
  currentMode = mode;
  clearMap();
  document.getElementById('tab-simple').classList.toggle('active', mode === 'simple');
  document.getElementById('tab-osm').classList.toggle('active',    mode === 'osm');
  document.getElementById('bar-simple').classList.toggle('hidden',  mode !== 'simple');
  document.getElementById('bar-osm').classList.toggle('hidden',     mode !== 'osm');
  document.getElementById('sidebar-simple').classList.toggle('hidden', mode !== 'simple');
  // OSM panels are inside map — toggled by loadOsmComparison / clearMap
}

// ─── shared layer registry ────────────────────────────────────────────────────
let activeLeafletLayers = [];

function clearMap() {
  activeLeafletLayers.forEach(l => map.removeLayer(l));
  activeLeafletLayers = [];
  document.getElementById('map-status').textContent  = '';
  document.getElementById('map-status') && (document.getElementById('osm-status').textContent = '');
  // simple mode UI
  document.getElementById('traj-info').innerHTML   = 'Select a trajectory.';
  document.getElementById('metric-info').innerHTML = '—';
  // OSM panels
  ['osm-layer-panel','osm-metric-panel','osm-legend-panel'].forEach(
    id => document.getElementById(id).classList.add('hidden')
  );
  document.getElementById('osm-layer-list').innerHTML  = '';
  document.getElementById('osm-metric-body').innerHTML = '';
  document.getElementById('osm-legend-body').innerHTML = '';
}

// ═══════════════════════════════════════════════════════════════════════════════
// SIMPLE MODE
// ═══════════════════════════════════════════════════════════════════════════════

async function initFilters() {
  const data = await fetch('/api/metrics/filters').then(r => r.json());
  const fAlgo = document.getElementById('f-algo');
  data.algorithms.forEach(a => fAlgo.add(new Option(algoLabel(a), a)));
  if ([...fAlgo.options].some(o => o.value === 'proposed')) fAlgo.value = 'proposed';

  const fUser = document.getElementById('f-user');
  data.users.slice(0, 80).forEach(u => fUser.add(new Option(`User ${u}`, u)));

  // also populate OSM trajectory list (all trajs)
  await loadTrajectoryList();
  await populateOsmTrajSelect();
}

async function loadTrajectoryList() {
  const user = document.getElementById('f-user').value;
  document.getElementById('traj-list-loading').style.display = 'block';
  document.getElementById('traj-list').innerHTML = '';

  const params = new URLSearchParams({ max: 60 });
  if (user) params.set('user_id', user);
  const trajs = await fetch(`/api/trajectories/list?${params}`).then(r => r.json());

  document.getElementById('traj-list-loading').style.display = 'none';

  const fTid = document.getElementById('f-tid');
  fTid.innerHTML = '<option value="">Select trajectory</option>';
  trajs.forEach(t => fTid.add(new Option(`#${t.trajectory_id} — ${t.num_points} pts`, t.trajectory_id)));

  const listEl = document.getElementById('traj-list');
  trajs.forEach(t => {
    const div = document.createElement('div');
    div.style.cssText = 'padding:.35rem .5rem;cursor:pointer;border-radius:5px;border-bottom:1px solid #f1f5f9;';
    div.innerHTML = `<b>#${t.trajectory_id}</b> &nbsp;user ${t.user_id} &nbsp;<span style="color:#64748b">${t.num_points} pts</span>`;
    div.onmouseover = () => div.style.background = '#eff6ff';
    div.onmouseleave = () => div.style.background = '';
    div.onclick = () => { document.getElementById('f-tid').value = t.trajectory_id; loadOnMap(); };
    listEl.appendChild(div);
  });
  if (!trajs.length) listEl.innerHTML = '<div style="color:#94a3b8;padding:.5rem">No trajectories found.</div>';
}

async function loadOnMap() {
  const tid  = document.getElementById('f-tid').value;
  const algo = document.getElementById('f-algo').value;
  const cr   = document.getElementById('f-cr').value;
  if (!tid)  { alert('Please select a trajectory.'); return; }
  if (!algo) { alert('Please select an algorithm.');  return; }

  clearMap();
  document.getElementById('map-status').textContent = 'Loading …';

  try {
    const params = new URLSearchParams({ trajectory_id: tid, algorithm: algo, compression_ratio: cr });
    const data = await fetch(`/api/trajectories/points?${params}`).then(r => r.json());
    if (data.error) throw new Error(data.error);
    const { original, simplified, meta } = data;

    if (original.length > 1) {
      const l = L.polyline(original.map(p => [p.lat, p.lon]),
        { color:'#3b82f6', weight:2, opacity:0.6 }).addTo(map);
      activeLeafletLayers.push(l);
    }
    if (simplified.length > 1) {
      const l = L.polyline(simplified.map(p => [p.lat, p.lon]),
        { color:'#ef4444', weight:3, opacity:0.9, dashArray:'8 5' }).addTo(map);
      activeLeafletLayers.push(l);
    }
    simplified.forEach((p, i) => {
      const m = L.circleMarker([p.lat, p.lon],
        { radius:5, color:'#fff', weight:1.5, fillColor:'#ef4444', fillOpacity:0.9 })
        .bindPopup(`<b>Point ${i+1}</b><br>${p.lat.toFixed(5)}, ${p.lon.toFixed(5)}`);
      m.addTo(map);
      activeLeafletLayers.push(m);
    });

    if (original.length) map.fitBounds(L.latLngBounds(original.map(p=>[p.lat,p.lon])), {padding:[30,30]});

    document.getElementById('traj-info').innerHTML = `
      <table style="width:100%;border-collapse:collapse">
        <tr><td style="color:#64748b">ID</td><td><b>${meta.trajectory_id}</b></td></tr>
        <tr><td style="color:#64748b">User</td><td>${meta.user_id}</td></tr>
        <tr><td style="color:#64748b">Input pts</td><td><b>${meta.n_original}</b></td></tr>
        <tr><td style="color:#64748b">Output pts</td><td><b>${meta.n_simplified}</b></td></tr>
        <tr><td style="color:#64748b">Algorithm</td><td><b>${algo}</b></td></tr>
        <tr><td style="color:#64748b">Target CR</td><td>${cr}×</td></tr>
        <tr><td style="color:#64748b">Actual CR</td><td>${meta.n_simplified ? (meta.n_original/meta.n_simplified).toFixed(2)+'×' : '—'}</td></tr>
      </table>`;

    // fetch metrics from CSV
    const mRows = await fetch(`/api/metrics/results?algorithm=${algo}&compression_ratio=${cr}`)
      .then(r => r.json());
    const match = mRows.find(r => String(r.trajectory_id) === String(tid));
    if (match) {
      const fields = [
        ['Hausdorff (m)','hausdorff_distance'],['APTE (m)','average_pte'],
        ['Fréchet (m)','frechet_distance'],['Turn Pres.','turn_preservation'],
        ['Stop Pres.','stop_preservation'],['Runtime (s)','runtime_seconds'],['Mem (MB)','memory_mb'],
      ];
      let html = '<table style="width:100%;border-collapse:collapse">';
      fields.forEach(([lbl,k]) => {
        const v = match[k]; if (v == null || v === '') return;
        const isPct = k.includes('preservation');
        const disp  = isPct
          ? `<b style="color:${parseFloat(v)>.7?'#15803d':'#d97706'}">${(parseFloat(v)*100).toFixed(1)}%</b>`
          : `<b>${parseFloat(v).toFixed(3)}</b>`;
        html += `<tr><td style="color:#64748b">${lbl}</td><td>${disp}</td></tr>`;
      });
      document.getElementById('metric-info').innerHTML = html + '</table>';
    } else {
      document.getElementById('metric-info').innerHTML =
        '<span style="color:#94a3b8">No stored metrics for this selection.</span>';
    }
    document.getElementById('map-status').textContent =
      `${meta.n_original} → ${meta.n_simplified} pts (${algo} ${cr}×)`;
  } catch(e) {
    document.getElementById('map-status').textContent = '';
    document.getElementById('traj-info').innerHTML = `<span class="error-msg">Error: ${e.message}</span>`;
    console.error(e);
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// OSM COMPARISON MODE
// ═══════════════════════════════════════════════════════════════════════════════

async function populateOsmTrajSelect() {
  const trajs = await fetch('/api/trajectories/list?max=100').then(r => r.json());
  const sel = document.getElementById('osm-f-tid');
  sel.innerHTML = '<option value="">Select trajectory</option>';
  trajs.forEach(t => sel.add(new Option(`#${t.trajectory_id} — ${t.num_points} pts (user ${t.user_id})`, t.trajectory_id)));
}

function fmt(v) {
  if (v == null || !Number.isFinite(Number(v))) return '—';
  const n = Number(v);
  return Math.abs(n) >= 1000 ? n.toFixed(1) : Math.abs(n) >= 100 ? n.toFixed(2) : n.toFixed(3);
}

// OSM comparison layer registry
let osmLayerReg = [];

async function loadOsmComparison() {
  const tid     = document.getElementById('osm-f-tid').value;
  const cr      = parseFloat(document.getElementById('osm-f-cr').value);
  const algoSel = document.getElementById('osm-f-algos');
  const algos   = [...algoSel.options].filter(o => o.selected).map(o => o.value);

  if (!tid) { alert('Please select a trajectory.'); return; }

  clearMap();
  document.getElementById('osm-status').textContent = 'Loading …';
  osmLayerReg = [];

  const metricRows = [];
  let bounds = null;

  for (const algo of algos) {
    try {
      const params = new URLSearchParams({ trajectory_id: tid, algorithm: algo, compression_ratio: cr });
      const data = await fetch(`/api/trajectories/points?${params}`).then(r => r.json());
      if (data.error || !data.original) continue;

      const { original, simplified, meta } = data;
      const pts = (algo === 'original' ? original : simplified);
      if (!pts || pts.length < 2) continue;

      const latlngs = pts.map(p => [p.lat, p.lon]);
      const color   = algoColor(algo);
      const weight  = algo === 'original' ? 3 : 2;
      const opacity = algo === 'original' ? 0.85 : 0.8;
      const dash    = algo === 'original' ? null : (algo === 'proposed' ? '6 4' : null);

      const line = L.polyline(latlngs, { color, weight, opacity, dashArray: dash })
        .bindTooltip(`${algoLabel(algo)} (${cr}×)  ${pts.length} pts`)
        .addTo(map);
      activeLeafletLayers.push(line);

      // start / end markers
      const sm = L.circleMarker(latlngs[0],  { radius:4, color:'#fff', weight:1, fillColor:color, fillOpacity:1 }).addTo(map);
      const em = L.circleMarker(latlngs[latlngs.length-1], { radius:4, color:'#fff', weight:1, fillColor:'#111', fillOpacity:0.8 }).addTo(map);
      activeLeafletLayers.push(sm, em);

      if (!bounds) bounds = L.latLngBounds(latlngs);
      else bounds.extend(latlngs);

      // fetch stored metrics
      const mRows = await fetch(`/api/metrics/results?algorithm=${algo}&compression_ratio=${cr}`)
        .then(r => r.json());
      const match = mRows.find(r => String(r.trajectory_id) === String(tid));

      osmLayerReg.push({ algo, color, line, visible: true, meta, match });
      if (match) metricRows.push({ algo, cr, match });

    } catch(e) { console.warn(algo, e.message); }
  }

  if (bounds) map.fitBounds(bounds, { padding:[30,30] });

  // layer panel
  const layerList = document.getElementById('osm-layer-list');
  layerList.innerHTML = '';
  osmLayerReg.forEach(reg => {
    const row = document.createElement('label');
    row.className = 'layer-row';
    const cb = document.createElement('input');
    cb.type = 'checkbox'; cb.checked = true;
    cb.onchange = () => {
      reg.visible = cb.checked;
      cb.checked ? reg.line.addTo(map) : map.removeLayer(reg.line);
      refreshOsmMetrics();
    };
    const sw = document.createElement('div');
    sw.className = 'swatch'; sw.style.background = reg.color;
    const txt = document.createElement('span');
    txt.textContent = `${algoLabel(reg.algo)}  ${reg.meta?.n_simplified ?? '?'} pts`;
    row.append(cb, sw, txt);
    layerList.appendChild(row);
  });

  // legend panel
  const legendBody = document.getElementById('osm-legend-body');
  legendBody.innerHTML = '';
  const seen = new Set();
  osmLayerReg.forEach(reg => {
    if (seen.has(reg.algo)) return; seen.add(reg.algo);
    const row = document.createElement('div');
    row.className = 'layer-row';
    const sw = document.createElement('div'); sw.className='swatch'; sw.style.background=reg.color;
    const txt = document.createElement('span'); txt.textContent = algoLabel(reg.algo);
    row.append(sw, txt); legendBody.appendChild(row);
  });

  refreshOsmMetrics();

  ['osm-layer-panel','osm-metric-panel','osm-legend-panel'].forEach(
    id => document.getElementById(id).classList.remove('hidden')
  );

  const n = osmLayerReg.length;
  document.getElementById('osm-status').textContent =
    n ? `Loaded ${n} algorithm${n>1?'s':''} for trajectory #${tid} at ${cr}×` : 'No data loaded.';
}

function refreshOsmMetrics() {
  const tbody = document.getElementById('osm-metric-body');
  tbody.innerHTML = '';
  osmLayerReg.forEach(reg => {
    if (!reg.visible || !reg.match) return;
    const m = reg.match;
    const tr = document.createElement('tr');
    const pct = v => (v == null || v === '') ? '—'
      : `<span style="color:${parseFloat(v)>.7?'#15803d':'#d97706'};font-weight:600">${(parseFloat(v)*100).toFixed(1)}%</span>`;
    tr.innerHTML = `
      <td><span style="display:inline-flex;align-items:center;gap:5px">
        <span style="width:10px;height:10px;background:${reg.color};border-radius:2px;display:inline-block"></span>
        ${algoLabel(reg.algo)}</span></td>
      <td>${parseFloat(m.compression_ratio).toFixed(1)}×</td>
      <td>${fmt(m.hausdorff_distance)}</td>
      <td>${fmt(m.frechet_distance)}</td>
      <td>${fmt(m.average_pte)}</td>
      <td>${fmt(m.ped)}</td>
      <td>${fmt(m.sed)}</td>
      <td>${fmt(m.dad)}</td>
      <td>${fmt(m.sad)}</td>
      <td>${fmt(m.issd)}</td>`;
    tbody.appendChild(tr);
  });
  // if no stored metrics, show note
  if (!tbody.children.length) {
    tbody.innerHTML = '<tr><td colspan="10" style="color:#94a3b8;padding:.5rem">No stored metrics for this selection — run experiments first.</td></tr>';
  }
}

// ─── init ─────────────────────────────────────────────────────────────────────
initFilters();

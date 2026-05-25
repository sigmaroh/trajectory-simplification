// ─── colour palette ──────────────────────────────────────────────────────────
const ALGO_COLORS = {
  dp:             '#FF6600',
  us:             '#3498DB',
  at:             '#1ABC9C',
  vw:             '#2ECC71',
  squish:         '#E91E63',
  rw:             '#9B59B6',
  greedy_policy:  '#FF5722',
  rl_dqn:         '#7F8C8D',
  proposed:       '#212121',
};
function algoColor(a) { return ALGO_COLORS[a] || '#94a3b8'; }

// ─── chart registry ───────────────────────────────────────────────────────────
const CHARTS = {};
function makeChart(id, type, labels, datasets, opts = {}) {
  if (CHARTS[id]) CHARTS[id].destroy();
  const ctx = document.getElementById(id);
  if (!ctx) return;
  CHARTS[id] = new Chart(ctx, {
    type,
    data: { labels, datasets },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { position: 'top', labels: { boxWidth: 12, font: { size: 11 } } } },
      scales: {
        x: { ticks: { font: { size: 11 } } },
        y: { ticks: { font: { size: 11 } }, beginAtZero: false },
      },
      ...opts,
    },
  });
}

// ─── populate filter selects ──────────────────────────────────────────────────
async function loadFilters() {
  const r = await fetch('/api/metrics/filters').then(r => r.json());

  const fAlgo = document.getElementById('f-algo');
  r.algorithms.forEach(a => { fAlgo.add(new Option(a, a)); });

  const fCR = document.getElementById('f-cr');
  r.compressionRatios.forEach(c => { fCR.add(new Option(`${c}×`, c)); });

  const fUser = document.getElementById('f-user');
  r.users.slice(0, 100).forEach(u => { fUser.add(new Option(`User ${u}`, u)); });
}

// ─── fetch one metric comparison ─────────────────────────────────────────────
async function fetchComparison(metric, algoFilter) {
  const params = new URLSearchParams({ metric });
  if (algoFilter) params.set('algorithms', algoFilter);
  return fetch(`/api/metrics/comparison?${params}`).then(r => r.json());
}

// ─── build line datasets ──────────────────────────────────────────────────────
function buildDatasets(data) {
  return Object.entries(data.series).map(([algo, values]) => ({
    label: algo,
    data: values,
    borderColor: algoColor(algo),
    backgroundColor: algoColor(algo) + '22',
    borderWidth: 2,
    pointRadius: 4,
    tension: 0.2,
  }));
}

// ─── stat cards ───────────────────────────────────────────────────────────────
async function updateStatCards(algoFilter, userFilter) {
  const params = new URLSearchParams();
  if (algoFilter) params.set('algorithm', algoFilter);
  if (userFilter) params.set('user_id', userFilter);
  const summary = await fetch(`/api/metrics/summary?${params}`).then(r => r.json());

  const algos = [...new Set(summary.map(r => r.algorithm))];
  document.getElementById('s-algos').textContent = algos.length || '—';
  document.getElementById('s-runs').textContent  = summary.reduce((s,r) => s + (r.count||0), 0) || '—';

  // best hausdorff
  const withH = summary.filter(r => r.hausdorff_distance_mean != null);
  if (withH.length) {
    const b = withH.reduce((a,b) => a.hausdorff_distance_mean < b.hausdorff_distance_mean ? a : b);
    document.getElementById('s-best-h').textContent      = `${b.hausdorff_distance_mean.toFixed(0)} m`;
    document.getElementById('s-best-h-algo').textContent = b.algorithm;
  }

  // best fréchet
  const withFr = summary.filter(r => r.frechet_distance_mean != null);
  if (withFr.length) {
    const b = withFr.reduce((a,b) => a.frechet_distance_mean < b.frechet_distance_mean ? a : b);
    document.getElementById('s-best-fr').textContent      = `${b.frechet_distance_mean.toFixed(0)} m`;
    document.getElementById('s-best-fr-algo').textContent = b.algorithm;
  }

  // best stop preservation
  const withSP = summary.filter(r => r.stop_preservation_mean != null);
  if (withSP.length) {
    const b = withSP.reduce((a,b) => a.stop_preservation_mean > b.stop_preservation_mean ? a : b);
    document.getElementById('s-best-sp').textContent      = `${(b.stop_preservation_mean*100).toFixed(1)}%`;
    document.getElementById('s-best-sp-algo').textContent = b.algorithm;
  }

  // fastest
  const withRT = summary.filter(r => r.runtime_seconds_mean != null);
  if (withRT.length) {
    const b = withRT.reduce((a,b) => a.runtime_seconds_mean < b.runtime_seconds_mean ? a : b);
    document.getElementById('s-fastest').textContent      = `${b.runtime_seconds_mean.toFixed(3)}s`;
    document.getElementById('s-fastest-algo').textContent = b.algorithm;
  }
}

// ─── summary table ────────────────────────────────────────────────────────────
function fmt(v, dp = 2) {
  if (v == null) return '<span style="color:#94a3b8">—</span>';
  return parseFloat(v).toFixed(dp);
}
function pct(v) {
  if (v == null) return '<span style="color:#94a3b8">—</span>';
  const p = parseFloat(v) * 100;
  const color = p > 80 ? '#15803d' : p > 50 ? '#d97706' : '#dc2626';
  return `<span style="color:${color};font-weight:600">${p.toFixed(1)}%</span>`;
}

async function renderTable(algoFilter, userFilter) {
  const loading = document.getElementById('table-loading');
  const wrap    = document.getElementById('summary-table');
  loading.style.display = 'block';
  wrap.style.display    = 'none';

  const params = new URLSearchParams();
  if (algoFilter) params.set('algorithm', algoFilter);
  if (userFilter) params.set('user_id', userFilter);
  const rows = await fetch(`/api/metrics/summary?${params}`).then(r => r.json());

  const cols = [
    { key: 'algorithm',                   label: 'Algorithm' },
    { key: 'compression_ratio',           label: 'CR' },
    { key: 'count',                       label: 'Runs' },
    { key: 'input_points_mean',           label: 'Input pts' },
    { key: 'output_points_mean',          label: 'Output pts' },
    { key: 'hausdorff_distance_mean',     label: 'Hausdorff (m)' },
    { key: 'frechet_distance_mean',       label: 'Fréchet (m)' },
    { key: 'average_pte_mean',            label: 'APTE (m)' },
    { key: 'turn_preservation_mean',      label: 'Turn Pres.' },
    { key: 'stop_preservation_mean',      label: 'Stop Pres.' },
    { key: 'runtime_seconds_mean',        label: 'Runtime (s)' },
    { key: 'throughput_traj_per_sec_mean',label: 'Throughput' },
    { key: 'memory_mb_mean',              label: 'Mem (MB)' },
  ];

  let html = '<table><thead><tr>';
  cols.forEach(c => { html += `<th>${c.label}</th>`; });
  html += '</tr></thead><tbody>';

  rows.forEach(row => {
    html += '<tr>';
    cols.forEach(c => {
      const v = row[c.key];
      if (c.key === 'algorithm')
        html += `<td><span class="badge badge-blue">${v}</span></td>`;
      else if (c.key === 'compression_ratio')
        html += `<td>${parseFloat(v).toFixed(1)}×</td>`;
      else if (c.key === 'count')
        html += `<td>${v}</td>`;
      else if (c.key === 'turn_preservation_mean' || c.key === 'stop_preservation_mean')
        html += `<td>${pct(v)}</td>`;
      else if (c.key === 'input_points_mean' || c.key === 'output_points_mean')
        html += `<td>${fmt(v, 0)}</td>`;
      else
        html += `<td>${fmt(v, 3)}</td>`;
    });
    html += '</tr>';
  });

  html += '</tbody></table>';
  wrap.innerHTML = html;
  loading.style.display = 'none';
  wrap.style.display    = 'block';
}

// ─── render all charts ────────────────────────────────────────────────────────
async function renderCharts(algoFilter) {
  // fetch all 6 metrics in parallel
  const [h, fr, sp, tp, apte, rt] = await Promise.all([
    fetchComparison('hausdorff_distance',  algoFilter),
    fetchComparison('frechet_distance',    algoFilter),
    fetchComparison('stop_preservation',   algoFilter),
    fetchComparison('turn_preservation',   algoFilter),
    fetchComparison('average_pte',         algoFilter),
    fetchComparison('runtime_seconds',     algoFilter),
  ]);

  const labels = h.compressionRatios.map(c => `${c}×`);

  // row 1: geometric error
  makeChart('chart-hausdorff', 'line', labels, buildDatasets(h));
  makeChart('chart-frechet',   'line', labels, buildDatasets(fr));

  // row 2: semantic
  makeChart('chart-stop', 'line', labels, buildDatasets(sp));
  makeChart('chart-turn', 'line', labels, buildDatasets(tp));

  // row 3: apte + runtime
  makeChart('chart-apte',    'line', labels, buildDatasets(apte));
  makeChart('chart-runtime', 'bar',  labels,
    buildDatasets(rt).map(d => ({
      ...d, type: 'bar', backgroundColor: algoColor(d.label) + '99',
    }))
  );
}

// ─── filter wiring ────────────────────────────────────────────────────────────
function getFilters() {
  return {
    algo: document.getElementById('f-algo').value,
    user: document.getElementById('f-user').value,
    cr:   document.getElementById('f-cr').value,
  };
}

async function applyFilters() {
  const { algo, user } = getFilters();
  await Promise.all([
    renderCharts(algo),
    renderTable(algo, user),
    updateStatCards(algo, user),
  ]);
}

function resetFilters() {
  ['f-algo','f-cr','f-user'].forEach(id => { document.getElementById(id).value = ''; });
  applyFilters();
}

// ─── init ─────────────────────────────────────────────────────────────────────
(async () => {
  await loadFilters();
  await applyFilters();
})();

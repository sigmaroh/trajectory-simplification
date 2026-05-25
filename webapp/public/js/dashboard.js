// ─── colour palette (one per algorithm) ────────────────────────────────────
const ALGO_COLORS = {
  dp:             '#f97316',
  vw:             '#22c55e',
  squish:         '#ec4899',
  rw:             '#a855f7',
  greedy_policy:  '#f59e0b',
  proposed:       '#2563eb',
  us:             '#14b8a6',
  at:             '#64748b',
  rl_dqn:         '#ef4444',
};
function algoColor(a) { return ALGO_COLORS[a] || '#94a3b8'; }

// ─── chart registry ──────────────────────────────────────────────────────────
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

// ─── populate filter selects ─────────────────────────────────────────────────
async function loadFilters() {
  const r = await fetch('/api/metrics/filters').then(r => r.json());

  const fAlgo = document.getElementById('f-algo');
  r.algorithms.forEach(a => { const o = new Option(a, a); fAlgo.add(o); });

  const fCR = document.getElementById('f-cr');
  r.compressionRatios.forEach(c => { const o = new Option(`${c}×`, c); fCR.add(o); });

  const fUser = document.getElementById('f-user');
  r.users.slice(0, 100).forEach(u => { const o = new Option(`User ${u}`, u); fUser.add(o); });
}

// ─── fetch comparison data for one metric ───────────────────────────────────
async function fetchComparison(metric, algoFilter) {
  const params = new URLSearchParams({ metric });
  if (algoFilter) params.set('algorithms', algoFilter);
  const d = await fetch(`/api/metrics/comparison?${params}`).then(r => r.json());
  return d;
}

// ─── build line chart datasets ───────────────────────────────────────────────
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

// ─── stat cards ──────────────────────────────────────────────────────────────
async function updateStatCards(algoFilter, userFilter) {
  const params = new URLSearchParams();
  if (algoFilter) params.set('algorithm', algoFilter);
  if (userFilter) params.set('user_id', userFilter);

  const summary = await fetch(`/api/metrics/summary?${params}`).then(r => r.json());

  const algos = [...new Set(summary.map(r => r.algorithm))];
  const trajs = [...new Set(summary.map(r => r.trajectory_id).filter(Boolean))];
  document.getElementById('s-algos').textContent = algos.length || '—';
  document.getElementById('s-runs').textContent  = summary.reduce((s,r) => s + (r.count||0), 0) || '—';

  // best hausdorff (lowest mean)
  const withH = summary.filter(r => r.hausdorff_distance_mean != null);
  if (withH.length) {
    const best = withH.reduce((a,b) => a.hausdorff_distance_mean < b.hausdorff_distance_mean ? a : b);
    document.getElementById('s-best-h').textContent = `${best.hausdorff_distance_mean.toFixed(0)} m`;
    document.getElementById('s-best-h').nextElementSibling.textContent = best.algorithm;
  }

  // best stop preservation
  const withSP = summary.filter(r => r.stop_preservation_mean != null);
  if (withSP.length) {
    const best = withSP.reduce((a,b) => a.stop_preservation_mean > b.stop_preservation_mean ? a : b);
    document.getElementById('s-best-sp').textContent = `${(best.stop_preservation_mean*100).toFixed(1)}%`;
    document.getElementById('s-best-sp').nextElementSibling.textContent = best.algorithm;
  }

  // fastest (lowest mean runtime)
  const withRT = summary.filter(r => r.runtime_seconds_mean != null);
  if (withRT.length) {
    const best = withRT.reduce((a,b) => a.runtime_seconds_mean < b.runtime_seconds_mean ? a : b);
    document.getElementById('s-fastest').textContent = `${best.runtime_seconds_mean.toFixed(3)}s`;
    document.getElementById('s-fastest').nextElementSibling.textContent = best.algorithm;
  }
}

// ─── summary table ───────────────────────────────────────────────────────────
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

// ─── render all charts ───────────────────────────────────────────────────────
async function renderCharts(algoFilter) {
  const [h, sp, tp, rt] = await Promise.all([
    fetchComparison('hausdorff_distance',       algoFilter),
    fetchComparison('stop_preservation',        algoFilter),
    fetchComparison('turn_preservation',        algoFilter),
    fetchComparison('runtime_seconds',          algoFilter),
  ]);

  const labels = h.compressionRatios.map(c => `${c}×`);

  makeChart('chart-hausdorff', 'line', labels, buildDatasets(h));
  makeChart('chart-stop',      'line', labels, buildDatasets(sp));
  makeChart('chart-turn',      'line', labels, buildDatasets(tp));
  makeChart('chart-runtime',   'bar',  labels,
    buildDatasets(rt).map(d => ({ ...d, type: 'bar', backgroundColor: algoColor(d.label) + '99' }))
  );
}

// ─── filter wiring ───────────────────────────────────────────────────────────
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
  ['f-algo','f-cr','f-user'].forEach(id => document.getElementById(id).value = '');
  applyFilters();
}

// ─── init ────────────────────────────────────────────────────────────────────
(async () => {
  await loadFilters();
  await applyFilters();
})();

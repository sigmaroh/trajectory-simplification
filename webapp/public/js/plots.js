// ─── colours — matches generate_plots.py algo_colors ────────────────────────
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
const ALGO_ORDER = ['dp','us','at','vw','squish','rw','greedy_policy','rl_dqn','proposed'];
const ALGO_LABEL = {
  dp:'DP', us:'Uniform', at:'Adaptive Threshold', vw:'VW', squish:'SQUISH',
  rw:'RW', greedy_policy:'Greedy Policy', rl_dqn:'RL DQN', proposed:'Proposed',
};
function color(a) { return ALGO_COLORS[a] || '#94a3b8'; }
function label(a) { return ALGO_LABEL[a] || a; }

// metrics that are "better lower" (error) vs "better higher" (preservation/throughput)
const LOG_METRICS = new Set(['hausdorff_distance','average_pte','frechet_distance','ped','sed','issd']);

// ─── global state ─────────────────────────────────────────────────────────────
let DATA   = null;
let CHARTS = {};

function destroyChart(id) {
  if (CHARTS[id]) { CHARTS[id].destroy(); delete CHARTS[id]; }
}

function makeChart(id, type, labels, datasets, extraOpts = {}) {
  destroyChart(id);
  const ctx = document.getElementById(id);
  if (!ctx) return;
  CHARTS[id] = new Chart(ctx, {
    type,
    data: { labels, datasets },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { position: 'top', labels: { boxWidth: 11, font: { size: 10 } } } },
      scales: {
        x: { ticks: { font: { size: 10 } } },
        y: { ticks: { font: { size: 10 } }, beginAtZero: false },
      },
      ...extraOpts,
    },
  });
}

// ─── tab switching ────────────────────────────────────────────────────────────
function showTab(id) {
  document.querySelectorAll('.tab-section').forEach(s => s.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(b => b.classList.remove('active'));
  document.getElementById(id).classList.add('active');
  [...document.querySelectorAll('.tab')].find(b => b.getAttribute('onclick') === `showTab('${id}')`)
    ?.classList.add('active');
}

// ─── populate filter selects ──────────────────────────────────────────────────
async function loadFilters() {
  const f = await fetch('/api/metrics/filters').then(r => r.json());
  const fa = document.getElementById('f-algo');
  f.algorithms.forEach(a => fa.add(new Option(label(a), a)));
  const fu = document.getElementById('f-user');
  f.users.slice(0, 80).forEach(u => fu.add(new Option(`User ${u}`, u)));
}

function getFilters() {
  return {
    algo: document.getElementById('f-algo').value,
    user: document.getElementById('f-user').value,
  };
}
function resetFilters() {
  ['f-algo','f-user'].forEach(id => { document.getElementById(id).value = ''; });
  reloadAll();
}

// ─── fetch plotdata ───────────────────────────────────────────────────────────
async function fetchPlotData(algo, user) {
  const p = new URLSearchParams();
  if (algo) p.set('algorithm', algo);
  if (user) p.set('user_id', user);
  return fetch(`/api/metrics/plotdata?${p}`).then(r => r.json());
}

// ─── algorithm legend ─────────────────────────────────────────────────────────
function renderLegend(algorithms) {
  const el = document.getElementById('algo-legend');
  el.innerHTML = '';
  const ordered = ALGO_ORDER.filter(a => algorithms.includes(a))
    .concat(algorithms.filter(a => !ALGO_ORDER.includes(a)));
  ordered.forEach(a => {
    const div = document.createElement('div');
    div.className = 'algo-dot';
    div.innerHTML = `<span style="background:${color(a)}"></span>${label(a)}`;
    el.appendChild(div);
  });
}

// ─── helper: build line datasets for one metric ──────────────────────────────
function lineDatasets(metric, algorithms, pivot, crs) {
  const ordered = ALGO_ORDER.filter(a => algorithms.includes(a))
    .concat(algorithms.filter(a => !ALGO_ORDER.includes(a)));
  return ordered.map(algo => {
    const pts = crs.map(cr => {
      const entry = pivot[metric]?.[algo]?.[cr];
      return entry ? entry.mean : null;
    });
    const errs = crs.map(cr => {
      const entry = pivot[metric]?.[algo]?.[cr];
      return entry ? entry.std : 0;
    });
    return {
      label: label(algo),
      data:  pts,
      borderColor: color(algo),
      backgroundColor: color(algo) + '22',
      borderWidth: 2.5,
      pointRadius: 4,
      tension: 0.15,
      errorBars: errs,   // stored for custom use; Chart.js uses error bars via plugin
    };
  }).filter(d => d.data.some(v => v !== null));
}

// ─── section 1: compression–error curves ─────────────────────────────────────
const CURVE_METRICS = [
  'hausdorff_distance','average_pte','frechet_distance',
  'ped','sed','dad','sad','issd','runtime_seconds',
];

function renderCurves(data) {
  const grid = document.getElementById('grid-curves');
  grid.innerHTML = '';
  const crs    = data.compressionRatios;
  const crLabels = crs.map(c => `${c}×`);
  const metrics = data.metrics.filter(m => CURVE_METRICS.includes(m.key));

  metrics.forEach(m => {
    const id = `chart-curve-${m.key}`;
    const card = document.createElement('div');
    card.className = 'plot-card';
    card.innerHTML = `<h4>${m.label}</h4><div class="plot-wrap"><canvas id="${id}"></canvas></div>`;
    grid.appendChild(card);

    // build after DOM paint
    requestAnimationFrame(() => {
      const datasets = lineDatasets(m.key, data.algorithms, data.pivot, crs);
      const useLog   = LOG_METRICS.has(m.key);
      makeChart(id, 'line', crLabels, datasets, {
        scales: {
          x: { title: { display: true, text: 'Compression Ratio', font: { size: 10 } } },
          y: {
            title: { display: true, text: m.label, font: { size: 9 } },
            type: useLog ? 'logarithmic' : 'linear',
            ticks: { font: { size: 9 } },
          },
        },
        plugins: {
          legend: { position: 'top', labels: { boxWidth: 10, font: { size: 9 } } },
          tooltip: {
            callbacks: {
              label: ctx => `${ctx.dataset.label}: ${ctx.parsed.y != null ? ctx.parsed.y.toFixed(3) : '—'}`,
            },
          },
        },
      });
    });
  });
}

// ─── section 2: bar comparison per CR ────────────────────────────────────────
function populateBarMetricSelect(data) {
  const sel = document.getElementById('bar-metric');
  sel.innerHTML = '';
  data.metrics.forEach(m => sel.add(new Option(m.label, m.key)));
}

function renderBarCharts() {
  if (!DATA) return;
  const metricKey = document.getElementById('bar-metric').value;
  const mDef      = DATA.metrics.find(m => m.key === metricKey);
  if (!mDef) return;

  const grid = document.getElementById('grid-bars');
  grid.innerHTML = '';
  const crs      = DATA.compressionRatios;
  const pivot    = DATA.pivot;

  const ordered = ALGO_ORDER.filter(a => DATA.algorithms.includes(a))
    .concat(DATA.algorithms.filter(a => !ALGO_ORDER.includes(a)));

  crs.forEach(cr => {
    const id = `chart-bar-${cr}-${metricKey}`;
    const card = document.createElement('div');
    card.className = 'plot-card';
    card.innerHTML = `<h4>${cr}× Compression — ${mDef.label}</h4><div class="plot-wrap"><canvas id="${id}"></canvas></div>`;
    grid.appendChild(card);

    requestAnimationFrame(() => {
      const algosWithData = ordered.filter(a => pivot[metricKey]?.[a]?.[cr]?.mean != null);
      const means = algosWithData.map(a => pivot[metricKey][a][cr].mean);
      const stds  = algosWithData.map(a => pivot[metricKey][a][cr].std);

      makeChart(id, 'bar', algosWithData.map(label), [{
        label: mDef.label,
        data:  means,
        backgroundColor: algosWithData.map(a => color(a) + 'cc'),
        borderColor:     algosWithData.map(a => color(a)),
        borderWidth: 1.5,
        borderRadius: 4,
      }], {
        scales: {
          x: { ticks: { font: { size: 9 } } },
          y: {
            title: { display: true, text: mDef.label, font: { size: 9 } },
            ticks: { font: { size: 9 } },
          },
        },
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: ctx => {
                const std = stds[ctx.dataIndex];
                return ` ${ctx.parsed.y.toFixed(3)} ± ${std.toFixed(3)}`;
              },
            },
          },
        },
      });
    });
  });
}

// ─── section 3: semantic preservation ────────────────────────────────────────
const SEMANTIC_METRICS = ['turn_preservation','stop_preservation'];

function renderSemantic(data) {
  const grid = document.getElementById('grid-semantic');
  grid.innerHTML = '';
  const crs      = data.compressionRatios;
  const crLabels = crs.map(c => `${c}×`);

  SEMANTIC_METRICS.forEach(key => {
    if (!data.pivot[key]) return;
    const mDef = data.metrics.find(m => m.key === key);
    if (!mDef) return;

    const id = `chart-sem-${key}`;
    const card = document.createElement('div');
    card.className = 'plot-card';
    card.innerHTML = `<h4>${mDef.label}</h4><div class="plot-wrap-tall"><canvas id="${id}"></canvas></div>`;
    grid.appendChild(card);

    requestAnimationFrame(() => {
      const datasets = lineDatasets(key, data.algorithms, data.pivot, crs);
      makeChart(id, 'line', crLabels, datasets, {
        scales: {
          x: { title: { display: true, text: 'Compression Ratio' } },
          y: {
            min: 0, max: 1,
            title: { display: true, text: 'Preservation fraction (0–1)' },
            ticks: {
              callback: v => `${(v*100).toFixed(0)}%`,
              font: { size: 9 },
            },
          },
        },
        plugins: {
          tooltip: {
            callbacks: {
              label: ctx => `${ctx.dataset.label}: ${ctx.parsed.y != null ? (ctx.parsed.y*100).toFixed(1)+'%' : '—'}`,
            },
          },
        },
      });
    });
  });

  // side-by-side bar: stop vs turn at each CR
  const id2 = 'chart-sem-compare';
  const card2 = document.createElement('div');
  card2.className = 'plot-card';
  card2.innerHTML = `<h4>Turn vs Stop Preservation by Algorithm (5× CR)</h4><div class="plot-wrap-tall"><canvas id="${id2}"></canvas></div>`;
  grid.appendChild(card2);

  requestAnimationFrame(() => {
    const cr     = 5;
    const ordered = ALGO_ORDER.filter(a => data.algorithms.includes(a))
      .concat(data.algorithms.filter(a => !ALGO_ORDER.includes(a)));
    const turnVals = ordered.map(a => data.pivot['turn_preservation']?.[a]?.[cr]?.mean ?? null);
    const stopVals = ordered.map(a => data.pivot['stop_preservation']?.[a]?.[cr]?.mean ?? null);
    makeChart(id2, 'bar', ordered.map(label), [
      { label:'Turn Pres.', data: turnVals, backgroundColor:'#3b82f6cc', borderColor:'#3b82f6', borderWidth:1.5, borderRadius:4 },
      { label:'Stop Pres.', data: stopVals, backgroundColor:'#22c55ecc', borderColor:'#22c55e', borderWidth:1.5, borderRadius:4 },
    ], {
      scales: {
        y: { min:0, max:1, ticks: { callback: v => `${(v*100).toFixed(0)}%`, font:{size:9} } },
        x: { ticks: { font:{size:9} } },
      },
    });
  });
}

// ─── section 4: efficiency ────────────────────────────────────────────────────
const EFFICIENCY_METRICS = ['runtime_seconds','memory_mb','throughput_traj_per_sec'];

function renderEfficiency(data) {
  const grid = document.getElementById('grid-efficiency');
  grid.innerHTML = '';
  const crs      = data.compressionRatios;
  const crLabels = crs.map(c => `${c}×`);

  EFFICIENCY_METRICS.forEach(key => {
    if (!data.pivot[key]) return;
    const mDef = data.metrics.find(m => m.key === key);
    if (!mDef) return;

    const id = `chart-eff-${key}`;
    const card = document.createElement('div');
    card.className = 'plot-card';
    card.innerHTML = `<h4>${mDef.label}</h4><div class="plot-wrap"><canvas id="${id}"></canvas></div>`;
    grid.appendChild(card);

    requestAnimationFrame(() => {
      const datasets = lineDatasets(key, data.algorithms, data.pivot, crs);
      makeChart(id, 'line', crLabels, datasets, {
        scales: {
          x: { title: { display: true, text: 'Compression Ratio', font:{size:10} } },
          y: { title: { display: true, text: mDef.label, font:{size:9} }, ticks:{font:{size:9}} },
        },
      });
    });
  });

  // runtime bar at 5× (like generate_plots "runtime by algorithm")
  const id2 = 'chart-eff-runtime-bar';
  const card2 = document.createElement('div');
  card2.className = 'plot-card';
  card2.innerHTML = `<h4>Mean Runtime at 5× Compression (seconds)</h4><div class="plot-wrap"><canvas id="${id2}"></canvas></div>`;
  grid.appendChild(card2);

  requestAnimationFrame(() => {
    const cr = 5;
    const ordered = ALGO_ORDER.filter(a => data.algorithms.includes(a))
      .concat(data.algorithms.filter(a => !ALGO_ORDER.includes(a)));
    const means = ordered.map(a => data.pivot['runtime_seconds']?.[a]?.[cr]?.mean ?? null);
    makeChart(id2, 'bar', ordered.map(label), [{
      data: means,
      backgroundColor: ordered.map(a => color(a) + 'cc'),
      borderColor:     ordered.map(a => color(a)),
      borderWidth: 1.5, borderRadius: 4,
    }], {
      plugins: { legend: { display: false } },
      scales: {
        y: { title: { display: true, text: 'seconds', font:{size:9} }, ticks:{font:{size:9}} },
        x: { ticks:{font:{size:9}} },
      },
    });
  });
}

// ─── main reload ──────────────────────────────────────────────────────────────
async function reloadAll() {
  const { algo, user } = getFilters();
  document.getElementById('plot-status').textContent = 'Loading …';

  DATA = await fetchPlotData(algo, user);

  renderLegend(DATA.algorithms);
  populateBarMetricSelect(DATA);
  renderCurves(DATA);
  renderBarCharts();
  renderSemantic(DATA);
  renderEfficiency(DATA);

  document.getElementById('plot-status').textContent =
    `${DATA.algorithms.length} algorithms · ${DATA.compressionRatios.join('×, ')}×`;
}

// ─── init ─────────────────────────────────────────────────────────────────────
(async () => {
  await loadFilters();
  await reloadAll();
})();

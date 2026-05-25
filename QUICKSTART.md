# Quick Start Guide

## Setup

```bash
# Option A — bundled setup script
./venv_setup.sh

# Option B — manual
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

> All commands below use `./venv/bin/python` — replace with `python` if your venv is activated.

---

## Full Workflow (5 Steps)

### Step 1 — Preprocess GeoLife Data

```bash
# All 182 users (full dataset — ~16,039 trajectories)
./venv/bin/python -m src.utils.preprocess_geolife --all-users

# Specific users only
./venv/bin/python -m src.utils.preprocess_geolife --user-ids 000 001 010

# First N users
./venv/bin/python -m src.utils.preprocess_geolife --max-users 10

# Custom output directory
./venv/bin/python -m src.utils.preprocess_geolife --all-users --output-dir data/processed
```

Outputs: `data/processed/trajectories.pkl`, `trajectory_properties.csv`, `trajectories_points.csv`

---

### Step 2 — Run Experiments

```bash
# All trajectories, key algorithms, 3 compression ratios (recommended)
./venv/bin/python -m src.experiments.run_experiments \
  --algorithms dp vw squish rw greedy_policy proposed \
  --compression-ratios 2 5 10

# Specific users only
./venv/bin/python -m src.experiments.run_experiments \
  --user-ids 000 001 \
  --algorithms vw rw proposed \
  --compression-ratios 2 5 10

# Fast run (skip slow DP)
./venv/bin/python -m src.experiments.run_experiments \
  --algorithms vw squish rw greedy_policy proposed \
  --compression-ratios 2 5 10

# Include all algorithms (us, at, rl_dqn, dp, etc.)
./venv/bin/python -m src.experiments.run_experiments \
  --algorithms dp us at vw squish rw greedy_policy rl_dqn proposed \
  --compression-ratios 2 5 10

# Limit trajectory count
./venv/bin/python -m src.experiments.run_experiments \
  --max-trajectories 20 \
  --algorithms vw proposed \
  --compression-ratios 2 5 10
```

Outputs: `results/experiment_results.csv`, `results/summary_table.csv`

---

### Step 3 — Generate Dataset Plots

```bash
./venv/bin/python -m src.experiments.generate_dataset_plots \
  --data-file data/processed/trajectories.pkl \
  --max-trajectories 300 \
  --output-dir results/figures
```

Outputs: `dataset_length_distribution.png`, `dataset_sampling_irregularity.png`, `dataset_speed.png`, `dataset_turns_stops.png`

---

### Step 4 — Generate Algorithm Comparison Plots

```bash
./venv/bin/python -m src.experiments.generate_plots \
  --results-file results/experiment_results.csv \
  --trajectories-file data/processed/trajectories.pkl \
  --output-dir results/figures
```

Outputs in `results/figures/`:

| File | Description |
|---|---|
| `trajectory_comparison.png` | Side-by-side map at 5× CR |
| `compression_error_curves.png` | Error vs CR for all algorithms |
| `compression_error_curves_2x/5x/10x.png` | Per-CR breakdown |
| `metric_comparison_2x/5x/10x.png` | Bar charts at each CR |
| `runtime_scalability.png` | Runtime vs trajectory size |
| `per_metric/metric_*.png` | One page per metric |
| `plot_data/*.csv` | Raw aggregated plot data for re-plotting |

---

### Step 5 — Generate Interactive OSM Map

```bash
# Full Folium map (HD + Fréchet + APTE in tooltips and table)
./venv/bin/python -m src.experiments.visualize_osm \
  --comparison \
  --algorithms "original,vw,rw,squish,greedy_policy,proposed" \
  --compression-ratios "2,5,10" \
  --max-trajectories 20 \
  --output-file results/figures/trajectories_osm_comparison.html

# Lightweight JSON + HTML viewer
./venv/bin/python -m src.experiments.export_osm_json_map \
  --algorithms "original,vw,rw,squish,greedy_policy,proposed" \
  --compression-ratios "5,10" \
  --max-trajectories 5 \
  --output-json results/figures/trajectories_osm_comparison_data.json \
  --output-html results/figures/trajectories_osm_comparison_from_json.html
```

---

## All Available CLI Flags

### `preprocess_geolife.py`

| Flag | Default | Description |
|---|---|---|
| `--data-dir` | `data/geolife` | Path to GeoLife root |
| `--all-users` | — | Load all 182 users |
| `--max-users N` | 50 | Load first N users (ignored if `--user-ids` set) |
| `--user-ids 000 010 …` | — | Load specific users only |
| `--min-points N` | 100 | Min points per raw trajectory |
| `--min-points-after-clean N` | 50 | Min points after outlier removal |
| `--output-dir` | `data/processed` | Output directory |
| `--export-csv-only PKL` | — | Convert existing `.pkl` to CSV only |

### `run_experiments.py`

| Flag | Default | Description |
|---|---|---|
| `--data-file` | `data/processed/trajectories.pkl` | Input trajectories |
| `--user-ids 000 010 …` | — | Filter to specific users |
| `--max-trajectories N` | *(no limit)* | Cap number of trajectories |
| `--algorithms …` | all | Space-separated list: `dp us at vw squish rw greedy_policy rl_dqn proposed` |
| `--compression-ratios …` | `2 5 10 20` | Target compression ratios |

### `generate_plots.py`

| Flag | Default | Description |
|---|---|---|
| `--results-file` | `results/experiment_results.csv` | Experiment results CSV |
| `--trajectories-file` | `data/processed/trajectories.pkl` | Trajectories pickle |
| `--output-dir` | `results/figures` | Plot output directory |

### `generate_dataset_plots.py`

| Flag | Default | Description |
|---|---|---|
| `--data-file` | `data/processed/trajectories.pkl` | Input trajectories |
| `--max-trajectories N` | 300 | Trajectories to analyse |
| `--output-dir` | `results/figures` | Output directory |

### `visualize_osm.py`

| Flag | Default | Description |
|---|---|---|
| `--comparison` | — | Enable algorithm comparison mode |
| `--algorithms` | `"original,dp,vw,squish,rw,greedy_policy,proposed"` | Comma-separated list |
| `--compression-ratios` | `"5"` | Comma-separated ratios |
| `--max-trajectories N` | 30 | Max trajectories to render |
| `--output-file` | `results/figures/trajectories_osm.html` | Output HTML path |

---

## Quick Code Snippets

### Load trajectories

```python
import pickle
with open('data/processed/trajectories.pkl', 'rb') as f:
    trajs = pickle.load(f)
print(f"{len(trajs)} trajectories, columns: {list(trajs[0].columns)}")
```

### Run one algorithm

```python
from src.algorithms.baseline_algorithms import simplify_with_budget
traj = trajs[0]
budget = len(traj) // 5   # 5× compression
simp = simplify_with_budget(traj, 'vw', budget)
print(f"VW: {len(traj)} → {len(simp)} pts")
```

### Run proposed method with metrics

```python
from src.algorithms.proposed_method import proposed_simplification
from src.metrics.evaluation_metrics import compute_all_metrics

simp, idx = proposed_simplification(traj, budget)
m = compute_all_metrics(traj, simp, idx)
print(f"Hausdorff: {m['hausdorff_distance']:.1f} m")
print(f"Fréchet:   {m['frechet_distance']:.1f} m")
print(f"Turn pres: {m.get('turn_preservation', 'N/A')}")
print(f"Stop pres: {m.get('stop_preservation', 'N/A')}")
```

### Compare all algorithms on one trajectory

```python
import sys; sys.path.insert(0, '.')
from src.algorithms.baseline_algorithms import simplify_with_budget
from src.algorithms.proposed_method import proposed_simplification

traj = trajs[0]
budget = len(traj) // 5

for algo in ['dp', 'us', 'at', 'vw', 'squish', 'rw', 'greedy_policy']:
    simp = simplify_with_budget(traj, algo, budget)
    cr = len(traj) / len(simp)
    print(f"  {algo:15s}: {len(traj)} → {len(simp)} pts  (CR={cr:.2f}×)")

simp, idx = proposed_simplification(traj, budget)
print(f"  {'proposed':15s}: {len(traj)} → {len(simp)} pts  (CR={len(traj)/len(simp):.2f}×)")
```

### Plot trajectory comparison

```python
from src.experiments.generate_plots import plot_trajectory_comparison
import matplotlib; matplotlib.use('Agg')

plot_trajectory_comparison(
    trajs[0],
    ['dp', 'vw', 'rw', 'greedy_policy', 'proposed'],
    compression_ratio=5.0,
    output_path='comparison.png'
)
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'src'` | Run from project root: `cd /path/to/CSIT-8-PROJECT` |
| `externally-managed-environment` pip error | Use `./venv/bin/python` instead of system `python` |
| DP is slow | Use `--algorithms vw squish rw greedy_policy proposed` to skip DP |
| Out of memory | Add `--max-trajectories 10` |
| OSM map takes too long | Add `--max-trajectories 5` |
| Plots show wrong CRs | Regenerate results with fixed DP: `run_experiments.py` (DP now returns exact budget) |

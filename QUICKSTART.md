# Quick Start Guide

## Installation

```bash
# Option A — use the bundled setup script
./venv_setup.sh
source venv/bin/activate

# Option B — manual
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Quick Test (GeoLife Data Already Preprocessed)

If `data/processed/trajectories.pkl` already exists, you can jump straight to experiments:

```python
import pickle, sys
sys.path.insert(0, '.')
from src.algorithms.baseline_algorithms import simplify_with_budget
from src.algorithms.proposed_method import proposed_simplification
from src.metrics.evaluation_metrics import compute_all_metrics

with open('data/processed/trajectories.pkl', 'rb') as f:
    trajectories = pickle.load(f)

traj = trajectories[0]
budget = len(traj) // 5  # 5× compression

# Test Greedy Policy (RL-inspired)
gp = simplify_with_budget(traj, 'greedy_policy', budget, alpha=0.5)
print(f"Greedy Policy: {len(traj)} → {len(gp)} points")

# Test Proposed method
simp, idx = proposed_simplification(traj, budget)
metrics = compute_all_metrics(traj, simp, idx)
print(f"Proposed:      {len(traj)} → {len(simp)} points")
print(f"Turn pres.: {metrics.get('turn_preservation', 'N/A'):.3f}  "
      f"Stop pres.: {metrics.get('stop_preservation', 'N/A'):.3f}")
```

---

## Full Workflow

### Step 1 — Preprocess GeoLife Data

```bash
python src/utils/preprocess_geolife.py
```

Outputs:
- `data/processed/trajectories.pkl` — used by experiments
- `data/processed/trajectory_properties.csv` — per-trajectory summary
- `data/processed/trajectories_points.csv` — all GPS points (long CSV)
- `data/processed/trajectories_index.csv` — trajectory index

### Step 2 — Run Experiments

```bash
python src/experiments/run_experiments.py \
    --max-trajectories 10 \
    --compression-ratios 2.0 5.0 10.0 20.0 \
    --algorithms dp vw squish rw greedy_policy proposed \
    --data-file data/processed/trajectories.pkl
```

Outputs: `results/experiment_results.csv`, `results/summary_table.csv`

> **Semantic metrics**: Turn/stop columns are filled for **proposed** rows only in the batch runner (baselines do not return selected indices).

> **Note on speed**: `dp` and `sw` are slow on long trajectories (O(n²) binary search).
> Omit them or limit `--max-trajectories 5` for a faster run.

### Step 3 — Generate Dataset Analysis Plots

```bash
python src/experiments/generate_dataset_plots.py \
    --data-file data/processed/trajectories.pkl \
    --max-trajectories 300
```

Outputs in `results/figures/`:
- `dataset_length_distribution.png`
- `dataset_sampling_irregularity.png`
- `dataset_speed.png`
- `dataset_turns_stops.png`

### Step 4 — Generate Algorithm Comparison Plots

```bash
python src/experiments/generate_plots.py \
    --results-file results/experiment_results.csv \
    --trajectories-file data/processed/trajectories.pkl \
    --output-dir results/figures
```

Outputs in `results/figures/`:
- `trajectory_comparison.png` — DP, VW, RW, Greedy Policy, Proposed side-by-side
- `compression_error_curves.png` — error vs compression ratio
- `runtime_scalability.png` — runtime vs trajectory size
- `metric_comparison_5x.png`, `metric_comparison_10x.png`

### Step 5 — Generate Interactive OSM Map

```bash
# Lightweight JSON + HTML viewer (recommended)
python src/experiments/export_osm_json_map.py \
    --algorithms "original,dp,vw,squish,rw,greedy_policy,proposed" \
    --compression-ratios "5,10" \
    --max-trajectories 1

# Full Folium comparison map
python src/experiments/visualize_osm.py \
    --comparison \
    --algorithms "original,dp,vw,squish,rw,greedy_policy,proposed" \
    --compression-ratios "5,10" \
    --max-trajectories 1 \
    --output-file results/figures/trajectories_osm_comparison.html
```

### Step 6 — Dataset Analysis Notebook

```bash
jupyter notebook notebooks/01_dataset_analysis.ipynb
```

---

## Project Structure at a Glance

```
src/algorithms/baseline_algorithms.py  ← DP, SW, VW, SQUISH, RW, Greedy Policy
src/utils/config.py                    ← Central constants & experiment defaults
src/algorithms/proposed_method.py      ← Proposed turn/stop/speed-aware method
src/metrics/evaluation_metrics.py      ← All metrics (geometric + semantic)
src/experiments/run_experiments.py     ← Main batch experiment runner
src/experiments/generate_plots.py      ← Algorithm comparison plots
src/experiments/generate_dataset_plots.py  ← Dataset characterisation plots
src/experiments/visualize_osm.py       ← Folium OSM interactive map
src/experiments/export_osm_json_map.py ← Lightweight JSON/HTML map
config/experiment_config.yaml          ← Reference copy of experiment defaults
```

---

## Common Snippets

### Test a Single Algorithm

```python
import pickle, sys
sys.path.insert(0, '.')
from src.algorithms.baseline_algorithms import simplify_with_budget

with open('data/processed/trajectories.pkl', 'rb') as f:
    trajs = pickle.load(f)
traj = trajs[0]
budget = len(traj) // 5

for algo in ['dp', 'vw', 'rw', 'greedy_policy']:
    simp = simplify_with_budget(traj, algo, budget)
    print(f"{algo:15s}: {len(traj)} → {len(simp)} pts")
```

### Compare Algorithms Visually

```python
from src.experiments.generate_plots import plot_trajectory_comparison
import pickle, matplotlib
matplotlib.use('Agg')

with open('data/processed/trajectories.pkl', 'rb') as f:
    trajs = pickle.load(f)

plot_trajectory_comparison(
    trajs[0],
    ['dp', 'vw', 'rw', 'greedy_policy', 'proposed'],
    compression_ratio=5.0,
    output_path='my_comparison.png'
)
```


---

## Troubleshooting

### Import Errors

Run from the project root:
```bash
cd /home/sanjay/Desktop/DESKALL/CSIT-8-PROJECT
python src/experiments/run_experiments.py
```

### DP / SW Are Very Slow

For long GeoLife trajectories, DP uses binary search (O(n²) worst case). Use the fast subset:
```bash
python src/experiments/run_experiments.py \
    --algorithms vw squish rw greedy_policy proposed \
    --max-trajectories 10
```

### Memory Issues

```bash
python src/experiments/run_experiments.py --max-trajectories 5
```

### Externally Managed Python (no pip install)

Use the bundled venv:
```bash
./venv_setup.sh
venv/bin/python src/experiments/run_experiments.py ...
```

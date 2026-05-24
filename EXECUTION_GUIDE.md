# Complete Execution Guide

## Prerequisites

- Python 3.8+
- Virtual environment activated: `source venv/bin/activate`
- Dependencies installed: `pip install -r requirements.txt`
- GeoLife dataset in `data/geolife/` (or use preprocessed `data/processed/trajectories.pkl`)

---

## Complete Workflow

### Step 1 — Preprocess GeoLife Dataset

```bash
cd /home/sanjay/Desktop/DESKALL/CSIT-8-PROJECT
python src/utils/preprocess_geolife.py
```

**Expected outputs**:
- `data/processed/trajectories.pkl` — list of cleaned trajectory DataFrames (e.g. 5,716 with default `--max-users 50`)
- `data/processed/trajectory_properties.csv` — per-trajectory statistics
- `data/processed/trajectories_points.csv` — all GPS points (long CSV)
- `data/processed/trajectories_index.csv` — trajectory index

**Time**: ~5–15 minutes (processes up to 50 users by default)

---

### Step 2 — Run Main Experiments

```bash
# Recommended: fast algorithms only (vw, squish, rw, greedy_policy, proposed)
python src/experiments/run_experiments.py \
    --max-trajectories 10 \
    --compression-ratios 2.0 5.0 10.0 20.0 \
    --algorithms vw squish rw greedy_policy proposed \
    --data-file data/processed/trajectories.pkl

# Full comparison including DP (slow on long trajectories)
python src/experiments/run_experiments.py \
    --max-trajectories 10 \
    --compression-ratios 2.0 5.0 10.0 20.0 \
    --algorithms dp vw squish rw greedy_policy proposed \
    --data-file data/processed/trajectories.pkl
```

**Expected outputs**:
- `results/experiment_results.csv` — one row per (trajectory, algorithm, compression ratio)
- `results/summary_table.csv` — mean ± std grouped by algorithm × compression ratio

**Row count**: 10 trajectories × 6 algorithms (`dp vw squish rw greedy_policy proposed`) × 4 CRs = **240 rows** (omit `sw` and `original` in this command)
> `dp` adds ~4 s/trajectory; `sw` adds ~23 s/trajectory — use with small `--max-trajectories`

---

### Step 3 — Generate Dataset Characterisation Plots

```bash
python src/experiments/generate_dataset_plots.py \
    --data-file data/processed/trajectories.pkl \
    --max-trajectories 300 \
    --output-dir results/figures
```

**Expected outputs** in `results/figures/`:
- `dataset_length_distribution.png`
- `dataset_sampling_irregularity.png`
- `dataset_speed.png`
- `dataset_turns_stops.png`

**Time**: ~30 s

---

### Step 4 — Generate Algorithm Comparison Plots

```bash
python src/experiments/generate_plots.py \
    --results-file results/experiment_results.csv \
    --trajectories-file data/processed/trajectories.pkl \
    --output-dir results/figures
```

**Expected outputs** in `results/figures/`:
- `trajectory_comparison.png` — DP / VW / RW / Greedy Policy / Proposed at 5× CR
- `compression_error_curves.png` — error metrics vs compression ratio
- `runtime_scalability.png` — runtime vs trajectory size
- `metric_comparison_5x.png` — bar chart at 5× compression
- `metric_comparison_10x.png` — bar chart at 10× compression

**Time**: ~2–5 min

---

### Step 5 — Generate Interactive OSM Maps

```bash
# Lightweight JSON + HTML viewer (fast, opens in any browser)
python src/experiments/export_osm_json_map.py \
    --algorithms "original,dp,vw,squish,rw,greedy_policy,proposed" \
    --compression-ratios "5,10" \
    --max-trajectories 1 \
    --output-json results/figures/trajectories_osm_comparison_data.json \
    --output-html results/figures/trajectories_osm_comparison_from_json.html

# Full Folium comparison map
python src/experiments/visualize_osm.py \
    --comparison \
    --algorithms "original,dp,vw,squish,rw,greedy_policy,proposed" \
    --compression-ratios "5,10" \
    --max-trajectories 1 \
    --output-file results/figures/trajectories_osm_comparison.html
```

**Expected outputs** in `results/figures/`:
- `trajectories_osm_comparison_from_json.html` — lightweight viewer
- `trajectories_osm_comparison.html` — full Folium map (metrics table includes **Fréchet**)

**Time**: ~5 min per map

---

### Step 6 — Open Dataset Analysis Notebook

```bash
jupyter notebook notebooks/01_dataset_analysis.ipynb
```

Covers: trajectory length distribution, sampling irregularity, noise, speed, turns, stops.

---

## Testing Individual Components

### Test Loading

```python
import pickle
with open('data/processed/trajectories.pkl', 'rb') as f:
    trajs = pickle.load(f)
print(f"Loaded {len(trajs)} trajectories")
print(f"First trajectory: {len(trajs[0])} points, columns: {trajs[0].columns.tolist()}")
```

### Test All Simplification Algorithms

```python
import pickle, sys
sys.path.insert(0, '.')
from src.algorithms.baseline_algorithms import simplify_with_budget
from src.algorithms.proposed_method import proposed_simplification

with open('data/processed/trajectories.pkl', 'rb') as f:
    trajs = pickle.load(f)
traj = sorted(trajs[:50], key=len)[5]   # short trajectory
budget = len(traj) // 5

for algo in ['dp', 'vw', 'squish', 'rw', 'greedy_policy']:
    simp = simplify_with_budget(traj, algo, budget)
    print(f"  {algo:15s}: {len(traj)} → {len(simp)} pts")

simp, idx = proposed_simplification(traj, budget)
print(f"  {'proposed':15s}: {len(traj)} → {len(simp)} pts")
```

### Test Metrics

```python
from src.metrics.evaluation_metrics import compute_all_metrics
metrics = compute_all_metrics(traj, simp, idx)
for k, v in metrics.items():
    if v is not None:
        print(f"  {k}: {v:.4f}")
```

---

## Results File Reference

### `results/experiment_results.csv` — Column Description

| Column | Description |
|---|---|
| `algorithm` | Algorithm ID (`dp`, `vw`, `squish`, `rw`, `greedy_policy`, `proposed`) |
| `compression_ratio` | Actual compression ratio (may differ slightly from target due to integer floor) |
| `budget` | Number of output points |
| `trajectory_id` | Index into the loaded trajectory list |
| `trajectory_size` | Number of points in the original trajectory |
| `runtime_seconds` | Wall-clock time for one simplification |
| `memory_mb` | Peak memory overhead (tracemalloc) |
| `throughput_traj_per_sec` | `1 / runtime_seconds` |
| `hausdorff_distance` | Max one-way deviation (metres) |
| `average_pte` | Mean point-to-polyline error (metres) |
| `frechet_distance` | Discrete Fréchet distance (metres) |
| `ped` | Perpendicular Euclidean Distance (metres) |
| `sed` | Synchronised Euclidean Distance (metres) |
| `dad` | Direction Angle Difference (degrees) |
| `sad` | Speed Accuracy Difference (m/s) |
| `issd` | Integrated Square Speed Difference (m·s) |
| `turn_preservation` | Fraction of turns preserved (0–1; `proposed` only) |
| `stop_preservation` | Fraction of stops preserved (0–1; `proposed` only) |
| `original_points` | Point count of original trajectory |
| `simplified_points` | Point count after simplification |

---

## Customising Experiments

### Change Weights for Proposed Method

```python
from src.algorithms.proposed_method import proposed_simplification

simp, idx = proposed_simplification(
    traj,
    budget=len(traj) // 5,
    weights={'turn': 0.4, 'stop': 0.4, 'speed': 0.1, 'irregular': 0.1}
)
```

### Change Alpha for Greedy Policy

```python
from src.algorithms.baseline_algorithms import simplify_with_budget

simp = simplify_with_budget(traj, 'greedy_policy', budget, alpha=0.7)
```

### Add a New Algorithm

1. Implement function in `src/algorithms/baseline_algorithms.py`
2. Add alias in the `algorithm_aliases` dict in `simplify_with_budget()`
3. Add a color in `ALGORITHM_COLORS` in `visualize_osm.py`
4. Add display name in all `display_names` dicts

---

## Troubleshooting

### Import Error: `ModuleNotFoundError: No module named 'src'`

Run from project root:
```bash
cd /home/sanjay/Desktop/DESKALL/CSIT-8-PROJECT
python src/experiments/run_experiments.py
```

### `externally-managed-environment` pip error

Use the bundled venv:
```bash
./venv_setup.sh
venv/bin/python src/experiments/run_experiments.py ...
```

### DP / SW Are Very Slow

DP and SW use binary search on ε — O(n²) worst case. For long GeoLife trajectories (median 930 points) they can take 4–23 s each. Run with fast algorithms only:
```bash
--algorithms vw squish rw greedy_policy proposed
```

### Out of Memory

```bash
python src/experiments/run_experiments.py --max-trajectories 5
```

### Maps Take Too Long

Reduce to 1 trajectory:
```bash
--max-trajectories 1
```

---

## Performance Reference

| Algorithm | Runtime (mean, s) | Throughput (traj/s) |
|---|---|---|
| Greedy Policy | 0.049 | 20.4 |
| RW | 0.069 | 14.5 |
| VW | 0.165 | 6.1 |
| Proposed | 0.180 | 5.6 |
| SQUISH | 0.229 | 4.4 |
| DP | ~4.1 | ~0.24 |
| SW | ~23.2 | ~0.04 |

*Measured on 10 GeoLife trajectories (95–209 points) at 4 compression ratios.*

---

**Ready to execute!** Follow the steps above to reproduce all project results.

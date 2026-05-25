# Complete Execution Guide

## Prerequisites

- Python 3.8+
- GeoLife dataset in `data/geolife/` (or use existing `data/processed/trajectories.pkl`)

```bash
# Activate venv (or use ./venv/bin/python directly in all commands below)
source venv/bin/activate
# If venv doesn't exist yet:
./venv_setup.sh
```

---

## Step 1 — Preprocess GeoLife Dataset

```bash
# Full dataset — all 182 users
./venv/bin/python -m src.utils.preprocess_geolife --all-users

# Specific users (overrides --all-users and --max-users)
./venv/bin/python -m src.utils.preprocess_geolife --user-ids 000 010 050

# First N users
./venv/bin/python -m src.utils.preprocess_geolife --max-users 10

# Save to a custom directory
./venv/bin/python -m src.utils.preprocess_geolife \
  --all-users \
  --output-dir data/processed

# Convert an existing .pkl to CSV without re-running preprocessing
./venv/bin/python -m src.utils.preprocess_geolife \
  --export-csv-only data/processed/trajectories.pkl \
  --output-dir data/processed
```

**All flags:**

| Flag | Default | Description |
|---|---|---|
| `--data-dir` | `data/geolife` | GeoLife root directory |
| `--all-users` | — | Load every user in the dataset |
| `--max-users N` | 50 | Load first N users (ignored if `--user-ids` set) |
| `--user-ids ID …` | — | Load specific users, e.g. `000 010 050` |
| `--min-points N` | 100 | Min points per raw `.plt` trajectory |
| `--min-points-after-clean N` | 50 | Min points required after outlier removal |
| `--output-dir` | `data/processed` | Output directory |
| `--export-csv-only PKL` | — | Skip loading; convert `.pkl` → CSV only |

**Outputs:**
- `data/processed/trajectories.pkl` — list of cleaned trajectory DataFrames
- `data/processed/trajectory_properties.csv` — per-trajectory statistics
- `data/processed/trajectories_points.csv` — all GPS points (long CSV)
- `data/processed/trajectories_index.csv` — trajectory index

**Time:** ~15–30 min for all 182 users

---

## Step 2 — Run Experiments

```bash
# Recommended: fast algorithms at 3 compression ratios, all trajectories
./venv/bin/python -m src.experiments.run_experiments \
  --algorithms vw squish rw greedy_policy proposed \
  --compression-ratios 2 5 10

# Full comparison including DP
./venv/bin/python -m src.experiments.run_experiments \
  --algorithms dp vw squish rw greedy_policy proposed \
  --compression-ratios 2 5 10

# All algorithms
./venv/bin/python -m src.experiments.run_experiments \
  --algorithms dp us at vw squish rw greedy_policy rl_dqn proposed \
  --compression-ratios 2 5 10

# Filter to specific users
./venv/bin/python -m src.experiments.run_experiments \
  --user-ids 000 001 010 \
  --algorithms vw rw proposed \
  --compression-ratios 2 5 10

# Limit to N trajectories (useful for testing)
./venv/bin/python -m src.experiments.run_experiments \
  --max-trajectories 20 \
  --algorithms vw proposed \
  --compression-ratios 2 5 10

# Custom data file
./venv/bin/python -m src.experiments.run_experiments \
  --data-file data/processed/trajectories.pkl \
  --algorithms vw proposed \
  --compression-ratios 2 5 10
```

**All flags:**

| Flag | Default | Description |
|---|---|---|
| `--data-file` | `data/processed/trajectories.pkl` | Input trajectories pickle |
| `--user-ids ID …` | — | Filter to specific GeoLife user IDs |
| `--max-trajectories N` | *(no limit)* | Cap number of trajectories processed |
| `--algorithms …` | all | Space-separated: `dp us at vw squish rw greedy_policy rl_dqn proposed` |
| `--compression-ratios …` | `2 5 10 20` | Target compression ratios |

**Available algorithm IDs:**

| ID | Algorithm | Speed |
|---|---|---|
| `dp` | Douglas-Peucker (RDP) | Slow (~4 s/traj) |
| `us` | Uniform Sampling | Fast |
| `at` | Adaptive Threshold | Slow (~50 s/traj) |
| `vw` | Visvalingam-Whyatt | Medium |
| `squish` | SQUISH | Medium |
| `rw` | Reumann-Witkam | Fast |
| `greedy_policy` | Greedy Policy (RL-inspired) | Fast |
| `rl_dqn` | RL DQN (Wang et al. 2021) | Fast (needs pre-trained weights) |
| `proposed` | Proposed semantic method | Medium |

> All algorithms now return **exactly** the requested budget (DP is padded post-hoc to guarantee identical compression ratios for fair comparison).

**Outputs:**
- `results/experiment_results.csv` — one row per (trajectory × algorithm × compression ratio)
- `results/summary_table.csv` — mean ± std grouped by algorithm × CR

**Row count example:** 100 trajectories × 5 algorithms × 3 CRs = 1,500 rows

---

## Step 3 — Generate Dataset Characterisation Plots

```bash
./venv/bin/python -m src.experiments.generate_dataset_plots \
  --data-file data/processed/trajectories.pkl \
  --max-trajectories 300 \
  --output-dir results/figures
```

**All flags:**

| Flag | Default | Description |
|---|---|---|
| `--data-file` | `data/processed/trajectories.pkl` | Input trajectories |
| `--max-trajectories N` | 300 | How many trajectories to analyse |
| `--output-dir` | `results/figures` | Where to save plots |

**Outputs in `results/figures/`:**
- `dataset_length_distribution.png`
- `dataset_sampling_irregularity.png`
- `dataset_speed.png`
- `dataset_turns_stops.png`
- `plot_data/dataset_*.csv` — raw data for each plot

**Time:** ~30 s

---

## Step 4 — Generate Algorithm Comparison Plots

```bash
./venv/bin/python -m src.experiments.generate_plots \
  --results-file results/experiment_results.csv \
  --trajectories-file data/processed/trajectories.pkl \
  --output-dir results/figures
```

**All flags:**

| Flag | Default | Description |
|---|---|---|
| `--results-file` | `results/experiment_results.csv` | Experiment results CSV |
| `--trajectories-file` | `data/processed/trajectories.pkl` | Trajectories pickle |
| `--output-dir` | `results/figures` | Output directory |

**Outputs in `results/figures/`:**

| File | Description |
|---|---|
| `trajectory_comparison.png` | Side-by-side map view at 5× CR |
| `compression_error_curves.png` | Error metrics vs compression ratio |
| `compression_error_curves_2x.png` | Error curves at 2× CR only |
| `compression_error_curves_5x.png` | Error curves at 5× CR only |
| `compression_error_curves_10x.png` | Error curves at 10× CR only |
| `metric_comparison_2x.png` | Bar chart of all metrics at 2× CR |
| `metric_comparison_5x.png` | Bar chart of all metrics at 5× CR |
| `metric_comparison_10x.png` | Bar chart of all metrics at 10× CR |
| `runtime_scalability.png` | Runtime vs trajectory size (log-log) |
| `per_metric/metric_*.png` | One page per metric, all CRs |
| `plot_data/compression_error_aggregated.csv` | Mean ± std data for error curves |
| `plot_data/metric_comparison_*.csv` | Raw data for bar charts |

**Time:** ~1–3 min

> Plots group results by **target** compression ratio (2×/5×/10×), derived from `input_points / budget`. This ensures DP rows (which may output fewer points than the budget) are still correctly placed in the right group.

---

## Step 5 — Generate Interactive OSM Maps

```bash
# Full Folium map — Fréchet distance shown in hover tooltip and bottom table
./venv/bin/python -m src.experiments.visualize_osm \
  --comparison \
  --algorithms "original,vw,rw,squish,greedy_policy,proposed" \
  --compression-ratios "2,5,10" \
  --max-trajectories 20 \
  --output-file results/figures/trajectories_osm_comparison.html

# Simple overview map (no algorithm comparison)
./venv/bin/python -m src.experiments.visualize_osm \
  --output-file results/figures/trajectories_osm.html \
  --max-trajectories 30

# Lightweight JSON + HTML viewer (fast, no Folium dependency)
./venv/bin/python -m src.experiments.export_osm_json_map \
  --algorithms "original,vw,rw,squish,greedy_policy,proposed" \
  --compression-ratios "5,10" \
  --max-trajectories 5 \
  --output-json results/figures/trajectories_osm_comparison_data.json \
  --output-html results/figures/trajectories_osm_comparison_from_json.html
```

**Flags for `visualize_osm.py`:**

| Flag | Default | Description |
|---|---|---|
| `--comparison` | — | Enable multi-algorithm comparison mode |
| `--algorithms` | `"original,dp,vw,squish,rw,greedy_policy,proposed"` | Comma-separated algorithm list |
| `--compression-ratios` | `"5"` | Comma-separated ratios |
| `--max-trajectories N` | 30 | Max trajectories to render |
| `--max-points-per-trajectory N` | 1200 | Cap points per trajectory for rendering |
| `--trajectories-file` | `data/processed/trajectories.pkl` | Input trajectories |
| `--output-file` | `results/figures/trajectories_osm.html` | Output HTML |

**Map features:**
- Dropdown selectors for compression ratio and trajectory
- Hover tooltip on each polyline: algorithm name, HD, **Fréchet**, APTE
- Bottom-left table: Hausdorff (m), Fréchet (m), APTE (m) per algorithm
- Algorithm colour legend
- Basemap switcher (OSM / Carto / Esri)

**Time:** ~2–5 min depending on trajectory count

---

## Step 6 — RL DQN Policy (Optional)

Pre-train the RL DQN weights before using `rl_dqn` in experiments:

```bash
# Train on all trajectories (50 epochs)
./venv/bin/python -m src.algorithms.rl_policy \
  --epochs 50 \
  --output models/rl_policy.npz

# Quick smoke test (5 epochs)
./venv/bin/python -m src.algorithms.rl_policy --epochs 5
```

Pre-trained weights are saved at `models/rl_policy.npz` and loaded automatically when `rl_dqn` is requested.

---

## Step 7 — Dataset Analysis Notebook

```bash
jupyter notebook notebooks/01_dataset_analysis.ipynb
```

Covers: trajectory length, sampling irregularity, noise levels, speed profiles, turns, stops.

---

## Testing Individual Components

```bash
# Test proposed method end-to-end
./venv/bin/python -m src.algorithms.proposed_method

# Test all baseline algorithms
./venv/bin/python -m src.algorithms.baseline_algorithms

# Test evaluation metrics
./venv/bin/python -m src.metrics.evaluation_metrics

# Check how many trajectories are loaded
./venv/bin/python -c "
import pickle
with open('data/processed/trajectories.pkl','rb') as f:
    t = pickle.load(f)
print(f'{len(t)} trajectories')
users = sorted(set(x[\"user_id\"].iloc[0] for x in t if \"user_id\" in x.columns))
print(f'{len(users)} users: {users[:5]} ...')
"
```

---

## Results File Reference

### `results/experiment_results.csv`

| Column | Description |
|---|---|
| `algorithm` | Algorithm ID (`dp`, `vw`, `squish`, `rw`, `greedy_policy`, `proposed`, …) |
| `compression_ratio` | Actual achieved ratio = `n / output_pts` |
| `budget` | Requested number of output points |
| `input_points` | Points in the original trajectory |
| `output_points` | Points in the simplified trajectory |
| `runtime_seconds` | Wall-clock time for one simplification |
| `memory_mb` | Peak memory (tracemalloc) |
| `throughput_traj_per_sec` | `1 / runtime_seconds` |
| `hausdorff_distance` | Max one-way deviation (m) |
| `frechet_distance` | Discrete Fréchet distance (m) |
| `average_pte` | Mean point-to-polyline error (m) |
| `ped` | Perpendicular Euclidean Distance (m) |
| `sed` | Synchronised Euclidean Distance (m) |
| `dad` | Direction Angle Difference (degrees) |
| `sad` | Speed Accuracy Difference (m/s) |
| `issd` | Integrated Square Speed Difference (m·s) |
| `turn_preservation` | Fraction of turns preserved (0–1; `proposed` only) |
| `stop_preservation` | Fraction of stops preserved (0–1; `proposed` only) |

> **Grouping for plots:** use `input_points / budget` as the target CR, not `compression_ratio`. `generate_plots.py` does this automatically.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'src'` | Run from project root: `cd /path/to/CSIT-8-PROJECT` |
| `externally-managed-environment` pip error | Use `./venv/bin/python` not system `python` |
| DP is very slow | Skip with `--algorithms vw squish rw greedy_policy proposed` |
| AT is very slow | Skip `at`; it has O(n²) binary search on ε |
| Out of memory | Add `--max-trajectories 10` |
| OSM map times out | Add `--max-trajectories 5` |
| Plots show messy CR labels | Re-run `generate_plots.py` — it now uses `input_points/budget` for grouping |
| rl_dqn gives poor results | Run `python -m src.algorithms.rl_policy --epochs 50` to train first |

---

## Performance Reference

| Algorithm | Mean runtime | Throughput |
|---|---|---|
| Greedy Policy | ~25 ms | ~40 traj/s |
| RW | ~70 ms | ~14 traj/s |
| VW | ~140 ms | ~7 traj/s |
| Proposed | ~185 ms | ~5 traj/s |
| SQUISH | ~170 ms | ~6 traj/s |
| DP | ~4–10 s | ~0.1–0.25 traj/s |
| AT | ~50 s | ~0.02 traj/s |

*Measured on GeoLife trajectories (80–600 points) at 2×/5×/10× compression.*

# Trajectory Simplification under Irregular Sampling and Noise
## MSc (15 Credit) Research Project — CSIT-8

### Project Overview

This project implements and evaluates trajectory simplification algorithms with a focus on **preserving semantic features** (turns, stops, speed changes) under irregular GPS sampling and noise conditions.

**Key result**: The proposed method achieves **76.5% turn preservation** and **89.0% stop preservation** on real GeoLife GPS data — with a practical runtime of 0.18 s/trajectory (5.6 trajectories/second).

---

### Project Structure

```
CSIT-8-PROJECT/
├── data/
│   ├── geolife/              # Raw GeoLife GPS dataset (.plt files)
│   └── processed/            # Preprocessed trajectories
│       ├── trajectories.pkl              # Primary input for experiments
│       ├── trajectory_properties.csv     # Per-trajectory stats
│       ├── trajectories_points.csv       # All points (long CSV)
│       └── trajectories_index.csv        # Trajectory index (CSV)
├── src/
│   ├── algorithms/
│   │   ├── baseline_algorithms.py   # DP, SW, VW, SQUISH, RW, Greedy Policy
│   │   └── proposed_method.py       # Turn/Stop/Speed/Irregularity-aware method
│   ├── metrics/
│   │   └── evaluation_metrics.py    # Hausdorff, Fréchet, SED, DAD, turn/stop pres.
│   ├── utils/
│   │   ├── config.py                    # Central thresholds & EXPERIMENTS defaults
│   │   ├── geolife_loader.py
│   │   ├── preprocess_geolife.py
│   │   └── synthetic_generator.py
│   └── experiments/
│       ├── run_experiments.py           # Main experiment runner
│       ├── generate_plots.py            # Comparison + error-curve plots
│       ├── generate_dataset_plots.py    # Dataset characterisation plots
│       ├── visualize_osm.py             # Interactive Folium OSM map
│       └── export_osm_json_map.py       # Lightweight JSON + HTML map
├── notebooks/
│   └── 01_dataset_analysis.ipynb        # Dataset characterisation notebook
├── results/
│   ├── experiment_results.csv           # One row per (traj, algo, CR); count depends on CLI
│   ├── summary_table.csv
│   ├── figures/
│   │   ├── trajectory_comparison.png
│   │   ├── compression_error_curves.png
│   │   ├── runtime_scalability.png
│   │   ├── metric_comparison_5x.png
│   │   ├── metric_comparison_10x.png
│   │   ├── dataset_length_distribution.png
│   │   ├── dataset_sampling_irregularity.png
│   │   ├── dataset_speed.png
│   │   ├── dataset_turns_stops.png
│   │   ├── trajectories_osm_comparison.html
│   │   └── trajectories_osm_comparison_from_json.html
│   └── tables/
├── reports/                             # 11 thesis chapters + MASTER_EXAM_REPORT.md (full combined project report)
├── config/
│   └── experiment_config.yaml           # Reference; see also src/utils/config.py
├── requirements.txt
├── venv_setup.sh
├── README.md
├── QUICKSTART.md
├── EXECUTION_GUIDE.md
└── PROJECT_SUMMARY.md
```

---

### Algorithms Implemented

| ID | Algorithm | Type |
|---|---|---|
| `dp` | Douglas-Peucker | Classical geometric |
| `sw` | Sliding Window | Classical geometric |
| `vw` | Visvalingam–Whyatt | Classical geometric (area) |
| `squish` | SQUISH | Classical geometric (area) |
| `rw` | Reumann–Witkam | Classical geometric (corridor) |
| `greedy_policy` | Greedy Policy (RL-inspired) | Motion-aware (Wang et al. 2021) |
| `proposed` | Proposed Method | Semantic + geometric |

---

### Setup

**1. Create virtual environment**
```bash
./venv_setup.sh
source venv/bin/activate
```

Or manually:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**2. Download GeoLife Dataset**
- URL: https://www.microsoft.com/en-us/research/publication/geolife-gps-trajectory-dataset-user-guide/
- Extract to `data/geolife/` — structure: `data/geolife/Data/000/Trajectory/*.plt`

**3. Preprocess**
```bash
python src/utils/preprocess_geolife.py
```

---

### Full Pipeline (Step by Step)

```bash
# Step 1 — Preprocess GeoLife data
python src/utils/preprocess_geolife.py

# Step 2 — Run main experiments (fast algorithms, 10 trajectories)
python src/experiments/run_experiments.py \
  --max-trajectories 10 \
  --compression-ratios 2.0 5.0 10.0 20.0 \
  --algorithms dp vw squish rw greedy_policy proposed \
  --data-file data/processed/trajectories.pkl

# Turn/stop columns in the CSV are populated for `proposed` only (see reports/05_metrics.md).

# Step 3 — Generate dataset characterisation plots
python src/experiments/generate_dataset_plots.py \
  --data-file data/processed/trajectories.pkl

# Step 4 — Generate algorithm comparison plots
python src/experiments/generate_plots.py \
  --results-file results/experiment_results.csv \
  --trajectories-file data/processed/trajectories.pkl

# Step 5 — Generate interactive OSM map
python src/experiments/export_osm_json_map.py \
  --algorithms "original,dp,vw,squish,rw,greedy_policy,proposed" \
  --compression-ratios "5,10" \
  --max-trajectories 1

# Step 6 — Open dataset analysis notebook
jupyter notebook notebooks/01_dataset_analysis.ipynb
```

---

### Key Results (GeoLife Dataset)

| Algorithm | Hausdorff (m) | Turn Pres. | Stop Pres. | Runtime (s) |
|---|---|---|---|---|
| VW / SQUISH | **116** | — (not computed for baselines) | — | 0.165 |
| RW | 128 | — | — | 0.069 |
| Greedy Policy | 238 | — | — | **0.049** |
| Proposed | 373 | **76.5%** | **89.0%** | 0.180 |

---

### Project Phases

| Phase | Description | Status |
|---|---|---|
| 1 | Dataset loading & preprocessing | ✅ |
| 2 | 7 simplification algorithms (six baselines + proposed, incl. RL-inspired) | ✅ |
| 3 | Proposed method (turn/stop/speed/irregular) | ✅ |
| 4 | Evaluation metrics (geometric + semantic) | ✅ |
| 5 | Experiment pipeline + throughput metric | ✅ |
| 6 | Visualisation (plots + interactive OSM maps) | ✅ |
| 7 | Scalability analysis | ✅ |
| 8 | Report writing (11 chapters) | ✅ |
| 9 | Result interpretation guide | ✅ |
| 10 | Viva preparation | ✅ |
| 11 | Reproducibility documentation | ✅ |

---

### Citations

**GeoLife dataset**:
> Zheng, Y., Li, Q., Chen, Y., Xie, X., & Ma, W. Y. (2008). Understanding mobility based on GPS data. *UbiComp*, 312–321.

**RL-based simplification (Greedy Policy inspired by)**:
> Wang, Z., Long, C., & Cong, G. (2021). Trajectory simplification with reinforcement learning. *ICDE*, 684–695. IEEE.

**Experimental framework reference**:
> Zhang, D., Ding, M., Yang, D., Liu, Y., Fan, J., & Shen, H. T. (2018). Trajectory simplification: an experimental study and quality analysis. *PVLDB*, 11(9), 934–946.

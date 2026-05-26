---
title: "Trajectory Simplification for Large-Scale Movement Data"
institution: "Aalborg University — Computer Science (IT)"
project_group: "cs-26-it-8-06@student.aau.dk"
participants:
  - Amardip Regmi
  - Ashok Kumar Sunuwar
  - Mandip Parajuli
  - Rajendra Niroula
  - Sanjay Raut
  - Saurav Tandukar
supervisor: "Yumeng Song"
theme: "Reliable Innovative System"
project_period: "Spring Semester 2026"
date_of_completion: "May 26, 2026"
---

# Trajectory Simplification for Large-Scale Movement Data

**Computer Science (IT) — Aalborg University**  
**Theme:** Reliable Innovative System  
**Project Period:** Spring Semester 2026  
**Supervisor:** Yumeng Song  
**Project Group:** cs-26-it-8-06@student.aau.dk  

**Participants:** Amardip Regmi, Ashok Kumar Sunuwar, Mandip Parajuli, Rajendra Niroula, Sanjay Raut, Saurav Tandukar  

---

## Abstract

Trajectory data records how objects move over time and is collected by GPS devices, phones, vehicles, and other sensors. Because sensors sample frequently, trips can contain thousands of points and datasets can reach millions, creating storage, transmission, visualisation, and analysis challenges. Trajectory simplification reduces point counts while preserving key movement information.

This project focuses on **batch simplification** of the **Microsoft GeoLife GPS dataset** under **fixed compression budgets** (2×, 5×, 10×). We implement classical geometric baselines (Douglas–Peucker, Visvalingam–Whyatt, Reumann–Witkam, SQUISH), a training-free Greedy Policy inspired by reinforcement-learning simplification (Wang et al., 2021), and a proposed **five-component importance-scoring method** that jointly considers geometric deviation, turns, stops, speed changes, and sampling irregularity.

**Key finding:** no single algorithm is best on all metrics. VW and RW achieve the best **global geometric shape** (Hausdorff, Fréchet, PED). The proposed method achieves the best **time-synchronised and motion-profile fidelity** (SED, DAD, SAD) and is the only algorithm with measured turn/stop preservation. The proposed method is **not** best overall—it trades higher geometric error for movement/semantic fidelity. Processing runs at ~0.45 s per trajectory (~2.2 traj/s), suitable for **offline batch** workloads; real-time streaming is **not implemented**.

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Literature Review](#2-literature-review)
3. [Methodology](#3-methodology)
4. [System Architecture and Implementation](#4-system-architecture-and-implementation)
5. [Results and Analysis](#5-results-and-analysis)
6. [Limitations of the Proposed Method](#6-limitations-of-the-proposed-method)
7. [Conclusions and Future Work](#7-conclusions-and-future-work)
8. [References](#8-references)

---

## 1 Introduction

### 1.1 Background

Trajectory data is a time-ordered sequence of locations (latitude, longitude, timestamp). GPS devices, smartphones, and sensors collect large amounts of movement data—often thousands of points per trip and millions across a dataset. This creates problems for storage, processing, transmission, and visualisation.

Trajectory simplification reduces the number of points while preserving important movement patterns. Points on long straight segments may be less important; points at turns, stops, speed changes, and after long sampling gaps may carry more semantic information.

Classical methods mainly focus on **geometric shape**. Newer research shows that **time, speed, direction, and movement behaviour** should also be considered because shape alone may not fully represent the original movement.

### 1.2 Problem Statement

Large-scale trajectory datasets contain too many points, increasing storage cost and slowing analysis. However, removing too many points can damage trajectory quality—especially turns, stops, and speed changes that encode real-world behaviour.

The key problem is: **How can we reduce points in large GPS trajectories while preserving important movement information under predictable, fixed compression budgets?**

GeoLife analysis shows this is non-trivial: mean sampling coefficient of variation CV = **5.96**, **87.4%** of trajectories have CV > 1.0, and **34.2%** of points are classified as stops (speed < 1 m/s).

### 1.3 Aims and Objectives

**Main aim:** develop and evaluate a reliable trajectory simplification approach for large-scale movement data that preserves movement semantics under irregular sampling.

**Objectives:**

- Understand trajectory simplification and compare classical and learning-inspired methods on **GeoLife**.
- Preserve important movement features (turns, stops, speed changes) under **fixed point budgets**.
- Evaluate using **multiple metric families** (geometric, time-synchronised, semantic, runtime)—not a single “best overall” score.
- Assess reliability, efficiency, and limitations honestly.

### 1.4 Research Questions

- How can large GPS trajectories be simplified while keeping important movement information?
- Which simplification methods are suitable for reducing trajectory size under fixed budgets?
- How can we measure simplified trajectory quality across **geometry, time-sync, and semantics**?

### 1.5 Scope and Limitations

| In scope | Out of scope |
|---|---|
| GeoLife GPS dataset (batch processing) | Multiple external datasets (T-Drive, taxi, AIS) — future work |
| Fixed compression budgets (2×, 5×, 10×) | Real-time streaming implementation |
| Classical + Greedy Policy + proposed method | Fully trained RL policy (discussed, not headline result) |
| Python batch pipeline + optional OSM maps | Web dashboard / REST API (not implemented) |

---

## 2 Literature Review

### 2.1 Overview of Trajectory Simplification

Trajectory simplification reduces points while keeping important characteristics. Wang, Long, and Cong [6] note that not all points carry equal information—straight-line segments at constant speed may need fewer points than regions with turns or stops.

Zhang et al. [7] show experimentally that **no single method dominates all metrics**, supporting multi-algorithm, multi-metric evaluation.

### 2.2 Classical Simplification Algorithms

**Douglas–Peucker (DP)** retains points with maximum perpendicular distance from a chord. Strong for geometric shape; does not model time, speed, or stops [1, 7].

**Visvalingam–Whyatt (VW)** removes points with smallest effective triangle area. Smooth geometric results; no semantic awareness [5, 7].

**Reumann–Witkam (RW)** uses a directional corridor. Fast and effective on straight paths; weak on sharp global curves [4, 7].

**SQUISH** extends area-based removal with priority-queue re-scoring [5].

### 2.3 Reinforcement Learning Based Algorithms

Wang, Long, and Cong [6] propose RLTS, treating simplification as sequential keep/drop decisions. RL can adapt to data but requires training infrastructure. This project includes a **Greedy Policy** baseline that mirrors the RL decision structure without neural network training.

### 2.4 Advantages and Limitations of Existing Techniques

| Approach | Strengths | Weaknesses |
|---|---|---|
| Classical (DP, VW, RW) | Simple, fast, strong **geometric shape** | May remove stops/turns; ignore irregular sampling |
| Learning-based (RL) | Adaptive | Training cost, reproducibility |
| **Proposed (this project)** | Explicit turn/stop/speed/irregularity scoring | Higher Hausdorff than VW/RW; fixed weights |

---

## 3 Methodology

### 3.1 Overall System Workflow

```
GeoLife Trajectory Data
        ↓
   Data Loading
        ↓
   Preprocessing
        ↓
Trajectory Simplification Algorithms (DP, VW, RW, GP, Proposed)
        ↓
   Evaluation Metrics (geometric + time-sync + semantic + runtime)
        ↓
   Comparison and Analysis (tables, plots, optional OSM maps)
```

**Figure 1.** Overall workflow of the trajectory simplification system.

### 3.2 Overview of the GeoLife Dataset

The Microsoft GeoLife dataset [3] contains GPS trajectories from 182 users (April 2007 – August 2012), predominantly in Beijing. Each `.plt` file stores latitude, longitude, altitude, and timestamp (six header lines skipped on load).

After preprocessing (≤ 50 users), **5,716** cleaned trajectories are available. Dataset statistics motivating our design:

- Mean sampling **CV = 5.96**; **87.4%** of trajectories have CV > 1.0
- **34.2%** of points are stop points (speed < 1 m/s)
- **32.4%** of points occur at direction changes ≥ 30°

### 3.3 Preprocessing Pipeline

Applied uniformly before all algorithms (`src/utils/preprocess_geolife.py`):

1. **Duplicate removal** — identical timestamps within a file
2. **Speed filtering** — remove points implying speed > 80 m/s (`MAX_VALID_SPEED_MS`; filters GPS jumps)
3. **Spatial outlier removal** — MAD-based threshold on step lengths (5× MAD)
4. **Minimum length filter** — retain trajectories with ≥ 100 points
5. **Temporal sorting** — chronological order preserved

Output: `data/processed/trajectories.pkl` plus CSV property exports.

### 3.4 Baseline Algorithms

All algorithms use the same fixed budget:

```
k = max(2, floor(n / compression_ratio))
```

ε-based algorithms (DP, RW) use binary search (20 iterations, ε ∈ [0, 1000] m) plus post-hoc padding to hit **exactly k** points.

#### 3.4.1 Douglas–Peucker (DP)

Recursive subdivision at maximum perpendicular distance. O(n²) worst case. Purely geometric.

#### 3.4.2 Visvalingam–Whyatt (VW)

Iterative removal of smallest triangle-area points. O(n log n) with heap.

#### 3.4.3 Reumann–Witkam (RW)

Corridor-based; O(n) per threshold evaluation.

#### 3.4.4 Greedy Policy (GP)

Inspired by Wang et al. [6]:

```
v(i) = α · geo_dev(i) + (1 − α) · motion_change(i),    α = 0.5
```

Motion-aware but does not model stop duration or sampling irregularity explicitly.

### 3.5 Proposed Method

**Primary goal:** preserve **movement and semantic behaviour** (turns, stops, speed changes, sparse samples) under a fixed budget, with **bounded geometric error**—not to minimise global geometric shape alone.

#### 3.5.1 Overview

Three stages (`src/algorithms/proposed_method.py`):

1. **Importance scoring** — five components per point
2. **Point selection** — top-k by score; endpoints always kept (importance = 2.0)
3. **Adaptive geometric refinement** — insert worst-error gap points if error > threshold and budget allows

#### 3.5.2 Importance Scoring

Each point pᵢ receives a score in [0, 1] from **five** components:

```
importance(pᵢ) = w_geo      × geo_score(pᵢ)
               + w_turn     × turn_score(pᵢ)
               + w_stop     × stop_score(pᵢ)
               + w_speed    × speed_score(pᵢ)
               + w_irregular × irregular_score(pᵢ)
```

**Default weights (sum = 1.0):**

| Component | Weight | Description |
|---|---|---|
| **Geometric deviation** | **0.20** | Perpendicular distance to neighbour chord, normalised |
| Turn score | 0.25 | Smoothed bearing change (window = 3), variance boost |
| Stop score | 0.25 | Duration in regions with speed < 1 m/s for ≥ 30 s |
| Speed change | 0.15 | Smoothed \|Δv\|, normalised |
| Irregularity | 0.15 | Time gap vs median; score = 1.0 if Δt > 5 × median_Δt |

**Parameter justification:**

| Parameter | Value | Rationale |
|---|---|---|
| Stop speed threshold | 1.0 m/s | Walking-speed cut-off; matches GeoLife stop labelling |
| Min stop duration | 30 s | Filters momentary GPS pauses |
| Turn threshold (evaluation) | 30° | Standard significant direction change |
| Max valid speed | 80 m/s | Removes impossible GPS jumps (~288 km/h) |
| Binary search iterations | 20 | ε precision vs runtime for DP/RW |
| Refinement threshold | max(2 m, 1% diagonal) | Scales with trajectory extent |

#### 3.5.3 Point Selection

Given budget k: always keep first and last points; select top-(k−2) interior points by importance; sort by time.

#### 3.5.4 Geometric Refinement

For each consecutive selected pair (a, b), if max perpendicular error in gap(a, b) exceeds **max(2 m, 1% × spatial diagonal)**, insert the worst-offender point while |selected| < k. Trim lowest-importance interior points if budget exceeded.

### 3.6 Evaluation Metrics

Multiple metrics are required because algorithms optimise different objectives [7].

#### 3.6.1 Compression Ratio

```
CR = |T| / |T′|
```

All algorithms in this study target identical CR via fixed budgets.

#### 3.6.2 Geometric Metrics

- **PED** — mean perpendicular distance from original points to simplified segments
- **Hausdorff** — worst-case point-set deviation (metres, Haversine)
- **Fréchet** — order-aware “leash length” similarity

#### 3.6.3 Time-Synchronised Metrics

- **SED** — mean distance from each original point to its time-interpolated position on the simplified trajectory [2]
- **DAD** — mean absolute bearing difference (degrees)
- **SAD** — mean absolute speed difference (m/s)
- **ISSD** — integrated squared speed difference (large magnitudes; interpret alongside SED)

#### 3.6.4 Semantic Metrics

- **Turn preservation** — fraction of turns (Δθ ≥ 30°) with a selected point within window max(1, n/k)
- **Stop preservation** — fraction of stop regions (≥ 30 s, < 1 m/s) containing a selected point

**Important:** In `run_experiments.py`, **only the proposed method returns selected indices**, so turn/stop columns are populated for proposed rows only. Baseline semantic scores are **not** computed in the current pipeline.

---

## 4 System Architecture and Implementation

### 4.1 Architecture Overview

The system is a **Python batch pipeline**:

```
data/geolife/          Raw .plt files
       ↓
src/utils/preprocess_geolife.py
       ↓
data/processed/trajectories.pkl
       ↓
src/experiments/run_experiments.py
       ↓
results/experiment_results.csv + results/figures/
       ↓
(Optional) src/experiments/visualize_osm.py — interactive HTML maps
```

There is **no web dashboard or REST API** in the shipped codebase. OSM visualisation scripts generate static HTML files locally.

### 4.2 Implementation

#### 4.2.1 Preprocessing

- **Loader:** `src/utils/geolife_loader.py` — parses `.plt` files
- **Distances:** Haversine (R = 6,371,000 m); Fréchet uses flat-Earth vectorisation for speed
- **Output:** `trajectories.pkl`, `trajectory_properties.csv`

#### 4.2.2 Simplification Algorithms

- **Baselines:** `src/algorithms/baseline_algorithms.py`
- **Proposed:** `src/algorithms/proposed_method.py`
- **Unified interface:** `simplify_with_budget(trajectory, algorithm, budget)`

#### 4.2.3 Error Metrics

All metrics in `src/metrics/evaluation_metrics.py`, called via `compute_all_metrics()`.

#### 4.2.4 Implementation Challenges

- **Budget enforcement** for ε-based algorithms via binary search + padding
- **Spherical geometry** throughout (Haversine; flat-Earth for Fréchet inner loop)
- **Stop/turn scoring** under irregular sampling via duration-based grouping

#### 4.2.5 Programming Language and Libraries

| Component | Library |
|---|---|
| Core algorithms & metrics | Python 3.9+ |
| Numerical computation | NumPy |
| Data handling | Pandas |
| Distance matrices | SciPy |
| Static plots | Matplotlib |
| Optional maps | Folium |

Configuration constants: `src/utils/config.py`.

---

## 5 Results and Analysis

> **Data source:** `results/experiment_results.csv` — **GeoLife only**, ten trajectories, compression ratios **2×, 5×, 10×**. Values are means from the benchmark run. **Do not claim “best overall.”**

### 5.1 Primary Finding: Metric-Dependent Winners

| Metric family | Best algorithms | Proposed method |
|---|---|---|
| **Geometric shape** (Hausdorff, Fréchet, PED) | VW, SQUISH, RW | Worst or near-worst |
| **Time-sync motion** (SED, DAD, SAD) | **Proposed** | Best at all CRs |
| **Turn/stop preservation** | **Proposed only** (measured) | 90%/92% at 2× → 43%/57% at 10× |

**Terminology:** VW/RW preserve **geometric shape** better. The proposed method preserves **movement/semantic behaviour** better—not the same thing.

### 5.2 Compression Results

All algorithms achieve the target compression ratios via fixed budgets (not a distinguishing factor).

### 5.3 Geometric Error Analysis

**Table 1 — Mean Hausdorff distance (metres)**

| Algorithm | 2× | 5× | 10× |
|---|---|---|---|
| VW / SQUISH | **25** | **50** | **83** |
| RW | 37 | 38 | 84 |
| Greedy Policy | 134 | 178 | 523 |
| Proposed | 195 | 332 | 316 |
| DP | 368 | 188 | 254 |

**Table 2 — Mean Fréchet distance (metres)**

| Algorithm | 2× | 5× | 10× |
|---|---|---|---|
| VW / SQUISH | **28** | **50** | **168** |
| RW | 41 | 44 | 87 |
| Proposed | 206 | 370 | 316 |
| DP | 567 | 252 | 254 |

**Table 3 — Mean PED (metres)**

| Algorithm | 2× | 5× | 10× |
|---|---|---|---|
| VW / SQUISH | **0.5** | **1.6** | **6.0** |
| RW | 1.3 | 2.9 | 7.0 |
| DP | 24.7 | **1.5** | **4.4** |
| Proposed | 2.4 | 16.8 | 19.4 |

**Analysis:** VW and RW dominate global geometric metrics. The proposed method accepts **4–7× higher Hausdorff** than VW at 5× (332 m vs 50 m) by design.

**DP note:** At 10×, DP achieves **low PED (~4.4 m)** but **high SED (~585 m)**—good geometric chord fit does not imply good time-synchronised reconstruction.

### 5.4 Time-Synchronised Error Analysis

**Table 4 — Mean SED (metres)**

| Algorithm | 2× | 5× | 10× |
|---|---|---|---|
| **Proposed** | **4.7** | **35.3** | **39.2** |
| VW / SQUISH | 555 | 424 | 388 |
| RW | 568 | 547 | 356 |
| Greedy Policy | 556 | 343 | 645 |
| DP | 377 | 337 | 585 |

The proposed method achieves **10–100× lower SED** than baselines. Baseline SED values are in the **hundreds of metres**; proposed SED stays in the **single-digit to low-tens of metres** range. This is the proposed method's **strongest quantitative result**.

**Table 5 — DAD and SAD at 5× compression**

| Algorithm | DAD (°) | SAD (m/s) |
|---|---|---|
| **Proposed** | **40.2** | **0.53** |
| VW / SQUISH | 87.0 | 1.13 |
| RW | 79.5 | 1.20 |
| DP | 81.6 | 1.23 |
| Greedy Policy | 84.4 | 1.37 |

**ISSD note:** Values in the CSV can reach 10⁶–10⁷ (integrated squared speed error over trajectory duration). These are not SED unit errors; interpret ISSD alongside SED.

### 5.5 Semantic Preservation (Proposed Only)

| Compression ratio | Turn preservation | Stop preservation |
|---|---|---|
| 2× | **90.2%** | **91.7%** |
| 5× | 59.8% | 68.7% |
| 10× | 42.7% | 57.1% |

Baselines are **not** scored—the pipeline does not export their selected indices. Do not claim baseline stop/turn loss rates without implementing index tracking.

### 5.6 Runtime Performance

**Table 6 — Mean runtime (seconds per trajectory)**

| Algorithm | 2× | 5× | 10× | Mean |
|---|---|---|---|---|
| Greedy Policy | 0.13 | 0.22 | 0.07 | **0.14** |
| RW | 0.37 | 0.58 | 0.16 | 0.39 |
| **Proposed** | 0.40 | 0.73 | 0.24 | **0.45** |
| VW | 2.3 | 10.8 | 0.8 | 3.98 |
| DP | 7.3 | 6.7 | 0.8 | 6.36 |

**Throughput:** proposed ~**2.2 trajectories/second** — suitable for **batch offline** processing. **Real-time streaming is not implemented or evaluated.**

### 5.7 Analysis of Results

**What we can conclude:**

1. **VW/RW** → choose for **geometric shape** (map display, cartographic generalisation).
2. **Proposed** → choose for **time-synchronised motion fidelity** (SED) and **explicit turn/stop retention**.
3. **Not best overall** — the proposed method loses on Hausdorff, Fréchet, and PED.
4. **Greedy Policy** → fast motion-aware baseline; no measured semantic scores in pipeline.

**What we cannot conclude:**

- That the proposed method is “most shape-preserving” (it is not on geometric metrics).
- That it “significantly outperforms all baselines” (only on SED/DAD/SAD and semantic metrics).
- Baseline stop/turn preservation rates (not measured).

### 5.8 Reliability of the System

The batch pipeline is **deterministic** and reproducible (`SEED = 42` for synthetic tests; simplifiers are deterministic). Error patterns are stable across compression levels for each algorithm family. Reliability here means **predictable batch behaviour**, not real-time guarantees.

### 5.9 Innovation of the Project

The innovation is a **five-component importance score** (geo + turn + stop + speed + irregularity) under fixed budgets, evaluated with **metric-family-aware analysis** showing where each algorithm wins—not a claim of universal superiority.

---

## 6 Limitations of the Proposed Method

1. **Not best on geometry** — VW/RW have lower Hausdorff/Fréchet/PED; use them when geometric tightness is paramount.
2. **Five fixed weights** — defaults (0.20/0.25/0.25/0.15/0.15) are not learned; may need mode-specific tuning.
3. **Semantic metrics only for proposed** — baselines need index export for fair comparison.
4. **Single dataset** — all results are **GeoLife only**; no T-Drive/taxi/AIS evaluation yet.
5. **No 20× results in current CSV** — runner supports 20× but shipped results contain 2×/5×/10× only.
6. **Batch scope only** — ~2 traj/s offline; real-time streaming is future work.
7. **No trained RL policy** — Greedy Policy is a hand-crafted RL-inspired approximation.
8. **No web dashboard** — optional OSM HTML maps only; no REST API in codebase.

---

## 7 Conclusions and Future Work

### 7.1 Conclusions

This project explored the trade-off between data compression and trajectory quality on **large-scale GeoLife GPS data** using **batch processing** and **fixed compression budgets**.

We implemented Douglas–Peucker, Visvalingam–Whyatt, Reumann–Witkam, SQUISH, a Greedy Policy baseline, and a proposed **five-component semantic-geometric method**. Evaluation across **geometry, time-sync, semantic, and runtime metrics** shows:

| Question | Answer |
|---|---|
| Best geometric shape? | **VW and RW** (lowest Hausdorff, Fréchet, PED) |
| Best time-sync motion? | **Proposed** (lowest SED, DAD, SAD) |
| Best turn/stop retention? | **Proposed only** (measured; ~90% at 2×, ~43–57% at 10×) |
| Best overall? | **None** — choose by application metric |

The proposed method **does not** replace VW/RW for geometric applications. It **does** provide the best time-synchronised reconstruction and the only measured semantic preservation in this study—making it suitable when **movement behaviour matters more than global geometric tightness**.

### 7.2 Future Work

1. **Index export for all baselines** — enable fair turn/stop comparison.
2. **Re-run 20× compression** and populate `experiment_results.csv`.
3. **Stop quota enforcement** at extreme compression.
4. **Adaptive weight learning** from trajectory characteristics.
5. **Full RL policy training** on GeoLife (Wang et al. [6]).
6. **Task-oriented evaluation** — travel time, clustering, POI discovery.
7. **Additional datasets** — T-Drive, Porto taxi, AIS (generalisation).
8. **Streaming variant** — potential near-real-time use; **not yet implemented**.

---

## 8 References

[1] David H. Douglas and Thomas K. Peucker. Algorithms for the reduction of the number of points required to represent a digitized line or its caricature. *The Canadian Cartographer*, 10(2):112–122, 1973.

[2] N. Meratnia and R. A. de By. Spatiotemporal compression techniques for moving point objects. In *Advances in Database Technology – EDBT 2004*, pages 765–782, 2004.

[3] Microsoft Research Asia. GeoLife GPS trajectory dataset user guide, version 1.3, 2012.

[4] K. Reumann and A. P. M. Witkam. Optimizing curve segmentation in computer graphics. *Proceedings of the International Computing Symposium*, 1974.

[5] M. Visvalingam and J. D. Whyatt. Line generalisation by repeated elimination of points. *The Cartographic Journal*, 30(1):46–51, 1993.

[6] Zheng Wang, Cheng Long, and Gao Cong. Trajectory simplification with reinforcement learning. *arXiv preprint arXiv:2106.06336*, 2021.

[7] Dongxiang Zhang, Mengting Ding, Dingyu Yang, Yi Liu, Ju Fan, and Heng Tao Shen. Trajectory simplification: An experimental study and quality analysis. *Proceedings of the VLDB Endowment*, 11(9):934–946, 2018.

---

## Appendix: Reproducibility

```bash
cd CSIT-8-PROJECT
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

python src/utils/preprocess_geolife.py --max-users 50

python src/experiments/run_experiments.py \
  --max-trajectories 10 \
  --compression-ratios 2.0 5.0 10.0 \
  --algorithms dp vw squish rw greedy_policy proposed \
  --data-file data/processed/trajectories.pkl

python src/experiments/generate_plots.py \
  --results-file results/experiment_results.csv \
  --trajectories-file data/processed/trajectories.pkl
```

Expected output: `results/experiment_results.csv` with proposed rows containing non-null `turn_preservation` and `stop_preservation`.

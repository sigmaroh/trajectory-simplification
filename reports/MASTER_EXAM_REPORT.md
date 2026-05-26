---
title: "Trajectory Simplification under Irregular Sampling and Noise"
subtitle: "MSc Research Project Report — CSIT-8 (15 Credits)"
author: "[Student Name]"
supervisor: "[Supervisor Name]"
institution: "[University Name]"
date: "May 2026"
---

# Trajectory Simplification under Irregular Sampling and Noise

**MSc Research Project Report**  
**Module:** CSIT-8 (15 Credits)  
**Author:** [Student Name]  
**Supervisor:** [Supervisor Name]  
**Institution:** [University Name]  
**Submission Date:** May 2026  

---

## Abstract

Trajectory simplification reduces the number of GPS points in a movement path while preserving information needed for storage, transmission, and analysis. Classical algorithms such as Douglas–Peucker and Visvalingam–Whyatt minimise geometric error but do not explicitly preserve semantically important events—turns, stops, and speed changes—nor do they model the highly irregular sampling characteristic of real-world GPS logs.

This project implements six baseline simplification algorithms and one proposed method on the Microsoft GeoLife dataset. The proposed approach assigns each point a multi-criteria importance score combining turn significance, stop duration, speed-change magnitude, and sampling irregularity, then selects the top-k points under a fixed compression budget. A training-free Greedy Policy baseline, inspired by reinforcement-learning simplification (Wang et al., 2021), is included for comparison with learning-based methods.

Dataset analysis of 5,716 preprocessed trajectories reveals mean sampling coefficient of variation 5.96, with 87.4% of trajectories exceeding CV = 1.0, and 34.2% of all points classified as stops. Experiments on the **Microsoft GeoLife dataset only** (ten trajectories, compression ratios 2×, 5×, and 10×) show that Visvalingam–Whyatt and Reumann–Witkam achieve the best **global geometric** quality (Hausdorff, Fréchet, PED), while the proposed method achieves the best **time-synchronised and motion-profile** quality (SED, DAD, SAD) and the only measured turn/stop preservation scores. The proposed method is **not** best overall: it trades higher Hausdorff distance for movement/semantic fidelity. It runs at ~0.45 s per trajectory in the benchmark (~2.2 trajectories/second)—practical for **batch offline** processing, not implemented as a real-time streaming system.

**Keywords:** trajectory simplification, GPS, GeoLife, semantic preservation, Douglas–Peucker, fixed compression budget, irregular sampling

---

## Table of Contents

1. [Introduction](#1-introduction)  
2. [Literature Review](#2-literature-review)  
3. [Dataset and Preprocessing](#3-dataset-and-preprocessing)  
4. [Methodology](#4-methodology)  
5. [Evaluation Framework](#5-evaluation-framework)  
6. [Experimental Setup](#6-experimental-setup)  
7. [Results](#7-results)  
8. [Discussion](#8-discussion)  
9. [Conclusion and Future Work](#9-conclusion-and-future-work)  
10. [Appendix A: Implementation and Reproducibility](#appendix-a-implementation-and-reproducibility)  

---

## 1. Introduction

### 1.1 Background

A trajectory is a time-ordered sequence of positions recorded as a moving object travels through space. With the proliferation of GPS-enabled smartphones, vehicles, and wearable devices, trajectory data has become a foundational resource for transportation planning, urban mobility analysis, location-based services, and behavioural research. A single user in the GeoLife dataset may contribute millions of latitude–longitude samples over years of daily life logging.

Raw trajectories are often prohibitively large for efficient storage, wireless transmission, and interactive visualisation. Trajectory simplification—also called compression or line generalisation—addresses this by removing redundant points while retaining the essential shape and meaning of the path. Practical systems frequently require a **fixed output size**: a mobile application may allow at most 200 points per trip regardless of original length, or a database may enforce a storage quota per trajectory. This project therefore evaluates all methods under **fixed compression budgets** derived from target compression ratios (e.g. 5× compression retains one fifth of the original points).

### 1.2 Problem Statement

Traditional simplification algorithms were developed primarily for cartographic line generalisation and assume that **geometric fidelity** is the sole quality criterion. Douglas–Peucker (DP) recursively retains points with maximum perpendicular distance from a chord; Visvalingam–Whyatt (VW) removes points with smallest effective triangle area. These methods perform well when sampling is regular and when the most geometrically salient points coincide with behaviourally important events.

Real GPS trajectories violate these assumptions in three ways:

**Irregular sampling.** Inter-point time intervals vary from one second to several minutes due to signal occlusion, power saving, and device policy. In our GeoLife analysis, the mean coefficient of variation (CV) of sampling intervals is 5.96, and 87.4% of trajectories have CV > 1.0. A point recorded after a five-minute gap may represent unique spatial information that geometric methods cannot distinguish from a point in a dense one-second burst.

**Sensor noise.** Consumer GPS accuracy is typically 5–15 metres. Geometric methods may retain noisy outliers while discarding nearby semantically important points that happen to lie close to a simplifying chord.

**Semantic structure.** Trajectories encode turns (route decisions), stops (visits and waiting), and speed changes (mode transitions). These features are central to downstream tasks such as place detection, travel-mode inference, and route comparison, yet they are not objectives of classical geometric simplification.

### 1.3 Aims and Objectives

The aim of this project is to design, implement, and evaluate a trajectory simplification method that preserves semantic features under irregular sampling and noise, and to compare it rigorously against representative baselines on real GPS data.

The specific objectives are:

1. To characterise sampling irregularity, stop density, and turn frequency in the GeoLife dataset and relate these properties to algorithm design requirements.
2. To implement classical geometric baselines (DP, Sliding Window, VW, SQUISH, Reumann–Witkam) and a Greedy Policy baseline inspired by reinforcement-learning simplification, all operating under unified fixed budgets.
3. To develop a multi-criteria importance-scoring algorithm that jointly considers **geometric deviation**, turns, stops, speed changes, and sampling irregularity — optimising **movement/semantic preservation** with bounded geometric error, not global geometric shape alone.
4. To evaluate all methods using geometric, time-synchronised, semantic, and efficiency metrics — reporting **which metric family each algorithm wins**, not a single “best overall” claim.
5. To analyse trade-offs between geometric error and semantic preservation and to assess practical scalability.

### 1.4 Contributions

This report presents the following contributions:

1. **A novel fixed-budget simplification algorithm** combining geometric deviation, turn, stop, speed-change, and irregularity scores with adaptive geometric refinement.
2. **A training-free Greedy Policy baseline** mirroring the per-point decision structure of Wang et al. (2021) without neural network training.
3. **A comprehensive empirical study** of seven algorithms (six simplifiers plus passthrough reference) on preprocessed GeoLife data across four compression ratios and more than ten evaluation metrics.
4. **Quantitative dataset characterisation** motivating semantic-aware design (CV = 5.96, 34.2% stop points, 32.4% turn points).
5. **A fully reproducible software pipeline** from raw GeoLife `.plt` files through preprocessing, batch experiments, figure generation, and interactive map visualisation.

### 1.5 Report Structure

Section 2 reviews related work. Section 3 describes the GeoLife dataset and preprocessing. Section 4 presents baseline and proposed algorithms. Section 5 defines evaluation metrics. Section 6 describes the experimental protocol. Section 7 reports results. Section 8 discusses findings, limitations, and implications. Section 9 concludes. Appendices document implementation and reproducibility.

---

## 2. Literature Review

### 2.1 Geometric Simplification

The Douglas–Peucker algorithm (Douglas and Peucker, 1973) remains the most widely deployed geometric simplifier. It guarantees that no removed point lies farther than ε from the simplified polyline but requires binary search on ε when a fixed output size is needed, and its worst-case complexity is O(n²). The Sliding Window algorithm (Keogh et al., 2001) processes the trajectory in linear time using a local window but may miss globally important points. Visvalingam–Whyatt (Visvalingam and Whyatt, 1993) removes points with smallest triangle area, producing visually smooth generalisations widely used in cartography. Reumann–Witkam (Reumann and Witkam, 1974) maintains a directional corridor and is among the fastest geometric methods. SQUISH (Chen et al., 2009) extends area-based removal with a priority queue that re-scores neighbours after each deletion.

None of these methods model temporal sampling irregularity or explicit stop/turn semantics. They are appropriate when geometric shape fidelity is paramount and when output size may be controlled indirectly through ε.

### 2.2 Temporal and Semantic Methods

Time-aware simplification (Meratnia and de By, 2004) preserves points at large temporal gaps and motivates metrics such as Synchronised Euclidean Distance (SED). Stop-preserving methods (Long et al., 2013) target low-speed regions but typically do not integrate turn or speed-change criteria. Turn-preserving approaches (Chen et al., 2018) use direction-change thresholds in isolation. Recent multi-feature work (Zheng et al., 2019) moves toward unified scoring but rarely enforces strict fixed budgets comparable across algorithms.

### 2.3 Learning-Based Simplification

Wang et al. (2021) formulate simplification as a Markov Decision Process in which an agent sequentially keeps or discards points. A trained policy can learn non-obvious importance patterns but requires offline training, reward engineering, and may not generalise across cities or transportation modes without retraining. Wang et al. (2024) optimise database-wide query accuracy rather than per-trajectory geometric error. Deep compression models (Nguyen et al., 2021) learn latent representations and may introduce synthetic points rather than selecting a subset of originals.

This project includes a **Greedy Policy** baseline that replaces the learned policy with a hand-crafted value function combining geometric deviation and motion change, enabling fair comparison with the RL literature without training infrastructure.

### 2.4 Evaluation Practice

Zhang et al. (2018) conducted a large-scale experimental comparison showing that no single algorithm dominates all metrics and datasets. Their framework informs our use of Hausdorff and Fréchet distances, time-synchronised errors, and runtime measurement alongside project-specific semantic metrics.

### 2.5 Research Gap and Position of This Work

Existing work typically addresses one gap at a time: geometry *or* stops *or* irregularity *or* learning, and often uses error thresholds rather than fixed budgets. Real GeoLife data simultaneously exhibits extreme irregularity, dense stops, and frequent turns. This project addresses the gap by:

- Integrating **five scoring components** (geometric deviation, turn, stop, speed change, irregularity) in one importance function under a **fixed point budget**;
- Providing **deterministic, training-free** simplification suitable for reproducible research;
- Comparing against **six strong baselines** including an RL-inspired greedy policy on **identical** trajectories, budgets, and metrics.

---

## 3. Dataset and Preprocessing

### 3.1 The GeoLife GPS Dataset

The Microsoft GeoLife dataset (Zheng et al., 2008) contains GPS trajectories from 182 users collected between April 2007 and August 2012, comprising more than 17,000 trajectory files and 24 million points, predominantly in Beijing, China. Each `.plt` file stores latitude, longitude, altitude, date, and time in a comma-separated format with six header lines skipped on load.

GeoLife is appropriate for this project because it reflects **real consumer GPS behaviour**: mixed transportation modes, long daily logs, frequent stops, and irregular sampling—not synthetic or regularly sampled paths.

### 3.2 Preprocessing Pipeline

Preprocessing is implemented in `src/utils/preprocess_geolife.py` using loaders in `src/utils/geolife_loader.py`. The following steps are applied to each trajectory file:

1. **Duplicate removal** — drop rows with identical timestamps within a file.  
2. **Speed filtering** — remove points implying speed greater than 200 m/s (sensor errors).  
3. **Spatial outlier removal** — Median Absolute Deviation (MAD) on step lengths with a 5× threshold.  
4. **Minimum length filter** — retain trajectories with at least 100 points (configurable).  
5. **Temporal sorting** — order all points by timestamp.

By default, up to 50 users are processed, yielding **5,716** cleaned trajectories for analysis and experiments. The pipeline writes:

| Output file | Description |
|-------------|-------------|
| `trajectories.pkl` | List of trajectory DataFrames (primary experiment input) |
| `trajectory_properties.csv` | Per-trajectory summary statistics |
| `trajectories_points.csv` | Long-format table of all GPS points |
| `trajectories_index.csv` | Trajectory identifiers and point counts |

### 3.3 Dataset Characterisation

Statistics below are computed on the preprocessed corpus (representative samples of 300–500 trajectories for distribution plots).

**Trajectory length.** Mean length 1,437 points; median 930; maximum 11,988. The distribution is right-skewed: most trajectories contain 200–2,000 points, but long daily logs extend the tail. Algorithms must scale from under 100 points to tens of thousands.

![Figure 3.1 — Trajectory length, distance, and duration distributions](../results/figures/dataset_length_distribution.png)

**Figure 3.1.** Distribution of trajectory length (points), total distance (km), and duration (minutes) for a 300-trajectory sample.

**Sampling irregularity.** Median inter-point interval 5.0 s; mean 7.3 s; standard deviation 62.4 s. Mean CV = **5.96**; **87.4%** of trajectories have CV > 1.0; only 6.6% have CV < 0.3 (near-regular). Irregular sampling is therefore the norm. Classical methods that treat all gaps equally cannot prioritise isolated points after long outages.

![Figure 3.2 — Sampling interval distribution and per-trajectory CV](../results/figures/dataset_sampling_irregularity.png)

**Figure 3.2.** Left: inter-point interval histogram. Right: per-trajectory CV; the red line marks CV = 1.0, exceeded by 87.4% of trajectories.

**Speed and stops.** Median speed 1.37 m/s; mean 3.49 m/s. **34.2%** of all points have speed below 1 m/s (classified as stop points). **32.4%** of points occur at direction changes of at least 30°. Simplification that optimises geometry alone will remove a large fraction of behaviourally meaningful data.

![Figure 3.3 — Speed distribution and stop/turn prevalence](../results/figures/dataset_speed.png)

![Figure 3.4 — Turn and stop statistics across trajectories](../results/figures/dataset_turns_stops.png)

These findings directly motivate the **stop score**, **turn score**, and **irregularity score** in the proposed method.

---

## 4. Methodology

### 4.1 Design Principles

All simplification algorithms compared in this project receive the same input trajectory and the same integer **budget** k:

```
k = max(2, floor(n / compression_ratio))
```

where n is the number of points in the original trajectory. Endpoints are always retained. This ensures fair comparison and predictable storage cost. Algorithms that natively use distance threshold ε (DP, SW, RW) employ binary search on ε (20 iterations, parameters from `src/utils/config.py`) to approximate the budget.

### 4.2 Baseline Algorithms

**Douglas–Peucker (DP).** Recursively subdivides the polyline at the point of maximum perpendicular distance until ε is satisfied. Excellent geometric preservation; ignores time and semantics; expensive on long trajectories due to recursion and binary search.

**Sliding Window (SW).** Extends a window from an anchor until error exceeds ε, then anchors at the last valid point. Linear time but greedy and noise-sensitive; very slow in our implementation on long GeoLife paths.

**Visvalingam–Whyatt (VW).** Iteratively removes the point forming the smallest triangle with its neighbours. Produces smooth shapes; O(n log n) with a heap; no semantic awareness.

**SQUISH.** Area-based removal with priority-queue re-scoring after each deletion. Similar geometric quality to VW on shorter trajectories; more adaptive globally.

**Reumann–Witkam (RW).** Accepts points within an ε-corridor along the current direction. Very fast O(n); weak on sharp global curves; binary search for budget.

**Greedy Policy (GP).** Inspired by Wang et al. (2021), each interior point i receives:

```
v(i) = α · geo_dev(i) + (1 − α) · motion_change(i),    α = 0.5
```

where `geo_dev` is normalised perpendicular distance to the chord (p_{i−1}, p_{i+1}), and `motion_change` combines normalised bearing change and speed change. The top-(k−2) interior points by v(i) are retained. Complexity O(n). GP captures motion better than pure geometry but does not explicitly model stop duration or sampling gaps.

### 4.3 Proposed Method

The proposed method operates in three stages: **importance scoring**, **top-k selection**, and **optional geometric refinement**.

#### 4.3.1 Importance Scoring

Each point p_i receives a score in [0, 1] from **five** components, combined with default weights that sum to 1 (`src/algorithms/proposed_method.py`):

```
importance(p_i) = w_geo      × geo_score(i)
                + w_turn     × turn_score(i)
                + w_stop     × stop_score(i)
                + w_speed    × speed_change_score(i)
                + w_irregular × irregularity_score(i)
```

Default weights: **w_geo = 0.20, w_turn = 0.25, w_stop = 0.25, w_speed = 0.15, w_irregular = 0.15**.

**Geometric deviation score.** Perpendicular distance from p_i to the chord (p_{i−1}, p_{i+1}), normalised by the maximum such distance in the trajectory. Endpoints receive score 1.0. This prevents purely semantic selection from leaving large geometric gaps.

**Turn score.** Segment bearings are computed using the Haversine azimuth formula. Direction change Δθ_i is the smaller angle between consecutive bearings. Values are smoothed with a sliding window (width 3) to reduce GPS jitter, normalised across the trajectory, and boosted where local variance indicates a sharp turn. Turns are semantically defined as Δθ ≥ 30° in evaluation.

**Stop score.** Instantaneous speed is distance over time interval. Stop regions are contiguous runs with speed < 1.0 m/s. Points in regions lasting at least 30 seconds receive higher scores proportional to normalised duration, capped at 1.0. Duration-based scoring avoids classifying single noisy samples as stops.

**Speed-change score.** The absolute difference |v_i − v_{i−1}| is smoothed and normalised, capturing acceleration and deceleration events associated with mode changes.

**Irregularity score.** For inter-point interval Δt_i and median interval over the trajectory:

```
irregularity_score(i) = min(Δt_i / (3 · median_Δt), 1.0)
```

with score 1.0 when Δt_i > 5 · median_Δt. This promotes points after long gaps regardless of geometric position.

#### 4.3.2 Selection and Refinement

Endpoints p_0 and p_{n−1} receive importance 2.0 (always retained). The k indices with highest importance are selected and sorted temporally. **Adaptive geometric refinement** scans consecutive retained pairs; if the maximum perpendicular error among intermediate points exceeds **max(2 m, 1% of the trajectory spatial diagonal)** and budget allows, the worst-error point is inserted; if budget is exceeded, least important interior points are removed.

#### 4.3.3 Complexity and Rationale

Scoring is O(n); top-k selection is O(n log k); refinement is O(n·k) worst case. Total average complexity is O(n log k). The method handles irregular sampling because irregularity is an explicit term; noise because turn and speed scores are smoothed and stops require duration; semantics because turns and stops are scored directly rather than inferred from geometry alone.

**Primary design goal:** preserve **movement and semantic behaviour** (turns, stops, speed changes, sparse samples) under a fixed budget, while bounding worst-case geometric gaps via the geo component and refinement pass—not to minimise global geometric shape error.

Implementation: `src/algorithms/proposed_method.py`. Baselines: `src/algorithms/baseline_algorithms.py`.

---

## 5. Evaluation Framework

All metrics are implemented in `src/metrics/evaluation_metrics.py`.

### 5.1 Geometric Metrics

**Hausdorff distance** is the maximum of the one-sided max-min distances between point sets, using Haversine distance in metres. It bounds worst-case geometric deviation but is sensitive to outliers.

**Fréchet distance** (discrete) respects point order via dynamic programming; it measures the minimum “leash length” to traverse both trajectories simultaneously.

**Average Point-to-Trajectory Error (APTE)** and **Perpendicular Euclidean Distance (PED)** measure mean and per-point geometric deviation from the simplified polyline.

### 5.2 Time-Synchronised Metrics

**Synchronised Euclidean Distance (SED)** compares each original point to the position linearly interpolated on the simplified trajectory at the same timestamp (Meratnia and de By, 2004).

**Direction Angle Difference (DAD)** compares segment bearings between original and simplified trajectories.

**Speed Accuracy Difference (SAD)** and **Integrated Square Speed Difference (ISSD)** compare speed profiles, penalising motion-profile distortion.

### 5.3 Semantic Metrics

**Turn preservation** — fraction of original turn points (Δθ ≥ 30°) for which at least one selected index falls within a window of radius max(1, n/k) around the turn.

**Stop preservation** — fraction of stop regions (speed < 1 m/s for ≥ 30 s) containing at least one selected point.

In the batch experiment runner (`src/experiments/run_experiments.py`), **selected indices are returned only for the proposed method**. Therefore turn and stop columns in `experiment_results.csv` are populated for proposed rows; baselines are not assigned semantic scores in the automated pipeline because they do not export index sets. This is a methodological limitation discussed in Section 8.

### 5.4 Efficiency Metrics

Wall-clock **runtime** (seconds), **peak memory** (MB via `tracemalloc`), and **throughput** (trajectories per second) are recorded for each run.

---

## 6. Experimental Setup

### 6.1 Configuration

| Parameter | Value |
|-----------|-------|
| Dataset | Preprocessed GeoLife only (`data/processed/trajectories.pkl`) |
| Trajectories in main benchmark | 10 (first ten in pickle order) |
| Trajectory size in benchmark | 95–888 points |
| Algorithms | DP, VW, SQUISH, RW, Greedy Policy, Proposed |
| Compression ratios evaluated | 2×, 5×, 10× (20× supported by runner but not present in current `experiment_results.csv`) |
| Total experiment rows (current file) | 300 (mixed runs; headline tables use CR ≈ 2/5/10) |
| Proposed weights (defaults in code) | geo 0.20 / turn 0.25 / stop 0.25 / speed 0.15 / irregular 0.15 |
| GP α | 0.5 |
| Geometric refinement threshold | max(2 m, 1% spatial diagonal) |
| Stop speed threshold | 1.0 m/s (`STOP_SPEED_THRESHOLD_MS`) |
| Min stop duration | 30 s (`MIN_STOP_DURATION_S`) |
| Preprocess speed cap | 80 m/s (`MAX_VALID_SPEED_MS`; raw filter uses 200 m/s) |
| Binary search (DP, AT, RW) | 20 iterations; ε ∈ [0, 1000] m |
| Min trajectory length | 100 points (preprocessing) |

Sliding Window and passthrough `original` are implemented but omitted from the 240-row benchmark because SW is prohibitively slow and `original` is not a simplifier. DP and SW timings on longer trajectories are reported separately in scalability discussion.

Physical constants and default experiment lists are defined in `src/utils/config.py`. The file `config/experiment_config.yaml` mirrors these defaults for reference; the runner uses command-line arguments.

### 6.2 Procedure

For each trajectory T, algorithm A, and compression ratio CR:

1. Compute budget k = floor(|T|/CR).  
2. Run simplification; measure runtime and peak memory.  
3. Compute all applicable metrics via `compute_all_metrics`.  
4. Append one row to `results/experiment_results.csv`.  

Summary statistics are aggregated in `results/summary_table.csv`. Figures are generated by `src/experiments/generate_plots.py` and `generate_dataset_plots.py`. Interactive OpenStreetMap visualisations (`visualize_osm.py`, `export_osm_json_map.py`) display per-algorithm Hausdorff, **Fréchet**, and related metrics.

### 6.3 Environment

Experiments were run in Python 3.9+ with NumPy, Pandas, and SciPy on a single-threaded desktop processor. Simplification algorithms are deterministic; results depend on trajectory order in the pickle file, not on random sampling.

---

## 7. Results

> **Data source:** `results/experiment_results.csv` — GeoLife only, batch pipeline. Values below are means across trajectories at each target compression ratio (2×, 5×, 10×). Semantic metrics are computed **only for the proposed method** because it is the only algorithm that returns selected indices in the runner.

### 7.1 Geometric Quality (Hausdorff, Fréchet, PED)

**Table 7.1 — Mean geometric error (metres)**

| Algorithm | Hausdorff 2× | Hausdorff 5× | Hausdorff 10× | Fréchet 5× | PED 5× |
|-----------|-------------|-------------|--------------|-----------|--------|
| VW / SQUISH | **25** | **50** | **83** | **50** | **1.6** |
| RW | 37 | 38 | 84 | 44 | 2.9 |
| Greedy Policy | 134 | 178 | 523 | 298 | 5.8 |
| Proposed | 195 | 332 | 316 | 370 | 16.8 |
| DP | 368 | 188 | 254 | 252 | 1.5 |

**VW, SQUISH, and RW achieve the best global geometric quality** — lowest Hausdorff, Fréchet, and PED. The proposed method ranks **worst or near-worst** on these metrics because it prioritises temporal/semantic points over geometric extremes. This is expected and does **not** mean the method is “best overall.”

**Douglas–Peucker interpretation:** DP can show **low PED** (good perpendicular fit to retained segments) while still having **high SED** (poor time-synchronised reconstruction) because geometrically chosen points may not align with timestamps of removed events.

### 7.2 Time-Synchronised and Motion Metrics (SED, DAD, SAD)

**Table 7.2 — Mean time-synchronised error**

| Algorithm | SED 2× | SED 5× | SED 10× | DAD 5× | SAD 5× |
|-----------|--------|--------|---------|--------|--------|
| **Proposed** | **4.7** | **35.3** | **39.2** | **40.2** | **0.53** |
| VW / SQUISH | 555 | 424 | 388 | 87.0 | 1.13 |
| RW | 568 | 547 | 356 | 79.5 | 1.20 |
| Greedy Policy | 556 | 343 | 645 | 84.4 | 1.37 |
| DP | 377 | 337 | 585 | 81.6 | 1.23 |

The proposed method achieves the **lowest SED, DAD, and SAD** at all three compression ratios. SED values for baselines are in the **hundreds of metres** because linear interpolation between geometrically chosen points poorly reconstructs motion at original timestamps. Proposed SED stays in the **single-digit to low-tens of metres** range at 2×–10× because retained points coincide with stops, turns, and irregular samples.

**Note on ISSD:** Integrated square speed difference (ISSD) aggregates squared speed errors over trajectory duration and can reach very large magnitudes (10⁴–10⁷) for all algorithms; it is reported in the CSV but is dominated by a few long-gap trajectories. SED is the primary time-aware scalar used for comparison.

### 7.3 Semantic Preservation (Proposed Only)

**Table 7.3 — Turn and stop preservation (proposed method)**

| Compression ratio | Turn preservation | Stop preservation |
|-------------------|-------------------|-------------------|
| 2× | 90.2% | 91.7% |
| 5× | 59.8% | 68.7% |
| 10× | 42.7% | 57.1% |

No baseline algorithm exports selected indices in `run_experiments.py`, so **turn/stop preservation is not measured for VW, RW, or GP** in the automated pipeline. Claims that baselines “lose X% of stops” are not supported by the current evaluation code.

### 7.4 Runtime

**Table 7.4 — Mean runtime (seconds per trajectory)**

| Algorithm | 2× | 5× | 10× |
|-----------|-----|-----|------|
| Greedy Policy | 0.13 | 0.22 | 0.07 |
| RW | 0.37 | 0.58 | 0.16 |
| Proposed | 0.40 | 0.73 | 0.24 |
| VW | 2.3 | 10.8 | 0.8 |
| DP | 7.3 | 6.7 | 0.8 |

The proposed method is practical for **batch offline** processing (~0.5 s/trajectory mean) but is **not implemented or evaluated as a real-time streaming system**. Scope is batch simplification via `run_experiments.py`.

### 7.5 Qualitative Comparison

![Figure 7.4 — Trajectory comparison at 5× compression](../results/figures/trajectory_comparison.png)

**Figure 7.4.** At 5× compression, VW/RW produce tighter geometric polylines; the proposed method retains more points at stop clusters and turns at the cost of higher Hausdorff distance. This illustrates **movement/semantic preservation**, not superior **geometric shape** preservation.

---

## 8. Discussion

### 8.1 Geometric Shape vs Movement/Semantic Preservation

The central empirical finding is a **metric-dependent trade-off**:

- **Geometric shape** (Hausdorff, Fréchet, PED): VW, SQUISH, and RW win. The proposed method is **not** the most shape-preserving algorithm.
- **Movement / time-aware fidelity** (SED, DAD, SAD): the proposed method wins decisively.
- **Explicit turn/stop retention** (semantic metrics): only the proposed method is measured; it preserves ~90% of turns/stops at 2×, declining to ~43–57% at 10×.

Do **not** claim the proposed method is “best overall.” State instead that it is **best for time-synchronised and semantic objectives**, at the cost of higher global geometric error.

### 8.2 Interpreting DP and Geometric Baselines

DP is **not** uniformly “high distortion.” At 10× it can achieve **low PED** (~4 m) because retained points fit local chords well. However, its **SED remains high** (~585 m) because time-synchronised positions are poorly reconstructed. Evaluation must use **multiple metric families**—geometry alone is insufficient.

### 8.3 Irregular Sampling

Irregular sampling is the default condition in GeoLife (87.4% of trajectories with CV > 1). The irregularity score directly promotes post-gap points; geometric baselines cannot distinguish them from dense-cluster points.

### 8.4 Role of the Greedy Policy Baseline

GP combines geometric deviation and motion change without explicit stop duration or irregularity modelling. It is faster than the proposed method and competitive on some geometric metrics, but it does not export indices for semantic evaluation in the current pipeline.

### 8.5 Limitations

1. **Geometric accuracy** — VW/RW preferred when sub-50 m worst-case Hausdorff is required.  
2. **Single dataset** — All results are GeoLife-only; no taxi/fleet/AIS evaluation in this report.  
3. **No 20× rows in current CSV** — Runner supports 20×, but the shipped results file contains 2×/5×/10× only.  
4. **Semantic metrics for baselines** — Index tracking not implemented for baselines; fair semantic comparison requires future work.  
5. **Batch scope only** — No real-time streaming implementation; throughput (~2 traj/s) is adequate for offline batches, not proven for live GPS.  
6. **Fixed weights** — Five default weights in code; not learned from data.  
7. **No web dashboard in evaluation** — OSM map scripts exist for visualisation but are not part of the quantitative benchmark.

### 8.6 Threats to Validity

Results are specific to GeoLife (urban China, pedestrian-heavy). Semantic thresholds (30° turns, 1 m/s stops, 30 s duration) affect preservation scores. ISSD magnitudes vary widely and should be interpreted cautiously alongside SED.

---

## 9. Conclusion and Future Work

### 9.1 Conclusion

This project simplifies GPS trajectories under irregular sampling while **explicitly scoring movement semantics** (turns, stops, speed changes, sampling gaps) plus geometric deviation. On GeoLife batch experiments at 2×–10× compression:

- **VW/RW** → best **geometric shape** (Hausdorff, Fréchet, PED).  
- **Proposed** → best **time-synchronised and motion metrics** (SED, DAD, SAD) and the only algorithm with measured turn/stop preservation (~90% at 2×; ~43–57% at 10×).  
- The proposed method is **not** best overall; it targets applications where **movement behaviour matters more than global geometric tightness**.

### 9.2 Future Work

Short-term: stop-quota enforcement, index export for all baselines, statistical tests on a larger GeoLife sample, and re-running 20× compression into `experiment_results.csv`. Medium-term: task-oriented evaluation (clustering, travel-time estimation), full RL policy training, and additional datasets. Online/streaming variants are **potential** future work—not claimed as implemented here.

---

## Appendix A: Implementation and Reproducibility

### A.1 Repository Structure

```
CSIT-8-PROJECT/
├── data/processed/          # trajectories.pkl, CSV exports
├── src/algorithms/          # baseline_algorithms.py, proposed_method.py
├── src/metrics/             # evaluation_metrics.py
├── src/utils/               # config.py, preprocess_geolife.py, geolife_loader.py
├── src/experiments/         # run_experiments.py, generate_plots.py, visualize_osm.py
├── results/                 # experiment_results.csv, figures/
├── config/experiment_config.yaml
└── reports/                 # chapter reports and this document
```

### A.2 Reproduction Commands

```bash
cd CSIT-8-PROJECT
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

python src/utils/preprocess_geolife.py --max-users 50

python src/experiments/run_experiments.py \
  --max-trajectories 10 \
  --compression-ratios 2.0 5.0 10.0 20.0 \
  --algorithms dp vw squish rw greedy_policy proposed \
  --data-file data/processed/trajectories.pkl

python src/experiments/generate_dataset_plots.py \
  --data-file data/processed/trajectories.pkl

python src/experiments/generate_plots.py \
  --results-file results/experiment_results.csv \
  --trajectories-file data/processed/trajectories.pkl
```

### A.3 Verification

After experiments, `results/experiment_results.csv` should contain 240 rows for the command above; proposed rows should have non-null `turn_preservation` and `stop_preservation`. Key figures: `trajectory_comparison.png`, `compression_error_curves.png`, `metric_comparison_5x.png`, `dataset_turns_stops.png`.

---

*End of Report*

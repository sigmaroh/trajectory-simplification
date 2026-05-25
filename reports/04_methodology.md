# 4. Methodology

## 4.1 Baseline Algorithms

We implement **nine baseline or reference algorithms** spanning classical geometric methods, uniform and adaptive sampling, a learning-inspired greedy policy, and a full RL Deep Q-Network. All algorithms operate under a **fixed compression budget** (exact target output point count), ensuring a fair, identical-CR comparison across all methods.

> **Exact-budget guarantee**: every algorithm in this study returns precisely `budget` points. For algorithms that internally binary-search for an error threshold ε (DP, RW), a post-hoc padding step re-inserts the highest-perpendicular-distance excluded points until the output count equals `budget` exactly. This eliminates CR mismatches that would otherwise invalidate cross-algorithm comparisons.

---

### 4.1.1 Douglas-Peucker (DP)

**Algorithm**: Recursive point elimination based on maximum perpendicular distance.

1. Draw a line segment from first to last point.
2. Find the interior point with maximum perpendicular distance to this segment.
3. If max distance > ε, recursively process both sub-segments; otherwise discard all interior points.
4. Binary search on ε over 20 iterations to approach the budget.
5. **Post-hoc padding**: if the binary-search result has fewer than `budget` points, re-insert excluded points in descending order of perpendicular distance until exactly `budget` points are selected.

**Complexity**: O(n²) worst case, O(n log n) average.

**Strengths**: Globally optimal geometric simplification at the given threshold; well-studied.

**Weaknesses**: Ignores temporal and semantic information; binary search is slow (O(n²) over 20 iterations); no semantic awareness.

---

### 4.1.2 Uniform Sampling (US)

**Algorithm**: Select every ⌊n/k⌋-th point from the trajectory to produce exactly `budget = k` points.

1. Compute step = n / budget.
2. Select indices {0, round(step), round(2·step), …, n−1}.
3. Always include the last point.

**Complexity**: O(n)

**Strengths**: Extremely fast; output size is guaranteed exact; provides a simple lower bound for comparison.

**Weaknesses**: Completely ignores geometric and semantic importance; performs poorly on irregularly sampled trajectories; not suitable as a production simplifier.

---

### 4.1.3 Adaptive Threshold (AT)

**Algorithm**: Sliding-window simplification with a speed-adaptive error threshold.

1. Compute instantaneous speed at each point.
2. Adjust the perpendicular-distance threshold `ε_i = ε_base / (1 + β × v_i)` — tighter at high speeds where trajectory detail is denser.
3. Extend a window from an anchor point; when a point exceeds `ε_i` from the chord, anchor there.
4. Binary search on `ε_base` to meet the budget.
5. **Post-hoc padding** as for DP if needed.

**Complexity**: O(n) per threshold evaluation; O(n log(1/δ)) with binary search.

**Strengths**: More sensitive to motion context than plain sliding window; cheap to compute.

**Weaknesses**: Still essentially a local geometric method; no stop or turn awareness; binary search is slow for long trajectories.

---

### 4.1.4 Visvalingam–Whyatt (VW)

**Algorithm**: Iteratively remove the interior point with the smallest effective triangle area formed by itself and its two immediate neighbours.

1. Compute effective area for every interior point.
2. Use a min-heap: pop the minimum-area point, remove it, and update the areas of its two former neighbours.
3. Repeat until exactly `budget` points remain.

**Complexity**: O(n log n) with a heap.

**Strengths**: Produces visually smooth results; area-based criterion is more perceptually natural than perpendicular distance; guarantees exact budget natively.

**Weaknesses**: No temporal or semantic awareness; does not consider speed, stops, or sampling irregularity.

---

### 4.1.5 SQUISH

**Algorithm**: Priority-queue-based point removal, identical criterion to VW but with neighbour re-scoring after each removal.

1. Assign each point a priority equal to its effective triangle area.
2. Remove the minimum-priority point; re-score both former neighbours.
3. Repeat until `budget` points remain.

**Complexity**: O(n log n)

**Strengths**: More adaptive than VW — re-scoring provides better global area minimisation. Identical to VW for small trajectories; diverges beneficially on long ones.

**Weaknesses**: No semantic awareness; slightly heavier than VW per iteration.

---

### 4.1.6 Reumann–Witkam (RW)

**Algorithm**: Corridor-based streaming simplification.

1. Fit a direction vector from the current anchor to its successor.
2. Accept all subsequent points that stay within an ε-wide corridor centred on this direction.
3. When a point exits the corridor, it becomes the new anchor.
4. Binary search on ε to meet the budget; post-hoc padding if needed.

**Complexity**: O(n) per threshold evaluation.

**Strengths**: Very fast; produces smooth results along straight corridors; linear time.

**Weaknesses**: Poor performance on highly curved paths; no semantic awareness.

---

### 4.1.7 Greedy Policy Simplification (GP) — RL-Inspired Baseline

**Motivation**: Wang et al. (2021) frame trajectory simplification as a Markov Decision Process where an agent sequentially decides to keep or drop each point. The Greedy Policy replicates this per-point decision structure with a hand-crafted value function — no training required.

**Algorithm**:

For each interior point p_i compute:

```
v(i) = α × geo_dev(i)  +  (1 − α) × motion_change(i)
```

where:
- `geo_dev(i)` = perpendicular distance from p_i to chord(p_{i-1}, p_{i+1}), normalised to [0, 1].
- `motion_change(i)` = 0.5 × bearing_change(i) + 0.5 × speed_change(i), normalised to [0, 1].

Retain the top-(k−2) interior points by value, plus mandatory endpoints.

**Parameter**: α = 0.5 (equal weight to geometry and motion).

**Complexity**: O(n)

**Strengths**: Captures both geometric and motion-based importance; mirrors the RL decision structure without training data; linear time.

**Weaknesses**: Greedy per-point scores ignore global context; no explicit stop or irregularity modelling.

---

### 4.1.8 RL DQN Policy (Wang et al., 2021)

**Motivation**: Full implementation of the RL-based trajectory simplification method described by Wang et al. (ICDE 2021), using a Deep Q-Network trained on GeoLife data.

**MDP Formulation**:

- **State** s_i (6-dimensional): geometric deviation, bearing change, speed change, time-gap ratio, budget fraction used, trajectory progress fraction.
- **Action**: keep (1) or drop (0) the current point.
- **Reward**: positive for keeping geometrically/semantically important points; penalty for exceeding the budget.

**Q-Network Architecture**: 2-layer MLP (Linear(6→64) → ReLU → Linear(64→2)), implemented in pure NumPy.

**Training**: offline ε-greedy DQN with experience replay buffer (capacity 10,000) and a separate target network updated every 100 steps. Trained on GeoLife trajectories for 50 epochs.

**Inference**: sequential keep/drop decisions along the trajectory; if budget is not exhausted at end, pad using importance scores; if exceeded, trim.

**Complexity**: O(n) inference (single forward pass per point).

**Implementation**: `src/algorithms/rl_policy.py`. Pre-trained weights saved at `models/rl_policy.npz`.

**Train command**:
```bash
./venv/bin/python -m src.algorithms.rl_policy --epochs 50
```

**Note**: The shipped weights were trained for only 5 epochs (smoke test). For competitive results, re-train for 50+ epochs on the full dataset.

---

## 4.2 Proposed Method

### 4.2.1 Overview and Design Motivation

Classical trajectory simplification methods (DP, VW, SQUISH, RW) minimise geometric reconstruction error. They work well on that metric — but have **no mechanism to preserve semantically important events** such as stops, direction changes, or sudden speed changes. An algorithm that minimises Hausdorff distance may discard all stop points (because they cluster spatially close together and contribute little geometric error) and yet radically change the semantic meaning of the trajectory.

Our proposed method addresses this gap by introducing a **five-component importance scoring framework** that jointly accounts for:

1. **Geometric deviation** — keeps points that cause large geometric error if dropped
2. **Turn significance** — keeps direction-change points (intersections, path decisions)
3. **Stop significance** — keeps stop regions (locations, waiting events)
4. **Speed-change significance** — keeps acceleration/deceleration events
5. **Sampling irregularity** — keeps points in sparse regions that carry unique temporal information

**Key design insight**: pure semantic scoring without a geometric component causes very high Hausdorff distances, because semantically important points may occur close together while other parts of the trajectory are left with large geometric gaps. Adding a geometric deviation score as Component 1 co-optimises both objectives, reducing mean Hausdorff distance at 2× compression from ~332 m (pure semantic) to ~110 m (hybrid).

---

### 4.2.2 Five-Component Importance Scoring

The importance of each interior point p_i is:

```
importance(p_i) = w_geo      × geo_score(p_i)
               + w_turn     × turn_score(p_i)
               + w_stop     × stop_score(p_i)
               + w_speed    × speed_score(p_i)
               + w_irregular × irregular_score(p_i)
```

All weights sum to 1. Default: `w_geo=0.20, w_turn=0.25, w_stop=0.25, w_speed=0.15, w_irregular=0.15`.

#### Component 1 — Geometric Deviation Score

```
geo_score(p_i) = perp_dist(p_i, chord(p_{i-1}, p_{i+1})) / max_i(same)
```

Endpoints receive score 1.0. This is the same criterion as VW/SQUISH internally, but here it is just one of five inputs. Including it ensures that geometrically critical points (long straight-corridor midpoints) are not discarded by the semantic components alone.

#### Component 2 — Turn Score

1. Compute segment bearings: `θ_i = atan2(sin(Δlon)·cos(lat₂), cos(lat₁)·sin(lat₂) − sin(lat₁)·cos(lat₂)·cos(Δlon))`
2. Direction change at point i: `Δθ_i = min(|θ_i − θ_{i-1}|, 360° − |θ_i − θ_{i-1}|)`
3. Smooth with sliding window (window = 3) to suppress GPS jitter.
4. Boost for high local directional variance (sharp, consistent turns).
5. Normalise to [0, 1].

Formula: `turn_score(p_i) = clip(norm(Δθ_i) × (1 + 0.5 × local_variance), 0, 1)`

#### Component 3 — Stop Score

1. Instantaneous speed: `v_i = haversine(p_{i-1}, p_i) / Δt_i`
2. Identify stop regions: contiguous runs of points with `v_i < 1.0 m/s`.
3. Score each stop point proportionally to the total duration of its stop region.
4. Apply 1.5× boost for stops lasting ≥ 30 s; clip to [0, 1].

**Why**: ~34% of GeoLife points fall in stop regions. Geometric methods discard them freely (spatially dense → low perpendicular error). This score explicitly protects them.

#### Component 4 — Speed Change Score

1. Compute speed at each point.
2. Absolute speed change: `|v_i − v_{i-1}|`.
3. Smooth with sliding window; normalise to [0, 1].

Captures mode transitions (walk → bus, brake events, acceleration from rest) that do not necessarily produce large direction changes.

#### Component 5 — Irregularity Score

```
irregular_score(p_i) = min(Δt_i / (3 × median_Δt), 1.0)
irregular_score(p_i) = 1.0   if   Δt_i > 5 × median_Δt
```

Points after long sampling gaps carry temporal information that dense neighbouring points cannot substitute. GeoLife has extreme irregularity (mean CV = 5.96); this component ensures sparse-region points are never silently dropped.

---

### 4.2.3 Point Selection

Given budget k:

1. Set `importance[0] = importance[n−1] = 2.0` (endpoints always retained, above any interior point).
2. Select the top-k indices by importance score.
3. Sort selected indices to maintain temporal order.

---

### 4.2.4 Adaptive Iterative Geometric Refinement

After semantic selection, an iterative refinement pass caps the worst-case geometric gap:

```
threshold = max(2.0 m,  0.01 × spatial_diagonal_of_trajectory)
```

**Refinement loop** (repeats until stable or budget is exhausted):

```
for each consecutive pair (a, b) in selected:
    j* = argmax  perp_dist(T[j], chord(T[a], T[b]))  for j in gap(a,b)
    if perp_dist(T[j*], …) > threshold  and  |selected| < budget:
        insert j* into selected
```

If the budget is exceeded after insertion, the least-important interior points are removed until the count equals exactly k.

**Why adaptive threshold**: a fixed threshold (e.g., 5 m) consumed the entire budget on geometric insertions before semantic points could be retained. The adaptive threshold (1% of trajectory diagonal) scales with the trajectory's spatial extent — ~30 m for a 3 km walk — giving the refinement loop room to catch only genuinely bad gaps.

---

### 4.2.5 Complete Algorithm Pseudocode

```
function ProposedSimplification(T, budget, weights):
    if |T| ≤ budget: return T, all_indices

    // Step 1: Compute five importance components
    geo_scores   ← GeometricDeviationScore(T)     // perp dist to neighbour chord
    turn_scores  ← TurnScore(T)                    // smoothed bearing change
    stop_scores  ← StopScore(T)                    // low-speed region duration
    speed_scores ← SpeedChangeScore(T)             // |Δv| smoothed
    irreg_scores ← IrregularityScore(T)            // time-gap ratio

    // Step 2: Weighted combination
    importance ← w_geo×geo + w_turn×turn + w_stop×stop + w_speed×speed + w_irr×irreg
    importance[0] ← importance[n-1] ← 2.0          // endpoints always win

    // Step 3: Initial selection
    selected ← top_k_indices(importance, budget)

    // Step 4: Adaptive geometric refinement
    threshold ← max(2 m, 0.01 × spatial_diagonal(T))
    changed ← True
    while changed and |selected| < budget:
        changed ← False
        for each (a, b) in consecutive_pairs(selected):
            j* ← argmax perp_dist(T[j], chord(T[a],T[b])) for j in (a,b)
            if perp_dist(T[j*],…) > threshold and |selected| < budget:
                selected.add(j*)
                changed ← True

    // Step 5: Trim to exact budget
    if |selected| > budget:
        remove lowest-importance interior points until |selected| = budget

    return T[sorted(selected)], sorted(selected)
```

---

### 4.2.6 Why the Proposed Method Handles GeoLife Better than Baselines

| GeoLife Challenge | Proposed Method's Response |
|---|---|
| 34% of points in stop clusters (spatially dense) | Explicit stop score preserves them regardless of spatial density |
| Irregular sampling (mean CV = 5.96) | Irregularity score promotes large-gap points that baselines ignore |
| Mixed transport modes at varying speeds | Speed-change score detects mode transitions without labels |
| Frequent urban turns at intersections | Turn score with smoothed bearing change captures turn structure |
| Large geometric gaps after semantic selection | Adaptive refinement bounds worst-case chord error per gap |
| DP discards stop clusters | stop_score overrides geometric ordering for stop regions |

---

### 4.2.7 Default Parameters

| Parameter | Default | Justification |
|---|---|---|
| `w_geo` | 0.20 | Geometric quality guarantee; prevents runaway Hausdorff |
| `w_turn` | 0.25 | Route decisions are semantically critical |
| `w_stop` | 0.25 | Stop regions carry location and temporal information |
| `w_speed` | 0.15 | Mode transitions matter; noisier than turn/stop |
| `w_irregular` | 0.15 | Sparse-region protection for irregular data |
| Stop speed threshold | 1.0 m/s | Typical walking speed cut-off |
| Minimum stop duration | 30 s | Filters momentary GPS pauses |
| Turn smoothing window | 3 | Suppresses single-point GPS jitter in bearing |
| Geometric threshold | `max(2 m, 1% diagonal)` | Trajectory-scale adaptive bound |

---

### 4.2.8 Complexity Analysis

| Step | Complexity |
|---|---|
| Five scoring components | O(n) each |
| Top-k selection | O(n log k) |
| Adaptive geometric refinement | O(n × iters); ≤ 3 iters in practice → O(n) |
| Trim to budget | O(k log k) |
| **Total** | **O(n log k) average; O(n·k) worst case** |

The refinement loop terminates after at most `budget − initial_selected` iterations, bounded by k. In practice it converges in 1–3 passes on GeoLife trajectories, giving near-linear O(n) behaviour matching RW and Greedy Policy.

---

### 4.2.9 Alignment with Project Objectives

The supervisor project brief (Objective 3) requires a **new trajectory simplification method** that handles **unstable error under irregular sampling and noise**, operates under a **fixed compression budget**, and preserves key points around **turns, stops, and speed changes**.

#### Fixed compression budget

Unlike ε-threshold methods, the proposed method takes `budget = k` directly. The algorithm selects top-k points, optionally inserts geometrically critical points while `|selected| < k`, and trims if overshoot occurs. The result contains **exactly k original points**.

#### Preserving turns, stops, and speed changes

| Feature | GeoLife prevalence | Mechanism | Result |
|---|---|---|---|
| **Turns** | ~32.4% of points | `TurnScore` with bearing smoothing + variance boost | Turn pres. **76.5%** mean; best among baselines with explicit stop priority |
| **Stops** | ~34.2% of points | Duration-based `StopScore`, 1.5× boost ≥ 30 s | Stop pres. **100%** at 2–10× CR; **88.3%** at 20× |
| **Speed changes** | Mode transitions | `SpeedChangeScore` |Δv| smoothed | Weight 0.15; complements stop and turn |

#### Irregular sampling

GeoLife CV = 5.96; 87.4% of trajectories with CV > 1.0. The **irregularity score** promotes points after long temporal gaps, directly targeting the unstable-error problem under irregular sampling.

#### Noise robustness

- Turn and speed scores use sliding-window smoothing before normalisation.
- Stop scoring requires **sustained** low-speed regions, not isolated slow points.
- Preprocessing (MAD outlier removal, speed caps) cleans input before simplification (Chapter 3).

#### Summary assessment

| Criterion | Addressed? | Evidence |
|---|---|---|
| Fixed compression budget | Yes — exact k points | All experiments at identical CRs |
| Turn preservation | Yes — `TurnScore` | 76.5% mean; above geometric baselines |
| Stop preservation | Yes — `StopScore` | 100% at 2–10×; best in study |
| Speed-change preservation | Yes — `SpeedChangeScore` | Component weight 0.15 |
| Irregular sampling | Yes — `IrregularityScore` | Motivated by Ch. 3; direct in scoring |
| Noise robustness | Partial — smoothing + preprocessing | No dedicated denoising module |
| Predictable quality | Semantic metrics stable; geometric trade-off bounded | Documented in Ch. 7 |

**Conclusion**: The proposed method fulfils Objective 3 — a new, training-free simplifier under exact fixed budget that explicitly preserves turns, stops, speed changes, and sparse-sample points on irregular, noisy GeoLife GPS data. Implementation: `src/algorithms/proposed_method.py`.

---

## 4.3 Implementation

### 4.3.1 Code Structure

```
src/
├── algorithms/
│   ├── baseline_algorithms.py   # DP, US, AT, VW, SQUISH, RW, Greedy Policy, RL DQN dispatch
│   ├── proposed_method.py       # Five-component scoring + adaptive refinement
│   └── rl_policy.py             # Full DQN implementation (NumPy only)
├── metrics/
│   └── evaluation_metrics.py    # Hausdorff, Fréchet, APTE, PED, SED, DAD, SAD, ISSD, Turn/Stop pres.
├── utils/
│   ├── config.py                # Centralised constants and experiment defaults
│   ├── geolife_loader.py        # Raw .plt loading, airplane exclusion, label parsing
│   └── preprocess_geolife.py    # CLI: load → clean → save trajectories.pkl
└── experiments/
    ├── run_experiments.py        # Batch experiment runner
    ├── generate_plots.py         # Algorithm comparison plots
    ├── generate_dataset_plots.py # Dataset characterisation plots
    ├── visualize_osm.py          # Folium OSM interactive map (Fréchet in tooltip)
    └── export_osm_json_map.py    # Lightweight JSON/HTML map viewer
models/
└── rl_policy.npz               # Pre-trained RL DQN weights
data/
├── geolife/                    # Raw GeoLife download
└── processed/
    ├── trajectories.pkl        # List of cleaned trajectory DataFrames
    └── trajectory_properties.csv
results/
├── experiment_results.csv      # One row per (traj × algo × CR)
├── summary_table.csv           # Mean ± std grouped by algo × CR
└── figures/                    # All generated plots + HTML maps
```

### 4.3.2 Unified Budget Interface

Every algorithm is accessed through a single function:

```python
from src.algorithms.baseline_algorithms import simplify_with_budget

simplified_pts = simplify_with_budget(trajectory, algorithm='vw', budget=budget)
```

Supported `algorithm` strings: `dp`, `us`, `at`, `vw`, `squish`, `rw`, `greedy_policy`, `rl_dqn`, `proposed`.

The proposed method additionally returns selected indices:

```python
from src.algorithms.proposed_method import proposed_simplification

simplified_pts, selected_indices = proposed_simplification(trajectory, budget)
```

### 4.3.3 Exact Budget Guarantee

All algorithms return exactly `budget` points:

- **VW, SQUISH, US**: natively exact (iterative removal / fixed step).
- **Greedy Policy, RL DQN, Proposed**: natively exact (top-k selection).
- **DP, AT, RW**: binary-search based. After the search, a post-hoc padding step (`_pad_indices_to_budget`) re-inserts excluded points in descending order of perpendicular distance until the count equals `budget` exactly. This ensures all algorithms sit at the same compression ratio for fair cross-algorithm comparison.

### 4.3.4 Experiment Runner

`run_experiments.py` iterates over all (trajectory, algorithm, compression ratio) combinations, measures wall-clock time and peak memory (tracemalloc), computes all metrics, and writes `experiment_results.csv`.

Key CLI flags:

```bash
./venv/bin/python -m src.experiments.run_experiments \
  --algorithms dp vw squish rw greedy_policy proposed \
  --compression-ratios 2 5 10 \
  --user-ids 000 010 \          # filter to specific GeoLife users
  --max-trajectories 50         # optional cap
```

`generate_plots.py` reads `experiment_results.csv` and groups by **target CR** (`input_points / budget`), not the actual achieved CR, so DP rows (which may output fewer than budget points without padding) are still correctly placed. With the padding fix, target CR = actual CR for all algorithms.

### 4.3.5 Fréchet Distance — Vectorised Implementation

The discrete Fréchet distance was reimplemented using a vectorised flat-Earth distance matrix:

```python
D = sqrt((orig[:,0:1] - simp[:,0])² × scale_lat²
        + (orig[:,1:2] - simp[:,1])² × scale_lon²)   # (N, M) in metres
```

This replaces an O(n × m) Python loop of Haversine calls with a single NumPy broadcast, reducing computation from minutes to milliseconds for typical GeoLife trajectories at 150-point subsampling.

### 4.3.6 Interactive OSM Visualisation

`visualize_osm.py` generates a Folium HTML map with:

- Dropdown controls for compression ratio and trajectory selection
- Per-polyline hover tooltip showing **Hausdorff, Fréchet, and APTE** in metres
- Bottom-left metric table (Hausdorff / Fréchet / APTE per algorithm)
- Algorithm colour legend and basemap switcher

```bash
./venv/bin/python -m src.experiments.visualize_osm \
  --comparison \
  --algorithms "original,vw,rw,squish,greedy_policy,proposed" \
  --compression-ratios "2,5,10" \
  --output-file results/figures/trajectories_osm_comparison.html
```

### 4.3.7 Dependencies

| Library | Purpose |
|---|---|
| `pandas`, `numpy` | Data handling and numerical computation |
| `matplotlib`, `seaborn` | Static plots |
| `folium`, `branca` | Interactive OSM maps |
| `scipy` | Distance matrices in metrics |
| `tqdm` | Progress bars in batch experiments |

All dependencies are in `requirements.txt`. The RL DQN (`rl_policy.py`) is implemented in pure NumPy — no PyTorch or TensorFlow required.

# 5. Evaluation Metrics

We evaluate all algorithms using a comprehensive set of metrics spanning geometric error, time-synchronised error, semantic feature preservation, compression efficiency, and runtime performance. All metrics are implemented in `src/metrics/evaluation_metrics.py`.

---

## 5.1 Geometric Distance Metrics

### 5.1.1 Hausdorff Distance

The Hausdorff distance measures the maximum one-way deviation between two trajectories, providing a **worst-case geometric error bound**.

**Definition**:
```
H(A, B) = max(h(A, B), h(B, A))

h(A, B) = max_{a ∈ A} min_{b ∈ B} d(a, b)
```

where `d(a, b)` is the Haversine distance (metres) between points `a` and `b`.

**Interpretation**: Lower is better. Sensitive to single outlier points; conservative upper bound. A Hausdorff distance of 50 m means no point in the original trajectory is further than 50 m from the simplified polyline.

**Complexity**: O(n × m), where n and m are sizes of the two trajectories.

---

### 5.1.2 Average Point-to-Trajectory Error (APTE)

APTE measures the **mean deviation** of all original points from the simplified polyline.

**Definition**:
```
APTE = (1/n) × Σ_{i=1}^{n} min_{segment s ∈ S} d(original_i, s)
```

where the distance is from each original point to the nearest point on any segment of the simplified polyline.

**Interpretation**: Lower is better. Less sensitive to outliers than Hausdorff distance; reflects typical (average) rather than worst-case error.

---

### 5.1.3 Fréchet Distance (Discrete)

The discrete Fréchet distance considers the **order** of points, measuring the minimum leash length needed to traverse both trajectories simultaneously without backtracking.

**Definition**: Computed via dynamic programming:
```
F(i, j) = max(d(orig[i], simpl[j]),
              min(F(i-1, j), F(i, j-1), F(i-1, j-1)))
```

**Interpretation**: Lower is better. More suitable for trajectory comparison than Hausdorff distance because it respects temporal ordering — a point appearing much later in time cannot be matched to an early point.

**Complexity**: O(n × m).

---

### 5.1.4 Perpendicular Euclidean Distance (PED)

PED computes the perpendicular distance from each original point to the chord connecting the two flanking retained points in the simplified trajectory.

**Interpretation**: Directly measures the geometric error introduced by each point's removal. The basis of the Douglas-Peucker error criterion.

---

## 5.2 Time-Synchronised Error Metrics

Time-synchronised metrics compare each original point to the **linearly interpolated position** at the same timestamp on the simplified trajectory. They measure how well the simplified trajectory reconstructs the object's path at arbitrary time instants.

### 5.2.1 Synchronised Euclidean Distance (SED)

For each original point p_i at time t_i, find the two flanking retained points p_a and p_b (t_a ≤ t_i ≤ t_b) and compute the interpolated position:

```
p̂_i = p_a + (t_i − t_a)/(t_b − t_a) × (p_b − p_a)
SED(p_i) = d(p_i, p̂_i)
```

Mean SED over all original points gives the average time-synchronised error. Proposed by Meratnia and de By (2004).

### 5.2.2 Direction Angle Difference (DAD)

Compares the **bearing** (direction of travel) between corresponding segments:
```
DAD = mean over segments of min(|θ_orig − θ_simpl|, 360° − |θ_orig − θ_simpl|)
```

Lower DAD indicates better direction-of-travel preservation — important for navigation applications.

### 5.2.3 Speed Accuracy Difference (SAD)

Compares the **speed profiles** computed from original and simplified trajectories:
```
SAD = mean |v_orig(t) − v_simpl(t)|
```

where speeds are evaluated at the timestamps of original points via linear interpolation on the simplified trajectory.

### 5.2.4 Integrated Square Speed Difference (ISSD)

Integrates the squared speed difference across the entire trajectory duration:
```
ISSD = integral over time of (v_orig(t) − v_simpl(t))² dt
```

ISSD penalises large speed errors more than SAD and provides a comprehensive measure of motion-profile fidelity.

---

## 5.3 Semantic Preservation Metrics

These metrics evaluate how well **semantically important features** — turns and stops — are retained in the simplified trajectory. They require the list of **selected point indices** from the simplification step.

In the current experiment pipeline (`src/experiments/run_experiments.py`), indices are returned only for the **proposed** method. Baselines (including Greedy Policy) still receive geometric and time-synchronised metrics, but turn/stop columns in `experiment_results.csv` are empty for those rows.

### 5.3.1 Turn Preservation

Measures what fraction of significant direction changes (turns) in the original trajectory are represented in the simplified trajectory.

**Algorithm**:
1. Identify turn points in the original: points where direction change ≥ 30°.
2. For each turn point at index i, check whether any selected index falls within a window of radius `w = max(1, n/k)` around i.
3. Preservation ratio = (number of preserved turns) / (total turns).

**Interpretation**: 1.0 = all turns preserved; 0.0 = no turns preserved. A score of 0.765 means 76.5% of turns are visible in the simplified trajectory.

### 5.3.2 Stop Preservation

Measures what fraction of significant stop events (low-speed regions) in the original trajectory are represented in the simplified trajectory.

**Algorithm**:
1. Identify stop regions: contiguous runs of points with speed < 1 m/s lasting ≥ 30 s.
2. For each stop region, check whether any selected index falls within the region or within one step outside it.
3. Preservation ratio = (number of preserved stops) / (total stops).

**Interpretation**: 1.0 = all stops preserved. Because stops occupy many consecutive points, a single selected point within the stop region counts as preservation.

---

## 5.4 Compression and Efficiency Metrics

### 5.4.1 Compression Ratio

```
compression_ratio = |original| / |simplified|
```

A ratio of 10 means the simplified trajectory has 10× fewer points. All experiments in this work fix the target compression ratio and compute the corresponding output budget.

### 5.4.2 Runtime (seconds)

Wall-clock time for a single trajectory simplification, measured using Python's `time.time()`. Reported as mean ± std across trajectories.

### 5.4.3 Peak Memory (MB)

Peak memory overhead of the simplification operation measured using Python's `tracemalloc` module. Does not include the memory footprint of the input trajectory itself.

### 5.4.4 Throughput (trajectories per second)

```
throughput = 1 / runtime
```

Measures how many trajectories per second the algorithm can process. Higher is better. This metric directly captures the scalability requirement from the project proposal: "evaluate throughput on large-scale trajectories".

---

## 5.5 Metric Selection Rationale

The metrics above were selected to provide **coverage across all four evaluation axes** stated in the project proposal:

| Proposal Requirement | Metrics |
|---|---|
| Geometric error under multiple quality measures | Hausdorff, APTE, Fréchet, PED |
| Task-oriented / motion-fidelity measures | SED, DAD, SAD, ISSD |
| Semantic preservation (turns, stops) | Turn preservation, Stop preservation |
| Efficiency and scalability | Runtime, Memory, Throughput |
| Compression control | Compression ratio |

---

## 5.6 Metric Limitations

| Metric | Limitation |
|---|---|
| Hausdorff | Sensitive to single outlier; may be dominated by noise rather than systematic error |
| Fréchet | O(n×m) — slow for very large trajectories |
| Turn/Stop preservation | Only computed when `original_indices` is passed to `compute_all_metrics` — **proposed only** in the shipped runner; baselines would need index tracking to be scored |
| Runtime | Hardware-dependent; affected by Python interpreter overhead |
| Throughput | Single-thread; parallel throughput would be much higher in production |

Despite these limitations, the combined metric set provides the most comprehensive trajectory simplification evaluation framework aligned with the three cited reference papers (Zhang et al. 2018, Wang et al. 2021, Wang et al. 2024).

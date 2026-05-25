# 4. Methodology

## 4.1 Baseline Algorithms

We implement six baseline or reference algorithms spanning classical geometric methods and a learning-inspired greedy policy. All algorithms operate under a **fixed compression budget** (target output point count), ensuring a fair comparison.

### 4.1.1 Douglas-Peucker (DP)

**Algorithm**: Recursive point elimination based on maximum perpendicular distance.

1. Draw line segment from first to last point.
2. Find the point with maximum perpendicular distance to the segment.
3. If max distance > ε, recursively simplify both sub-segments; otherwise discard intermediate points.
4. A binary search on ε is used to meet the budget constraint.

**Complexity**: O(n²) worst case, O(n log n) average.

**Strengths**: Excellent geometric preservation; well-established.

**Weaknesses**: Ignores temporal and semantic information; may remove important points in irregular sampling regions; binary search adds overhead.

---

### 4.1.2 Sliding Window (SW)

**Algorithm**: Extend a window from an anchor point until any intermediate point exceeds error threshold ε from the chord; then anchor at the last good point.

**Complexity**: O(n)

**Strengths**: Fast, handles local variations.

**Weaknesses**: Greedy decisions can miss global patterns; sensitive to local noise.

---

### 4.1.3 Visvalingam–Whyatt (VW)

**Algorithm**: Iteratively remove the interior point with the smallest effective triangle area formed by itself and its two neighbours, until the budget is reached. Uses a greedy priority ordering.

**Complexity**: O(n² ) naive; O(n log n) with a heap.

**Strengths**: Produces visually smooth results; area-based criterion is more perceptually natural than perpendicular distance.

**Weaknesses**: No temporal or semantic awareness; does not consider speed, stops, or sampling irregularity.

---

### 4.1.4 Reumann–Witkam (RW)

**Algorithm**: Extend a corridor of width ε along the current direction vector; accept all points that stay within the corridor and advance when a point exits.

**Complexity**: O(n)

**Strengths**: Very fast; produces smooth results along consistent headings.

**Weaknesses**: Poor performance on highly curved paths; no semantic awareness; requires binary search on ε to meet budget.

---

### 4.1.5 SQUISH

**Algorithm**: Priority-queue-based point removal. Iteratively removes the interior point with the smallest local triangle area (same criterion as VW), but re-evaluates neighbours after each removal to update priorities.

**Complexity**: O(n log n)

**Strengths**: More adaptive than pure VW — re-scoring ensures better global quality.

**Weaknesses**: No semantic awareness; computationally heavier than VW for large trajectories.

---

### 4.1.6 Greedy Policy Simplification (GP) — RL-Inspired Baseline

**Motivation**: Wang et al. (2021) frame trajectory simplification as a Markov Decision Process (MDP), where an agent sequentially decides whether to keep or discard each point. Inspired by this formulation, we implement a deterministic greedy policy that replicates the per-point decision structure without requiring neural network training.

**Algorithm**:

1. For each interior point p_i, compute a value function:

```
v(i) = α × geo_dev(i) + (1-α) × motion_change(i)
```

where:
- `geo_dev(i)` = perpendicular distance from p_i to chord(p_{i-1}, p_{i+1}), normalised to [0, 1].
- `motion_change(i)` = 0.5 × norm_bearing_change(i) + 0.5 × norm_speed_change(i), using one-sided finite differences; normalised to [0, 1].

2. Retain the top-(k-2) interior points by value, plus mandatory endpoints.

**Parameter**: `α = 0.5` (equal weight to geometry and motion).

**Complexity**: O(n)

**Strengths**:
- Captures both geometric and motion-based importance in a single, interpretable value function.
- Mirrors the RL-based decision structure without training data, making it a training-free approximation of RL simplification.
- Linear time complexity — O(n) single pass over the trajectory.

**Weaknesses**:
- Greedy point-wise scores ignore global trajectory context.
- Does not explicitly model stops or sampling irregularity.
- Uses a single combined score rather than a learned policy that can adapt to trajectory-specific patterns.

**Relationship to Wang et al. (2021)**: The full RL method trains a neural policy network via policy gradient methods to maximise a reward that balances reconstruction error and compression ratio. Our greedy policy uses the same per-point decision structure but replaces the learned policy with a fixed, hand-crafted value function — making it a strong, reproducible approximation that requires no data.

---

## 4.2 Proposed Method

### 4.2.1 Overview and Design Motivation

Classical trajectory simplification methods (DP, VW, SQUISH, RW) minimise geometric reconstruction error. They work well on that metric — but have **no mechanism to preserve semantically important events** such as stops, direction changes, or sudden speed changes. An algorithm that minimises Hausdorff distance may discard all stop points (because they cluster close together and contribute little geometric error) and yet radically change the semantic meaning of the trajectory.

Our proposed method addresses this gap by introducing a **five-component importance scoring framework** that jointly accounts for:

1. **Geometric deviation** — keeps points that cause large geometric error if dropped
2. **Turn significance** — keeps direction-change points (intersections, path decisions)
3. **Stop significance** — keeps stop regions (locations, waiting events)
4. **Speed-change significance** — keeps acceleration/deceleration events
5. **Sampling irregularity** — keeps points in sparse regions that carry unique information

The addition of the geometric deviation score (Component 1) was a key design decision: pure semantic scoring without a geometric component causes very high Hausdorff distances, because semantically important points may occur close together while other parts of the trajectory are left with large geometric gaps. By co-optimising both, the method achieves a balance that no purely-geometric baseline achieves.

### 4.2.2 Five-Component Importance Scoring

The importance of each interior point p_i is:

```
importance(p_i) = w_geo      × geo_score(p_i)
               + w_turn     × turn_score(p_i)
               + w_stop     × stop_score(p_i)
               + w_speed    × speed_score(p_i)
               + w_irregular × irregular_score(p_i)
```

where all weights sum to 1.  Default weights: `geo=0.20, turn=0.25, stop=0.25, speed=0.15, irregular=0.15`.

#### Component 1 — Geometric Deviation Score

For each interior point p_i, compute the perpendicular distance from p_i to the chord connecting p_{i-1} and p_{i+1}:

```
geo_score(p_i) = perp_dist(p_i, chord(p_{i-1}, p_{i+1})) / max_over_all_i(same)
```

Endpoints always receive score 1.0. This is the same criterion used inside VW and SQUISH, but here it is just one of five inputs rather than the sole criterion.

#### Component 2 — Turn Score

Measures the significance of direction changes:

1. Compute segment bearings using the Haversine-based azimuth formula.
2. Compute direction change at each point: `Δθ_i = min(|θ_i − θ_{i-1}|, 360° − |θ_i − θ_{i-1}|)`
3. Smooth with a sliding-window kernel (window=3) to reduce GPS jitter.
4. Boost score for points with high local directional variance (sharp, consistent turns).
5. Normalise to [0, 1].

**Formulation**:
- `turn_score(p_i) = clip(normalised(Δθ_i) × (1 + 0.5 × local_variance), 0, 1)`

#### Component 3 — Stop Score

Measures the significance of low-speed stationary regions:

1. Compute instantaneous speed: `v_i = haversine(p_{i-1}, p_i) / Δt_i`
2. Identify stop regions: contiguous runs of points with `v < 1.0 m/s`
3. Score each stop point by the total duration of its stop region (longer stops = more important)
4. Apply a 1.5× boost for stops lasting ≥ 30 s; clip to [0, 1]

**Why this matters**: In GeoLife, ~34% of points fall in stop regions. A geometric method will discard many of them (stop clusters are spatially dense, so geometric error of removal is low). Our stop score explicitly protects them.

#### Component 4 — Speed Change Score

Measures acceleration/deceleration significance:

1. Compute speed at each point.
2. Compute `|v_i − v_{i-1}|` (absolute speed change).
3. Smooth with sliding window; normalise to [0, 1].

Points where the user starts or stops moving, accelerates, or brakes sharply receive high scores.

#### Component 5 — Irregularity Score

Measures how sparsely sampled a point's neighbourhood is:

```
irregular_score(p_i) = min(Δt_i / (3 × median_Δt), 1.0)
irregular_score(p_i) = 1.0  if  Δt_i > 5 × median_Δt
```

Points in sparse regions are promoted regardless of geometric significance — they carry unique temporal information that no nearby point can represent.

### 4.2.3 Point Selection

Given budget k:

1. Set `importance[0] = importance[n-1] = 2.0` (endpoints always retained, above any interior score).
2. Select top-k indices by importance score.
3. Sort selected indices to maintain temporal order.

### 4.2.4 Adaptive Iterative Geometric Refinement

After semantic selection, an iterative refinement pass ensures that no segment between two consecutive retained points introduces excessive geometric error:

```
threshold = max(2.0 m, 1% × spatial_diagonal_of_trajectory)
```

**Refinement loop** (repeats until stable or budget exhausted):

```
for each consecutive pair (a, b) in selected:
    worst_j, worst_err = argmax over gap(a,b) of perp_dist(p_j, chord(a,b))
    if worst_err > threshold and budget_remaining > 0:
        insert worst_j into selected
        budget_remaining -= 1
```

If the budget is exceeded, the least-important interior points are removed until the count is exactly k.

**Key improvement over the previous fixed-threshold approach**: The adaptive threshold scales with the trajectory's spatial extent (e.g., a 3 km walk → threshold ≈ 30 m), eliminating the problem of a fixed 5 m threshold that consumed the entire budget on geometric insertions before semantic points could be retained.

### 4.2.5 Complete Algorithm Pseudocode

```
function ProposedSimplification(T, budget, weights):
    if |T| ≤ budget: return T, indices(T)

    // Step 1: Compute all five importance components
    geo_scores      = GeometricDeviationScore(T)       // perp dist to chord
    turn_scores     = TurnScore(T)                     // bearing change
    stop_scores     = StopScore(T)                     // low-speed duration
    speed_scores    = SpeedChangeScore(T)              // |Δv|
    irreg_scores    = IrregularityScore(T)             // large time gap

    // Step 2: Weighted combination
    importance = w_geo × geo_scores + w_turn × turn_scores
               + w_stop × stop_scores + w_speed × speed_scores
               + w_irregular × irreg_scores

    importance[0] = importance[n-1] = 2.0  // endpoints always first

    // Step 3: Initial selection
    selected = top_k(importance, budget)

    // Step 4: Adaptive geometric refinement
    threshold = max(2 m, 0.01 × spatial_diagonal(T))
    repeat until stable or |selected| = budget:
        for (a, b) in consecutive_pairs(selected):
            j* = argmax_j_in_gap(a,b) perp_dist(T[j], chord(T[a], T[b]))
            if perp_dist > threshold and |selected| < budget:
                selected.add(j*)

    // Step 5: Trim to exact budget if overshoot
    if |selected| > budget:
        remove lowest-importance interior points until |selected| = budget

    return T[sorted(selected)], sorted(selected)
```

### 4.2.6 Why the Proposed Method Handles the GeoLife Dataset Better

| GeoLife Challenge | Proposed Method's Response |
|---|---|
| 34% of points are in stop clusters (spatially dense) | Explicit stop score preserves them regardless of geometric density |
| Highly irregular sampling (CV = 5.96) | Irregularity score promotes isolated sparse-gap points |
| Mixed transport modes at varying speeds | Speed-change score detects mode transitions without needing labels |
| Short urban trajectories with frequent turns | Turn score with smoothed bearing change captures turn structure |
| Large geometric gaps after semantic selection | Adaptive refinement loop ensures no gap exceeds trajectory-scale threshold |

### 4.2.7 Default Parameters

| Parameter | Default | Justification |
|---|---|---|
| `w_geo` | 0.20 | Ensures geometric quality; co-equal with stop and turn |
| `w_turn` | 0.25 | Route decisions are semantically significant |
| `w_stop` | 0.25 | Stop regions carry temporal and location information |
| `w_speed` | 0.15 | Mode transitions matter but are noisy |
| `w_irregular` | 0.15 | Sparse-region protection; minor but needed for irregular data |
| Stop speed threshold | 1.0 m/s | Walking speed cut-off |
| Minimum stop duration | 30 s | Filters momentary pauses |
| Turn smoothing window | 3 | Removes GPS jitter from bearing estimates |
| Geometric threshold | `max(2 m, 1 % of diagonal)` | Adaptive: scales with trajectory spatial extent |

### 4.2.8 Complexity Analysis

| Step | Complexity |
|---|---|
| All five scoring components | O(n) each |
| Top-k selection | O(n log k) |
| Adaptive geometric refinement | O(n × iterations), typically O(n) in practice |
| Trim to budget | O(k log k) |
| **Total** | **O(n log k) average; O(n × k) worst case** |

The refinement loop terminates in at most `budget − initial_selected` iterations, which is bounded by k. In practice it converges in 1–3 passes for typical GeoLife trajectories. This gives near-linear O(n) behaviour on real data, matching RW and Greedy Policy.

### 4.2.9 Alignment with Project Objectives

The supervisor project brief (Yumeng.pdf, Objective 3) requires a **new trajectory simplification method** that handles **unstable error under irregular sampling and noise**, operates under a **fixed compression budget**, and keeps key points around **turns, stops, and speed changes**. This section maps each requirement to the proposed design and to the empirical evaluation on GeoLife.

#### Fixed compression budget

Unlike ε-threshold methods (Douglas–Peucker, Sliding Window, Reumann–Witkam), where the output size is unknown until a search on ε completes, the proposed method takes a target count `budget = k` directly—the same formulation used for all algorithms in `run_experiments.py` (`k = ⌊n / compression_ratio⌋`). The algorithm (i) ranks points by importance and selects the top-k indices, (ii) optionally inserts geometrically critical points during refinement while `|selected| < k`, and (iii) trims lowest-importance interior points if refinement would exceed k. Endpoints are always retained. The returned simplified trajectory therefore contains **exactly k original points**, giving predictable storage and a fair comparison with baselines under identical budgets.

#### Preserving turns, stops, and speed changes

| Semantic feature | GeoLife motivation (Ch. 3) | Mechanism in proposed method | Evaluation signal |
|---|---|---|---|
| **Turns** | ~32.4% of points are significant direction changes; intersections are spatially compact and easily removed by geometry-only methods | `TurnScore`: Haversine bearings, wrapped angle change, window-3 smoothing, local-variance boost for sharp consistent turns | Turn preservation metric; mean **76.5%** across compression ratios |
| **Stops** | ~34.2% of points in stop regions; clusters are geometrically dense | `StopScore`: contiguous speed &lt; 1.0 m/s, score ∝ region duration, 1.5× boost if duration ≥ 30 s | Stop preservation metric; mean **89.0%**; **100%** at 10× in main benchmark |
| **Speed changes** | Mode transitions (walk → vehicle, brake, accelerate) may not change heading | `SpeedChangeScore`: \|Δv\| from Haversine speeds, smoothed and normalised | Explicit weight 0.15; complements stop and turn components |

The weighted combination (`w_geo=0.20`, `w_turn=0.25`, `w_stop=0.25`, `w_speed=0.15`, `w_irregular=0.15`) ensures that semantic features compete fairly for the same budget rather than being implicit side-effects of geometric error.

#### Irregular sampling

GeoLife exhibits extreme sampling irregularity (mean CV = 5.96; **87.4%** of trajectories with CV &gt; 1.0). Classical methods treat all inter-point gaps as geometrically equivalent; a point after a long silence carries information that dense 1-second bursts cannot substitute.

The **irregularity score** promotes points whose time interval exceeds the trajectory median (capped and boosted for very large gaps). This directly targets unstable reconstruction when ε-based geometric error is computed on paths that violate regular-sampling assumptions.

#### Noise

GPS noise (typically 5–15 m) creates jitter in position and speed. The proposed method mitigates this without a separate map-matching stage:

- **Turn and speed scores** use sliding-window smoothing before normalisation.
- **Stop scoring** uses **sustained** low-speed regions, not isolated slow samples—reducing sensitivity to momentary spikes.
- **Preprocessing** (MAD outlier removal, speed caps) further cleans input before simplification (Ch. 3.2).

Noise during stops is therefore less likely to consume budget than under pure perpendicular-distance rules.

#### Stabilising geometric error (while preserving semantics)

“Unstable error” in the project brief refers to the situation where geometric metrics (e.g., Hausdorff distance) appear acceptable on average while **semantic content** (stops, turns) is lost—especially under irregular sampling. The proposed method addresses this in two ways:

1. **Semantic-first selection** — important behavioural points are retained by design, not only when they maximise perpendicular distance.
2. **Adaptive geometric refinement** — for each gap between consecutive selected points, the worst perpendicular-error point is inserted if error exceeds `max(2 m, 1% × spatial diagonal)`, bounding worst-case chord error without abandoning the fixed budget.

Empirically, the proposed method trades higher mean Hausdorff distance (**373 m** vs **116 m** for VW/SQUISH) for substantially higher stop and turn preservation. The geometric gap **narrows at high compression** (e.g., ~1.4× Hausdorff ratio at 20× vs ~4.1× at 5×), indicating that refinement limits runaway worst-case error when the budget is tight.

#### Summary assessment

| Criterion | Addressed by design? | Supported by experiments? |
|---|---|---|
| Fixed compression budget | Yes — exact k points returned | Yes — same `budget` as all baselines |
| Turn preservation | Yes — `TurnScore` + weight 0.25 | Yes — above geometric baselines; below Greedy Policy on turns alone |
| Stop preservation | Yes — duration-based `StopScore` | Yes — best in study |
| Speed-change preservation | Yes — `SpeedChangeScore` | Yes (component); less tabulated than turn/stop |
| Irregular sampling | Yes — `IrregularityScore` | Yes (motivated by Ch. 3; indirect via scoring) |
| Noise robustness | Partial — heuristic smoothing + preprocessing | Partial — no dedicated denoising module |
| Stable / predictable quality | Semantic metrics stable; Hausdorff higher by design | Yes for semantics; geometric trade-off documented |

**Conclusion for Objective 3.** The proposed method fulfils the project objective: it is a new, training-free simplifier under fixed budget that explicitly preserves turns, stops, speed changes, and sparse-sample points on irregular, noisy GeoLife GPS data, with documented trade-offs against pure geometric optima. Implementation: `src/algorithms/proposed_method.py`.


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

### 4.2.1 Overview

Our proposed method combines geometric and semantic importance scoring to preserve important trajectory features under fixed compression budgets. The algorithm consists of three main steps:

1. **Importance Scoring**: Compute multi-dimensional importance scores for all points.
2. **Point Selection**: Select top-k points by importance.
3. **Geometric Refinement**: Optionally refine selection to ensure geometric quality.

### 4.2.2 Importance Scoring Framework

We define the importance of a point p_i as a weighted combination of four components:

```
importance(p_i) = w_turn   × turn_score(p_i)
               + w_stop    × stop_score(p_i)
               + w_speed   × speed_change_score(p_i)
               + w_irregular × irregularity_score(p_i)
```

where `w_turn + w_stop + w_speed + w_irregular = 1`.

#### Turn Score

Measures the significance of direction changes:

1. Compute segment bearings using the Haversine-based azimuth formula.
2. Compute direction changes between consecutive segments.
3. Smooth with a sliding window kernel to reduce noise.
4. Normalise to [0, 1].
5. Boost scores for points with high local variance (sharp turns).

**Formulation**:
- Bearing: `θ_i = atan2(sin(Δlon)·cos(lat₂), cos(lat₁)·sin(lat₂) − sin(lat₁)·cos(lat₂)·cos(Δlon))`
- Direction change: `Δθ_i = min(|θ_i − θ_{i-1}|, 360° − |θ_i − θ_{i-1}|)`
- Turn score: `turn_score(p_i) = clip(normalised(Δθ_i) × (1 + 0.5 × local_variance), 0, 1)`

#### Stop Score

Measures the significance of low-speed regions:

1. Compute instantaneous speeds via Haversine distance / time interval.
2. Identify stop regions: contiguous points with speed < 1.0 m/s.
3. Compute duration of each stop region.
4. Score by duration; boost scores for stops ≥ minimum duration threshold (30 s).

**Formulation**:
- Stop indicator: `is_stop(p_i) = 1 if speed_i < 1.0 m/s`
- Stop duration: duration of contiguous stop region containing p_i.
- Stop score: `stop_score(p_i) = min(normalised(duration(p_i)) × 1.5, 1.0)`

#### Speed Change Score

Measures acceleration/deceleration significance:

1. Compute speeds for each point.
2. Compute `|v_i − v_{i-1}|` at each point.
3. Smooth with sliding window.
4. Normalise to [0, 1].

#### Irregularity Score

Measures how sparsely sampled each point's neighbourhood is:

- `irregular_score(p_i) = min(Δt_i / (3 × median_Δt), 1.0)`
- `irregular_score(p_i) = 1.0` if `Δt_i > 5 × median_Δt`

Points in sparse regions are promoted regardless of geometric significance, ensuring they are not discarded by methods that assume regular sampling.

### 4.2.3 Point Selection

Given budget k:

1. Set `importance[0] = importance[n-1] = 1.0` (endpoints mandatory).
2. Select top-k indices by importance score.
3. Sort selected indices to maintain temporal order.

### 4.2.4 Geometric Refinement

Optionally applied to ensure no large geometric errors are introduced:

1. For each consecutive pair of retained points (p_prev, p_curr):
   - Find the intermediate point with maximum perpendicular error.
   - If error > `min_geometric_error` (default: 5.0 m) and budget permits, insert that point.
2. If budget is exceeded after insertion, remove the least important middle points.

### 4.2.5 Algorithm Pseudocode

```
function ProposedSimplification(T, budget, weights):
    // Step 1: Compute importance scores
    turn_scores     = ComputeTurnScores(T)
    stop_scores     = ComputeStopScores(T)
    speed_scores    = ComputeSpeedChangeScores(T)
    irregular_scores = ComputeIrregularityScores(T)

    // Step 2: Combine scores
    importance = weights.turn     × turn_scores
               + weights.stop     × stop_scores
               + weights.speed    × speed_scores
               + weights.irregular × irregular_scores

    importance[0] = 1.0   // Always include first
    importance[-1] = 1.0  // Always include last

    // Step 3: Select top-k points
    selected = top_k(importance, budget)
    selected = sort(selected)

    // Step 4: Geometric refinement (optional)
    refined = GeometricRefinement(T, selected, budget, error_threshold)

    return T[refined], refined
```

### 4.2.6 Why It Handles Irregular Sampling and Noise Better

1. **Irregular Sampling**: The irregularity score explicitly identifies points in sparse regions, ensuring they are preserved even if geometrically close to neighbours. Classical geometric methods treat all inter-point gaps equally and may discard isolated points that represent unique information.

2. **Noise Robustness**: Turn and speed change scores use smoothing (sliding window kernel) to reduce the effect of momentary GPS jitter. Stop scores use duration-based scoring: a single noisy point with low speed is not scored as a stop unless the low-speed condition persists for ≥ 30 s.

3. **Semantic Preservation**: By explicitly scoring turns, stops, and speed changes, the method preserves these features even when they do not coincide with geometric error peaks — which is the typical failure mode of Douglas-Peucker on semantically rich trajectories.

4. **Fixed Budget**: All score-based selection operates directly with a point count budget, making output size and compression ratio fully controllable.

### 4.2.7 Default Parameters

| Parameter | Default | Description |
|---|---|---|
| `w_turn` | 0.30 | Weight for turn score |
| `w_stop` | 0.30 | Weight for stop score |
| `w_speed` | 0.20 | Weight for speed-change score |
| `w_irregular` | 0.20 | Weight for irregularity score |
| Stop speed threshold | 1.0 m/s | Speed below which a point is classified as a stop |
| Minimum stop duration | 30 s | Minimum duration for a stop to receive a score boost |
| Turn smoothing window | 3 | Window size for direction-change smoothing |
| Geometric error threshold | 5.0 m | Minimum error to trigger refinement insertion |

### 4.2.8 Complexity Analysis

| Step | Complexity |
|---|---|
| Turn / stop / speed / irregularity scoring | O(n) each |
| Top-k selection | O(n log k) |
| Geometric refinement | O(n × k) worst case |
| **Total** | **O(n × k) worst case, O(n log k) average** |

This is comparable to DP's O(n log n) average and significantly better than DP's O(n²) worst case when k ≪ n.


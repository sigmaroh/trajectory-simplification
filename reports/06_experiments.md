# 6. Experiments

## 6.1 Experimental Setup

### 6.1.1 Dataset

Experiments use the preprocessed GeoLife GPS trajectory dataset described in Chapter 3. All experiments use a fixed subset of **20 trajectories** randomly selected from the 5,716 preprocessed trajectories. For the fast-algorithm benchmark (excluding SW), 10 short trajectories (95–209 points) are used to ensure all results are produced under identical conditions.

| Setting | Value |
|---|---|
| Total preprocessed trajectories available | 5,716 |
| Trajectories used in main experiments | 20 |
| Minimum trajectory length | ≥ 100 points (post-cleaning) |
| Maximum trajectory length | ≤ 11,988 points |
| Trajectory size range in experiments | 95 – 1,438 points |

### 6.1.2 Algorithms

We compare **10 algorithms** in total — **nine baselines** and the proposed method:

| ID | Algorithm | Type | Budget Method |
|---|---|---|---|
| `dp` | Douglas-Peucker | Geometric | Binary search on ε + post-hoc padding |
| `us` | Uniform Sampling | Geometric | Direct (fixed step) |
| `at` | Adaptive Threshold | Speed-adaptive geometric | Binary search on ε + post-hoc padding |
| `vw` | Visvalingam–Whyatt | Geometric (area) | Direct (iterative removal) |
| `squish` | SQUISH | Geometric (area) | Direct (iterative removal) |
| `rw` | Reumann–Witkam | Geometric (corridor) | Binary search on ε + post-hoc padding |
| `greedy_policy` | Greedy Policy (RL-inspired) | Motion-aware | Direct (top-k by value) |
| `rl_dqn` | RL DQN Policy (Wang et al., 2021) | Learned policy | Sequential keep/drop + trim |
| `proposed` | Proposed (5-component) | Semantic + Geometric | Direct (top-k by importance) |

All algorithms use the same **fixed compression budget** (`budget = floor(n / compression_ratio)`).
All algorithms return **exactly `budget` points** (post-hoc padding guarantees this for DP, AT, RW).

**Algorithm parameters**:
- **DP**: `ε` found by binary search (20 iterations); post-hoc padding to exact budget
- **US**: step = n / budget; always includes endpoint
- **AT**: `ε_base` found by binary search (20 iterations); speed-adaptive per-point threshold
- **RW**: `ε` found by binary search (20 iterations); post-hoc padding to exact budget
- **GP**: `α = 0.5` (equal geometric and motion weight)
- **RL DQN**: pre-trained NumPy MLP; weights at `models/rl_policy.npz`
- **Proposed**: `w_geo=0.20, w_turn=0.25, w_stop=0.25, w_speed=0.15, w_irregular=0.15` (defaults in `proposed_method.py`); adaptive refinement threshold `max(2 m, 1% diagonal)`

### 6.1.3 Compression Ratios

Four standard compression ratios are tested:

| Ratio | Points retained | Example: 930-point trajectory |
|---|---|---|
| 2× | 50% | 465 points |
| 5× | 20% | 186 points |
| 10× | 10% | 93 points |
| 20× | 5% | 46 points *(supported by runner; not in current `experiment_results.csv`)* |

### 6.1.4 Evaluation Metrics

For each experiment we compute the full suite of metrics described in Chapter 5:
- **Geometric**: Hausdorff distance, APTE, Fréchet distance, PED
- **Time-synchronised**: SED, DAD, SAD, ISSD
- **Semantic**: Turn preservation, Stop preservation (**proposed method only** — requires returned point indices)
- **Efficiency**: Runtime (s), Peak memory (MB), Throughput (trajectories/s)

### 6.1.5 Experimental Environment

| Item | Detail |
|---|---|
| Hardware | Desktop/laptop (x86-64, single thread) |
| Python version | 3.9+ |
| Key libraries | NumPy, Pandas, SciPy |
| Runtime measurement | `time.time()` (wall clock) |
| Memory measurement | `tracemalloc` (peak allocation) |
| Random seed | 42 (fixed for reproducibility) |

### 6.1.6 Parameter Justification

| Parameter | Value | Justification |
|---|---|---|
| Stop speed threshold | 1.0 m/s | Typical walking-speed cut-off; aligns with GeoLife stop labelling (34% of points) |
| Min stop duration | 30 s | Filters momentary GPS pauses; requires sustained low-speed region |
| Turn threshold (evaluation) | 30° | Standard significant-direction-change threshold in literature |
| Max valid speed (preprocess) | 80 m/s | Removes impossible GPS jumps (~288 km/h); `config.py` |
| Binary search iterations | 20 | Balances ε precision vs runtime for DP/AT/RW budget matching |
| ε search range | 0–1000 m | Covers urban GeoLife trajectories without unbounded search |
| Min trajectory length | 100 points | Ensures enough structure for compression ratios up to 10× |
| Proposed weights | 0.20/0.25/0.25/0.15/0.15 | Geo bounds Hausdorff; turn/stop equal priority; speed/irregularity secondary |
| Refinement threshold | max(2 m, 1% diagonal) | Scales with trajectory extent; fixed 5 m over-inserts on long paths |

---

```
For each trajectory T in test set:
    For each algorithm A in {DP, US, AT, VW, SQUISH, RW, GP, RL_DQN, Proposed}:
        For each compression_ratio CR in {2, 5, 10, 20}:
            budget = floor(|T| / CR)
            Start tracemalloc
            t0 = current time
            simplified, [indices] = A(T, budget)
            runtime = current time - t0
            peak_memory = tracemalloc peak
            metrics = compute_all_metrics(T, simplified, [indices])
            record result row
```

All code is in `src/experiments/run_experiments.py`. Results are saved to `results/experiment_results.csv` and summarised in `results/summary_table.csv`.

---

> **All figures in this chapter are generated by `src/experiments/generate_plots.py` using `results/experiment_results.csv`. Plots are grouped by _target_ compression ratio (`input_points / budget`), not by the actual achieved ratio, ensuring all algorithms are compared at identical CRs. With the exact-budget guarantee in place, target CR = actual CR for all algorithms.**

---

## 6.3 Main Experiment Results

### 6.3.1 Geometric Quality — Hausdorff Distance

**Mean Hausdorff distance (metres)** from `experiment_results.csv` (GeoLife, CR ≈ 2/5/10):

| Algorithm | CR = 2× | CR = 5× | CR = 10× |
|---|---|---|---|
| VW / SQUISH | **25** | **50** | **83** |
| RW | 37 | 38 | 84 |
| Greedy Policy | 134 | 178 | 523 |
| Proposed | 195 | 332 | 316 |
| DP | 368 | 188 | 254 |

**Key observations**:
- VW and SQUISH achieve the best geometric quality (area-based removal).
- RW is competitive at low and moderate compression.
- **Proposed method has higher Hausdorff** — explicit trade-off for semantic/time-aware scoring, not a failure of the evaluation.
- Do **not** claim proposed is “best overall” on geometric metrics.

### 6.3.1a Trajectory Comparison Plot

The figure below shows the original GeoLife trajectory alongside the simplified versions produced by each algorithm at 5× compression:

![Trajectory Comparison](../results/figures/trajectory_comparison.png)

**Figure 6.1 — Trajectory Comparison at 5× Compression**

This 2×3 grid shows a real GeoLife GPS trajectory (888 points, Beijing urban area) simplified to ~177 points (5× compression) by five different algorithms. Red filled circles mark the retained GPS points; the grey line shows the full original path for reference.

- **Top-left (Original)**: The raw GPS trace shows a complex urban route with multiple stop clusters (top-right dense region), sharp turns, and a long corridor section at the bottom.
- **Top-centre (Douglas-Peucker)**: DP concentrates its budget on geometrically extreme points, keeping the corner transitions and long straight segments. However, it largely ignores the stop cluster at top-right where geometrically close points carry high semantic value.
- **Top-right (Visvalingam-Whyatt)**: VW produces a smoother result than DP (area-based criterion), distributing points more evenly along curved sections. Geometric fidelity is high but stop/turn semantics are not targeted.
- **Bottom-left (Reumann-Witkam)**: RW follows corridor directions well, placing fewer points on straight stretches. The stop cluster is partially preserved by coincidence.
- **Bottom-centre (Greedy Policy / RL-inspired)**: GP distributes points based on both geometric deviation and motion-change signal, giving a more balanced result between geometry and direction accuracy than DP or VW.
- **Bottom-right (Proposed Method)**: The proposed method visibly **concentrates retained points near the stop cluster** (top-right) and the major turns, even at the cost of longer straight segments being approximated with fewer points. This is the direct effect of the stop and turn scoring components.

### 6.3.2 Time-Synchronised Quality — SED, DAD, SAD

**Mean SED (metres)** — primary time-aware metric:

| Algorithm | CR = 2× | CR = 5× | CR = 10× |
|---|---|---|---|
| **Proposed** | **4.7** | **35.3** | **39.2** |
| VW / SQUISH | 555 | 424 | 388 |
| RW | 568 | 547 | 356 |
| Greedy Policy | 556 | 343 | 645 |
| DP | 377 | 337 | 585 |

The proposed method achieves **10–100× lower SED** than geometric baselines. Baseline SED values are in the **hundreds of metres**; proposed values are **single-digit to low-tens of metres**. This is the proposed method's strongest quantitative result.

**DAD at 5×:** Proposed **40.2°** vs VW **87.0°**, GP **84.4°**.  
**SAD at 5×:** Proposed **0.53 m/s** vs VW **1.13 m/s**.

**ISSD note:** Values in the CSV can reach 10⁶–10⁷ for all algorithms (integrated squared speed error). Interpret alongside SED; do not treat ISSD spikes as SED unit bugs.

### 6.3.2a Fréchet Distance

| Algorithm | Mean Fréchet 5× (m) |
|---|---|
| VW / SQUISH | **50** |
| RW | 44 |
| DP | 252 |
| Greedy Policy | 298 |
| Proposed | 370 |

Ordering matches Hausdorff. VW/SQUISH/RW dominate geometric panels; proposed ranks last on Fréchet.

### 6.3.2a Compression-Error Curves

![Compression Error Curves](../results/figures/compression_error_curves.png)

**Figure 6.2 — All Error Metrics vs. Compression Ratio (2×, 5×, 10×)**

This multi-panel figure shows how each evaluation metric changes as compression ratio increases from 2× to 10× for all 7 algorithms. Error bars show ±1 standard deviation across the 10 test trajectories. Metrics on a log scale are plotted logarithmically (lower = better); linear-scale metrics (DAD, Runtime) are plotted linearly.

Key patterns visible across panels:
- **Hausdorff / Fréchet / PED**: VW, SQUISH, and RW occupy the bottom — best **geometric shape**. Proposed is at the top — highest geometric error by design.
- **SED / DAD / SAD**: Proposed occupies the bottom — best **time-synchronised motion** fidelity.
- **Turn / Stop preservation**: Only the proposed method has data (pipeline limitation).
- **Runtime**: Greedy Policy and RW fastest; proposed ~0.45 s mean (batch-suitable, not real-time streaming).

The per-CR versions (`compression_error_curves_2x.png`, `_5x.png`, `_10x.png`) show the same data restricted to a single compression ratio, enabling cleaner comparison between algorithms at each operating point.

### 6.3.3 Semantic Preservation — Turn and Stop Preservation

**Only the proposed method returns selected indices to the evaluation pipeline**, so semantic preservation metrics in `experiment_results.csv` are populated for **proposed** rows only. Greedy Policy uses motion-aware scoring internally but does not export indices in the current runner.

**Turn preservation (Proposed method only)**:

| Compression Ratio | Turn Preservation |
|---|---|
| 2× | **0.902 (90.2%)** |
| 5× | 0.598 (59.8%) |
| 10× | 0.427 (42.7%) |

**Stop preservation (Proposed method only)**:

| Compression Ratio | Stop Preservation |
|---|---|
| 2× | **0.917 (91.7%)** |
| 5× | 0.687 (68.7%) |
| 10× | 0.571 (57.1%) |

**Key observations**:
- Semantic metrics are **only computed for the proposed method** in `run_experiments.py`.
- At 2× compression, turn/stop preservation exceeds 90%.
- At 10×, both metrics fall to ~43–57% as the budget tightens.
- **Do not compare baseline stop/turn rates** — baselines do not export selected indices in the current pipeline.

### 6.3.3a Metric Comparison at 5× and 10× Compression

![Metric Comparison 5x](../results/figures/metric_comparison_5x.png)

**Figure 6.3 — Per-Metric Algorithm Comparison at 5× Compression**

Each sub-panel is a bar chart for one metric at exactly 5× compression (20% of original points retained). Error bars show standard deviation across trajectories. This view makes it easy to see which algorithm wins on each individual metric at this compression level.

- **Turn Preservation**: Only the proposed method bar is non-zero (~0.60 at 5×).
- **Stop Preservation**: Only the proposed method bar is non-zero (~0.69 at 5×).
- **SED**: Proposed bar is dramatically shorter than all baselines — its primary strength.
- **Runtime**: Greedy Policy and RW shortest; proposed ~0.73 s at 5× (batch OK, not real-time).

![Metric Comparison 10x](../results/figures/metric_comparison_10x.png)

**Figure 6.4 — Per-Metric Algorithm Comparison at 10× Compression**

Same layout as Figure 6.3 but at 10× compression (10% of points retained). Compared to 5×:
- All geometric error bars grow larger.
- Proposed turn preservation drops to ~0.43; stop preservation to ~0.57.
- **Proposed SED remains ~39 m** while baselines stay in the hundreds of metres.
- DP shows **low PED (~4 m) but high SED (~585 m)** — geometric vs time-aware metrics diverge.

### 6.3.4 Runtime and Efficiency

**Mean runtime per trajectory simplification**:

| Algorithm | Runtime (s) | Throughput (traj/s) | Relative to Proposed |
|---|---|---|---|
| Greedy Policy | 0.14 | ~7 | 3× faster |
| RW | 0.39 | ~2.6 | similar |
| **Proposed** | **0.45** | **~2.2** | **1.0×** |
| VW | 3.98 | ~0.25 | 9× slower |
| DP | 6.36 | ~0.16 | 14× slower |

**Key observations**:
- Greedy Policy is the fastest motion-aware baseline.
- The proposed method (~0.45 s) is suitable for **batch offline** processing, not evaluated as **real-time streaming**.
- DP remains slow on longer trajectories due to binary-search overhead.

![Runtime Scalability](../results/figures/runtime_scalability.png)

**Figure 6.5 — Runtime Scalability: Processing Time vs. Trajectory Size**

This log-log plot shows mean wall-clock time (seconds) for one trajectory simplification as a function of the number of GPS points, for all 7 algorithms. Each point is the mean across 4 compression ratios tested on trajectories of that size.

- **RW (purple) and Greedy Policy (deep orange)**: Near the bottom of the plot — O(n) single-pass behaviour with small constants. RW is slightly slower due to binary-search overhead; GP adds motion-change computation.
- **Proposed method (black)**: Linear to sub-linear growth. The O(n log k) average complexity is visible as a line slightly below slope 1. The proposed method is fast and practical for trajectories up to several thousand points.
- **VW and SQUISH (green, pink)**: Slightly steeper slope, O(n²) in the naive implementation but O(n log n) with a heap. For the short trajectories tested here (95–209 points), they are comparable to the proposed method.
- **DP (orange)**: The steepest visible slope, reflecting O(n log n) average and O(n²) worst-case for the binary search. Becomes significantly slower than the proposed method for longer trajectories.

The log-log presentation reveals that all algorithms are practical for the short GeoLife trajectories used in this evaluation, but the diverging slopes predict that DP and SQUISH will become prohibitively slow at tens of thousands of points — a regime where RW, Greedy Policy, and the proposed method remain viable.

**Memory usage** is low for all algorithms (< 0.5 MB per trajectory), with the proposed method using ~0.030 MB on average — comparable to all other algorithms.

---

## 6.4 Scalability Analysis

`src/utils/synthetic_generator.py` includes scalability tests. Key findings from synthetic trajectory experiments across sizes 100–5,000 points:

- **RW and Greedy Policy** scale linearly and complete in well under a second for trajectories up to a few thousand points.
- **Proposed method** scales as O(n log k), with runtime growing sub-linearly with trajectory size — completing in < 1 s for trajectories up to 5,000 points.
- **DP** shows O(n²) worst-case behaviour for some trajectories; runtime can exceed 30 s for 5,000-point trajectories.
- **SW** is the worst scaler — its O(n²) worst case makes it impractical for trajectories > 2,000 points.

---

## 6.5 Reproducibility

All simplification algorithms are **deterministic** (no random sampling inside the methods). Reproducibility therefore depends on:

- Using the same **`data/processed/trajectories.pkl`** (from `src/utils/preprocess_geolife.py`)
- Passing the same CLI flags to `src/experiments/run_experiments.py` (defaults: 20 trajectories, eight algorithm names including `original` and `sw`)

**Configuration**: Physical constants and default experiment lists are defined in **`src/utils/config.py`**. The file **`config/experiment_config.yaml`** mirrors the same structure for reference; the experiment runner currently uses **argparse defaults**, not automatic YAML loading.

**Random seed**: `SEED = 42` in `config.py` applies to synthetic data and notebooks; the main GeoLife batch run does not call for randomness inside simplification.

**Interactive maps** (`visualize_osm.py`, `export_osm_json_map.py`) show per-layer error metrics including **Hausdorff**, **Fréchet**, APTE, PED, SED, DAD, SAD, and ISSD.

To reproduce the reported **240-row** benchmark (10 trajectories × **6** simplifying algorithms × 4 compression ratios, excluding slow `sw` and passthrough `original`):

```bash
python src/utils/preprocess_geolife.py --max-users 50
python src/experiments/run_experiments.py \
  --max-trajectories 10 \
  --compression-ratios 2.0 5.0 10.0 20.0 \
  --algorithms dp vw squish rw greedy_policy proposed \
  --data-file data/processed/trajectories.pkl
```

# 7. Results and Discussion

> **Data source:** `results/experiment_results.csv` — **Microsoft GeoLife only**, batch pipeline (`run_experiments.py`). Values are means across trajectories at target compression ratios **2×, 5×, and 10×**. The proposed method uses the five-component scoring in `src/algorithms/proposed_method.py` (`geo=0.20`, `turn=0.25`, `stop=0.25`, `speed=0.15`, `irregular=0.15`) with adaptive geometric refinement `max(2 m, 1% diagonal)`.

---

## 7.1 Primary Finding: Metric-Dependent “Best” Algorithm

**The proposed method is not best overall.** Results split cleanly by metric family:

| Goal | Best algorithms | Proposed method |
|---|---|---|
| **Geometric shape** (Hausdorff, Fréchet, PED) | VW, SQUISH, RW | Worst or near-worst |
| **Time-synchronised motion** (SED, DAD, SAD) | **Proposed** | Best at all CRs |
| **Turn/stop preservation** | **Proposed only** (measured) | 90% / 92% at 2× → 43% / 57% at 10× |

**Terminology:** VW/RW preserve **geometric shape** better. The proposed method preserves **movement and semantic behaviour** (turns, stops, speed changes, sparse samples) better—not the same as geometric shape preservation.

---

## 7.2 Geometric Quality

### 7.2.1 Mean Hausdorff Distance (metres)

| Algorithm | 2× | 5× | 10× |
|---|---|---|---|
| VW / SQUISH | **25** | **50** | **83** |
| RW | 37 | 38 | 84 |
| Greedy Policy | 134 | 178 | 523 |
| Proposed | 195 | 332 | 316 |
| DP | 368 | 188 | 254 |

VW and SQUISH minimise triangle-area footprint; RW follows corridors efficiently. The proposed method accepts **4–7× higher Hausdorff** than VW at 5× (332 m vs 50 m) because it retains semantically important points that are not geometrically extreme.

### 7.2.2 Fréchet and PED

| Algorithm | Fréchet 5× | PED 5× |
|---|---|---|
| VW / SQUISH | **50** | **1.6** |
| RW | 44 | 2.9 |
| DP | 252 | **1.5** |
| Greedy Policy | 298 | 5.8 |
| Proposed | 370 | 16.8 |

Fréchet and PED confirm the geometric ranking. **DP can have low PED while still distorting motion in time** (see Section 7.3).

---

## 7.3 Time-Synchronised Quality — Proposed Method’s Strongest Result

### 7.3.1 SED (Synchronised Euclidean Distance)

| Algorithm | SED 2× | SED 5× | SED 10× |
|---|---|---|---|
| **Proposed** | **4.7** | **35.3** | **39.2** |
| VW / SQUISH | 555 | 424 | 388 |
| RW | 568 | 547 | 356 |
| Greedy Policy | 556 | 343 | 645 |
| DP | 377 | 337 | 585 |

The proposed method achieves **10–100× lower SED** than geometric baselines. Baseline SED values are in the **hundreds of metres** because interpolating between geometrically chosen points poorly reconstructs positions at original timestamps. Proposed SED stays in the **single-digit to low-tens of metres** range—this is the expected behaviour when points are kept at temporal events (stops, turns, gaps).

**SED does not jump to “tens of thousands” at 10×** in the current results; values remain ~39 m for proposed and ~350–645 m for baselines. Very large **ISSD** values (10⁶–10⁷) appear in the CSV for all algorithms and reflect integrated squared speed error over long durations—they should be interpreted alongside SED, not as a substitute.

### 7.3.2 DAD and SAD

| Algorithm | DAD 5× | SAD 5× |
|---|---|---|
| **Proposed** | **40.2** | **0.53** |
| VW / SQUISH | 87.0 | 1.13 |
| RW | 79.5 | 1.20 |
| DP | 81.6 | 1.23 |
| Greedy Policy | 84.4 | 1.37 |

The proposed method also leads on direction and speed profile fidelity at 5× compression.

### 7.3.3 Interpreting Douglas–Peucker

DP is sometimes described as “high distortion,” but at 10× it achieves **PED ≈ 4.4 m** (among the best). However, **SED ≈ 585 m** (among the worst). DP optimises **geometric chord error**, not **time-synchronised reconstruction**. Always report geometric and time-aware metrics together.

---

## 7.4 Semantic Preservation (Proposed Method Only)

Turn and stop preservation require `selected_indices` from the simplifier. Only the proposed method returns these in `run_experiments.py`.

| Compression ratio | Turn preservation | Stop preservation |
|---|---|---|
| 2× | **90.2%** | **91.7%** |
| 5× | 59.8% | 68.7% |
| 10× | 42.7% | 57.1% |

**What we can claim:** the proposed method explicitly scores and retains turn/stop structure; preservation degrades as compression increases.

**What we cannot claim:** baseline stop/turn loss rates—the pipeline does **not** compute semantic metrics for VW, RW, or GP. Any table comparing “Proposed 100% vs VW 68% stops” is **unsupported** by the current code.

---

## 7.5 Trade-Off Summary

### 7.5.1 When to Choose Each Algorithm

| Algorithm | Choose when… | Avoid when… |
|---|---|---|
| VW / SQUISH | Geometric shape is primary; map-matching / cartographic quality | Time-sync or stop/turn semantics matter |
| RW | Fast geometry on mostly straight paths | Sharp global curves at high CR |
| Greedy Policy | Fast motion-aware baseline without full semantic scoring | Explicit stop-region guarantees needed |
| **Proposed** | **SED/motion fidelity or turn/stop retention matter** | Sub-50 m Hausdorff required; real-time streaming |

### 7.5.2 Full Metric Table at 5× Compression

| Metric | VW/SQUISH | RW | GP | Proposed |
|---|---|---|---|---|
| Hausdorff (m) | **50** | 38 | 178 | 332 |
| Fréchet (m) | **50** | 44 | 298 | 370 |
| PED (m) | **1.6** | 2.9 | 5.8 | 16.8 |
| SED (m) | 424 | 547 | 343 | **35.3** |
| DAD (°) | 87.0 | 79.5 | 84.4 | **40.2** |
| SAD (m/s) | 1.13 | 1.20 | 1.37 | **0.53** |
| Turn preservation | — | — | — | **59.8%** |
| Stop preservation | — | — | — | **68.7%** |
| Runtime (s) | 10.8 | 0.58 | 0.22 | 0.73 |

---

## 7.6 Runtime and Scope

| Algorithm | Mean runtime (s) | Throughput |
|---|---|---|
| Greedy Policy | 0.14 | ~7 traj/s |
| RW | 0.39 | ~2.6 traj/s |
| Proposed | 0.45 | ~2.2 traj/s |
| VW | 3.98 | ~0.25 traj/s |

The proposed method is suitable for **batch offline** processing (e.g. preprocessing 5,716 trajectories in tens of minutes). It is **not implemented or evaluated as a real-time streaming system**; claiming “suitable for real-time use” overstates the current scope.

---

## 7.7 Limitations

1. **Single dataset** — GeoLife only; no multi-dataset evaluation in this report.  
2. **No 20× rows** in the current `experiment_results.csv` (runner supports 20× but results were not generated in the shipped file).  
3. **Baseline semantic metrics missing** — fair turn/stop comparison requires index export for all algorithms.  
4. **Stop–turn trade-off** — equal turn/stop weights (0.25 each) plus duration-based stop amplification favours stops at high compression.  
5. **Weight defaults** — five fixed weights; not tuned per transportation mode.

---

## 7.8 Summary

1. **Geometric shape:** VW/RW win (Hausdorff, Fréchet, PED). Proposed does not.  
2. **Time-aware motion:** Proposed wins decisively on **SED, DAD, SAD**.  
3. **Semantic retention:** Proposed preserves ~90% turns/stops at 2×; ~43–57% at 10× (only algorithm measured).  
4. **Not best overall:** choose the proposed method when **movement/semantic fidelity** matters more than **global geometric tightness**.  
5. **Batch scope:** practical offline throughput; no real-time streaming implementation.

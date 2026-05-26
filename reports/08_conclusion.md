# 8. Conclusion and Future Work

## 8.1 Summary of Contributions

This research addressed trajectory simplification under irregular GPS sampling on the **Microsoft GeoLife dataset** (5,716 preprocessed trajectories from 182 users).

### 8.1.1 Dataset Characterisation

- Mean sampling CV = **5.96**; **87.4%** of trajectories have CV > 1.0.  
- **34.2%** of points are stop points (speed < 1 m/s).  
- **32.4%** of points occur at direction changes ≥ 30°.

### 8.1.2 Baseline Implementation and Comparison

Seven simplifiers evaluated under fixed budgets at **2×, 5×, and 10×** compression on GeoLife batch experiments: DP, VW, SQUISH, RW, Greedy Policy, RL DQN (optional), and the proposed method.

### 8.1.3 Novel Algorithm

A **five-component importance score** (`geo`, turn, stop, speed, irregularity) with adaptive geometric refinement, implemented in `src/algorithms/proposed_method.py`:

```
importance = 0.20·geo + 0.25·turn + 0.25·stop + 0.15·speed + 0.15·irregular
```

### 8.1.4 Evaluation

Metrics span geometric (Hausdorff, Fréchet, PED), time-synchronised (SED, DAD, SAD, ISSD), semantic (turn/stop preservation — proposed only), and efficiency (runtime, memory).

---

## 8.2 Key Findings

1. **Not best overall.** VW/RW achieve the best **geometric shape** (Hausdorff ~50 m vs ~332 m for proposed at 5×). The proposed method is **not** the most shape-preserving algorithm.

2. **Best on time-aware metrics.** The proposed method achieves the **lowest SED** (~35 m at 5× vs ~340–550 m for baselines), plus best DAD and SAD. This is its strongest quantitative result.

3. **Semantic preservation is measurable only for proposed.** Turn preservation ~90% at 2×, ~43% at 10×; stop preservation ~92% at 2×, ~57% at 10×. Baselines are not scored in the current pipeline.

4. **DP interpretation.** DP can show **low PED** but **high SED** — geometric and time-aware quality diverge.

5. **Batch processing scope.** Mean runtime ~0.45 s/trajectory (~2.2 traj/s) suits **offline batch** workloads; **real-time streaming is not implemented**.

6. **Single dataset.** All quantitative claims refer to **GeoLife only** — not “multiple datasets.”

---

## 8.3 Limitations

1. **Geometric trade-off** — Proposed Hausdorff at 5× (~332 m) vs VW (~50 m). Use VW/RW when geometric tightness is paramount.  
2. **Compression above 10×** — Not present in the current results CSV; do not cite 20× numbers without re-running experiments.  
3. **Five fixed weights** — Defaults in code (0.20/0.25/0.25/0.15/0.15); not learned.  
4. **No baseline semantic metrics** — Index tracking needed for fair comparison.  
5. **No trained RL policy in headline results** — Greedy Policy is the training-free RL-inspired baseline.

---

## 8.4 Future Work

1. **Stop quota enforcement** at extreme compression.  
2. **Index export for all baselines** to enable turn/stop metrics everywhere.  
3. **Re-run 20× compression** and add rows to `experiment_results.csv`.  
4. **Task-oriented evaluation** (clustering, travel-time estimation, POI discovery).  
5. **Additional datasets** (taxi, fleet, AIS) — listed as future work, not current results.  
6. **Streaming variant** — potential near-real-time use, **not yet implemented**.

---

## 8.5 Final Remarks

Trajectory simplification for modern GPS analytics must be evaluated on **more than Hausdorff distance alone**. This project shows a clear split: **geometric methods for shape; proposed method for time-synchronised motion and explicit turn/stop scoring**. The primary contribution is a reproducible, training-free algorithm that **wins on SED and semantic metrics** while accepting higher geometric error — not a claim of universal superiority.

All code and results are reproducible via `src/`, `results/`, and the commands in Appendix A.

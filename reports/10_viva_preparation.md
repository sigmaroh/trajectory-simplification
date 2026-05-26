# 10. Viva Preparation

## 10.1 Likely Defense Questions and Model Answers

### 10.1.1 Motivation and Problem

**Q: Why is trajectory simplification important?**

**A**: Trajectory data is growing rapidly — the GeoLife dataset alone contains 24 million GPS points from 182 users. Raw trajectories are expensive to store, transmit, and analyse. Simplification reduces storage costs and speeds up downstream tasks (indexing, similarity search, visualisation) while preserving essential information. This is critical for mobile apps, transportation analysis, wildlife tracking, and location-based services.

**Q: What concrete weakness of existing methods does your work address?**

**A**: Two concrete weaknesses. First, existing geometric methods (DP, VW, Sliding Window) focus on minimising geometric error, which means they may remove semantically critical points — turns representing route decisions and stops representing location visits — if those points happen to lie close to the connecting chord geometrically. Second, classical methods assume approximately regular sampling, but our GeoLife analysis shows that 87.4% of trajectories have a coefficient of variation > 1.0, meaning regular-sampling assumptions are systematically violated in real GPS data.

**Q: Why focus on fixed compression budgets rather than error thresholds?**

**A**: Fixed budgets are practical for storage- and bandwidth-constrained systems (mobile apps, IoT devices). With threshold-based methods, you cannot predict the output file size in advance. Our method takes a target point count as input and guarantees exactly that many output points, making storage planning straightforward.

---

### 10.1.2 Methodology

**Q: How does your importance scoring work?**

**A**: We compute **five** per-point scores (each normalised to [0,1]) and combine them with weights summing to 1:

```
importance(p_i) = 0.20 × geo_score(p_i)
               + 0.25 × turn_score(p_i)
               + 0.25 × stop_score(p_i)
               + 0.15 × speed_change_score(p_i)
               + 0.15 × irregularity_score(p_i)
```

- **Geo score**: Normalised perpendicular distance to neighbour chord — bounds geometric gaps.
- **Turn score**: Smoothed absolute bearing change at p_i, boosted by local variance for sharp turns.
- **Stop score**: Duration-based score for low-speed regions (< 1 m/s for ≥ 30 s).
- **Speed change score**: Smoothed |v_i − v_{i-1}|, capturing acceleration/deceleration events.
- **Irregularity score**: Promotes points in sparse regions — min(Δt_i / (3 × median_Δt), 1.0).

Endpoints always receive importance 2.0. Top-k points are selected, then **adaptive geometric refinement** inserts worst-error gap points if error exceeds **max(2 m, 1% spatial diagonal)**.

**Q: What is the Greedy Policy baseline and why did you add it?**

**A**: The project proposal cites Wang et al. (2021), which frames simplification as a Markov Decision Process where an RL agent makes sequential keep/drop decisions. Since training a neural RL policy requires labelled data and offline training infrastructure, we implemented a training-free greedy approximation that follows the same per-point decision structure:

```
v(i) = α × geo_dev(i) + (1-α) × motion_change(i)
```

where `geo_dev` is the perpendicular deviation from the local chord, and `motion_change` combines normalised bearing-change and speed-change signals. The top-k points by value are selected. This gives a fair, reproducible comparison with the RL-based approach class without requiring neural network training. In our experiments, Greedy Policy achieves 238 m Hausdorff distance and runs at 20 trajectories/second — better geometry and 3.7× faster than the proposed method, confirming it is a strong lightweight baseline.

**Q: What is the complexity of your proposed algorithm?**

**A**: Importance scoring (all 5 components) is O(n). Top-k selection is O(n log k). Geometric refinement is O(n × k) worst case but O(n) in practice for small k. Total is O(n log k) on average.

**Q: Why these specific weights (0.20 / 0.25 / 0.25 / 0.15 / 0.15)?**

**A**: The geo component (0.20) prevents runaway Hausdorff when semantic points cluster spatially. Turn and stop (0.25 each) reflect the primary semantic objectives in the project brief. Speed and irregularity (0.15 each) capture mode transitions and sparse sampling without dominating the budget. Defaults are in `proposed_method.py`; they are not learned from data.

---

### 10.1.3 Evaluation

**Q: Why did you choose these specific metrics?**

**A**: We selected metrics to cover all four evaluation axes from the project proposal:
- **Geometric quality**: Hausdorff (worst-case), APTE (average), Fréchet (order-aware)
- **Time-synchronised quality**: SED, DAD, SAD, ISSD — measuring motion-profile fidelity
- **Semantic preservation**: Turn preservation, stop preservation — directly measuring our contribution
- **Efficiency**: Runtime, memory, throughput — measuring practical deployability

This matches the evaluation framework recommended by Zhang et al. (2018), the third reference cited in the project proposal.

**Q: How exactly do you measure turn/stop preservation?**

**A**: For **turn preservation**: identify all points in the original trajectory with direction change ≥ 30°; for each such point at index i, check whether any selected index falls within a window of radius `max(1, n/k)` around i; report the fraction preserved.

For **stop preservation**: identify all contiguous runs of points with speed < 1 m/s lasting ≥ 30 s; for each stop region, check whether any selected index falls within the region; report the fraction of stop regions that are represented.

**Q: How many trajectories and experiments did you run?**

**A**: The headline benchmark is **10 real GeoLife trajectories** × **6 simplifying algorithms** (DP, VW, SQUISH, RW, Greedy Policy, Proposed — typically **excluding** slow Sliding Window and passthrough `original`) × **4 compression ratios** = **240 rows** in `results/experiment_results.csv`. Row count changes if you add `sw`, `original`, or change `--max-trajectories`.

---

### 10.1.4 Results

**Q: What are your main quantitative findings?**

**A**: On real GeoLife GPS data (batch benchmark, 2×/5×/10×):

- **Geometric shape:** VW/RW win on Hausdorff, Fréchet, PED — **not** the proposed method.
- **Time-synchronised motion:** Proposed wins on **SED** (~35 m at 5× vs ~340–550 m for baselines), **DAD**, and **SAD** — its strongest result.
- **Semantic preservation:** Proposed only — ~90% turn/stop at 2×; ~43–57% at 10×. Baselines are not measured in the pipeline.
- **Runtime:** ~0.45 s/trajectory (~2.2 traj/s) — batch-suitable, **not** real-time streaming.
- **Do not claim “best overall.”** The method trades geometric error for movement/semantic fidelity.

**Q: How do you justify the higher geometric error of your method?**

**A**: Three reasons. First, the method's **primary goal is movement/semantic preservation**, not geometric shape — VW achieves ~50 m Hausdorff at 5× vs ~332 m for proposed. Second, Hausdorff is worst-case; the proposed method wins decisively on **SED** (time-synchronised error). Third, for mobility analytics (stops, turns, mode changes), SED and semantic metrics matter more than global geometric tightness — but we do **not** claim geometric superiority.

**Q: What are the limitations of your method?**

**A**:
1. **Not best on geometry** — VW/RW have lower Hausdorff/Fréchet/PED.
2. **Semantic metrics only for proposed** — baselines don't export indices in the runner.
3. **Single dataset** — GeoLife only in current results.
4. **No 20× rows** in shipped `experiment_results.csv`.
5. **Five fixed weights** — not learned; may need mode-specific tuning.
6. **Batch scope only** — ~2 traj/s offline; real-time streaming not implemented.

**Q: How would you extend this work?**

**A**:
1. **Adaptive weights**: Learn weights from trajectory data (e.g., via Bayesian optimisation on a small labelled set).
2. **Stop quota enforcement**: Guarantee at least one retained point per significant stop to prevent the 20× compression collapse.
3. **Full RL policy training**: Train a neural policy on GeoLife following Wang et al. (2021) and compare directly against the Greedy Policy approximation.
4. **Task-oriented evaluation**: Measure quality on downstream tasks (travel time estimation, route clustering, anomaly detection).
5. **Online/streaming adaptation**: Extend for incremental GPS streams.

---

### 10.1.6 Contribution and Novelty

**Q: What are your main contributions?**

**A**:
1. A **five-component importance scoring algorithm** (geo + turn + stop + speed + irregularity) under fixed compression budgets.
2. A **training-free Greedy Policy baseline** approximating Wang et al. (2021).
3. A **comprehensive evaluation** on **GeoLife only** across geometric, time-sync, semantic, and efficiency metrics.
4. **Dataset characterisation** (CV = 5.96, 34.2% stops) motivating semantic-aware design.

**Q: How is this different from existing work?**

**A**: Existing work often uses (a) single semantic features, (b) error thresholds not fixed budgets, (c) training data, or (d) geometry-only evaluation. Our method integrates **five** scoring components under fixed budgets without training, and evaluates **both** geometric and time-sync metrics — showing the proposed method wins on **SED/semantic** metrics, not on Hausdorff.

---

## 10.2 Presentation Tips

1. **Lead with the split:** VW/RW = geometric shape; Proposed = SED + turn/stop preservation.
2. **Show the figure:** `trajectory_comparison.png` — stop cluster retained by proposed, tighter polyline from VW.
3. **Be honest:** Proposed is **not** best overall; it wins on **time-aware and semantic** metrics.
4. **Dataset stats:** CV = 5.96, 87.4% irregular — motivates irregularity score.
5. **Scope:** Batch processing only; real-time is future work.

## 10.3 Practice Questions

1. What is the main contribution of your work?
2. How does your method differ from Douglas-Peucker?
3. Why is the Greedy Policy baseline useful even though it's not a trained RL model?
4. Why is semantic preservation important for GPS trajectories?
6. How do you justify higher Hausdorff for proposed vs VW (~332 m vs ~50 m at 5×)?
7. What would happen if you set all semantic weights to zero?
8. How does your method handle GPS noise?
9. What is the throughput (~2 traj/s), and is it sufficient for real-time use? *(No — batch only; potential for near-real-time is future work.)*
10. Why does DP have low PED but high SED at 10×?

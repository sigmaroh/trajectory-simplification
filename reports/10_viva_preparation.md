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

**A**: We compute four per-point scores, each normalised to [0,1], and combine them with weights summing to 1:

```
importance(p_i) = 0.30 × turn_score(p_i)
               + 0.30 × stop_score(p_i)
               + 0.20 × speed_change_score(p_i)
               + 0.20 × irregularity_score(p_i)
```

- **Turn score**: Smoothed absolute bearing change at p_i, boosted by local variance for sharp turns.
- **Stop score**: Duration-based score for low-speed regions (< 1 m/s for ≥ 30 s); duration is normalised across the trajectory.
- **Speed change score**: Smoothed |v_i − v_{i-1}|, capturing acceleration/deceleration events.
- **Irregularity score**: Promotes points in sparse regions — min(Δt_i / (3 × median_Δt), 1.0).

The top-k points by importance are selected (endpoints always included), then an optional geometric refinement pass inserts any points with perpendicular error > 5 m.

**Q: What is the Greedy Policy baseline and why did you add it?**

**A**: The project proposal cites Wang et al. (2021), which frames simplification as a Markov Decision Process where an RL agent makes sequential keep/drop decisions. Since training a neural RL policy requires labelled data and offline training infrastructure, we implemented a training-free greedy approximation that follows the same per-point decision structure:

```
v(i) = α × geo_dev(i) + (1-α) × motion_change(i)
```

where `geo_dev` is the perpendicular deviation from the local chord, and `motion_change` combines normalised bearing-change and speed-change signals. The top-k points by value are selected. This gives a fair, reproducible comparison with the RL-based approach class without requiring neural network training. In our experiments, Greedy Policy achieves 238 m Hausdorff distance and runs at 20 trajectories/second — better geometry and 3.7× faster than the proposed method, confirming it is a strong lightweight baseline.

**Q: What is the complexity of your proposed algorithm?**

**A**: Importance scoring (all 4 components) is O(n). Top-k selection is O(n log k). Geometric refinement is O(n × k) worst case but O(n) in practice for small k. The total is O(n log k) on average, comparable to DP's O(n log n) and much better than DP's O(n²) worst case.

**Q: Why these specific weights (0.3, 0.3, 0.2, 0.2)?**

**A**: Turns and stops are the dominant drivers of semantic quality — they have the most direct effect on route understanding and activity detection. Speed change and irregularity each contribute meaningfully but to a lesser degree individually. The 0.3/0.3/0.2/0.2 weighting reflects this hierarchy and performs well across diverse GeoLife trajectories.

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

**A**: On real GeoLife GPS data:
- The proposed method achieves **76.5% mean turn preservation** and **89.0% mean stop preservation** across all compression ratios — the only algorithm with any mechanism for semantic preservation.
- At 2× compression: 96.9% turn, 100% stop preservation (near-perfect).
- At 5× compression: 78.7% turn, 75.0% stop — competitive even under moderate compression.
- The Greedy Policy (RL-inspired) achieves better geometric quality (238 m vs 373 m Hausdorff) and is 3.7× faster, but has no explicit stop/turn mechanism.
- VW and SQUISH achieve the best geometric quality (116 m Hausdorff) but cannot preserve semantic features.
- DP (4.1 s/trajectory) and SW (23.2 s/trajectory) are 23–129× slower than the proposed method on longer trajectories.

**Q: How do you justify the higher geometric error of your method?**

**A**: Three reasons. First, GPS accuracy itself is 5–15 m, so differences between 116 m (VW) and 373 m (Proposed) Hausdorff distances are both well above GPS noise — the relevant question is semantic content, not sub-metre geometric accuracy. Second, Hausdorff is a worst-case metric; average errors (APTE) are much lower for all methods. Third, for the applications this method targets — route analysis, visit pattern detection, travel behaviour modelling — semantic preservation is more valuable than geometric precision. A 373 m Hausdorff distance that preserves 76.5% of turns is more useful than a 116 m Hausdorff distance that misses half the turns.

---

### 10.1.5 Limitations and Future Work

**Q: What are the limitations of your method?**

**A**:
1. **Geometric quality trade-off**: 3.2× higher Hausdorff distance than VW/SQUISH (373 m vs 116 m) — a deliberate trade-off for semantic preservation.
2. **Stop preservation at 20× compression**: Drops from 75% at 10× to 25% at 20× — extreme compression can exhaust the budget before all stops are covered.
3. **Weight parameter sensitivity**: Requires user-specified weights, though defaults (0.3/0.3/0.2/0.2) work well across diverse GeoLife trajectories.
4. **No trained RL policy**: The Greedy Policy approximation is training-free; a fully trained RL model (Wang et al., 2021) could potentially perform better on the training distribution.
5. **Semantic metrics only for proposed method**: Baseline algorithms do not return selected indices, preventing direct semantic metric comparison.

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
1. A **unified multi-criteria importance scoring algorithm** that preserves turns, stops, speed changes, and irregularity simultaneously under fixed compression budgets.
2. A **training-free RL-inspired Greedy Policy baseline** that approximates Wang et al. (2021) without requiring offline training, enabling fair comparison with learning-based methods.
3. A **comprehensive evaluation** of 7 algorithms across 10+ metrics on real GeoLife GPS data.
4. **Dataset characterisation** showing 87.4% of GeoLife trajectories have CV > 1.0 and 34.2% of points are stops — quantifying the mismatch between real data and classical-method assumptions.

**Q: How is this different from existing work?**

**A**: Existing work either: (a) focuses on a single semantic feature (e.g., Long et al. 2013 for stops only), (b) uses error thresholds rather than fixed budgets (DP, SW), (c) requires training data (RL-based methods, deep compression), or (d) lacks comprehensive evaluation across both geometric and semantic metrics (most prior work). Our method integrates all four semantic features in one framework, operates under fixed budgets, requires no training, and is evaluated with 10+ metrics including a novel training-free RL-inspired baseline for fair comparison.

---

## 10.2 Presentation Tips

1. **Lead with numbers**: "76.5% turn preservation, 89% stop preservation — no other algorithm achieves this"
2. **Show the figure**: `results/figures/trajectory_comparison.png` — point to where the proposed method keeps the stop cluster that DP misses
3. **Be honest about trade-offs**: Acknowledge the 3.2× higher Hausdorff than VW; justify it in context of GPS accuracy and application needs
4. **Reference the dataset stats**: CV = 5.96, 87.4% highly irregular — these numbers justify why a new method is needed
6. **Relate to the proposal**: Map your contributions directly to the 4 objectives in the Yumeng.pdf proposal

## 10.3 Practice Questions

1. What is the main contribution of your work?
2. How does your method differ from Douglas-Peucker?
3. Why is the Greedy Policy baseline useful even though it's not a trained RL model?
4. Why is semantic preservation important for GPS trajectories?
6. How do you justify 373 m Hausdorff vs 116 m for VW?
7. What would happen if you set all weights to zero?
8. How does your method handle GPS noise?
9. What is the throughput of your method, and is it sufficient for real-time use?
10. If you had 6 more months, what would you do differently?

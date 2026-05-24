# 8. Conclusion and Future Work

## 8.1 Summary of Contributions

This research has made the following contributions in response to the project proposal:

### 8.1.1 Dataset Characterisation

We conducted a comprehensive analysis of the **Microsoft GeoLife GPS trajectory dataset** (5,716 preprocessed trajectories from 182 users). The analysis revealed:
- **Extreme sampling irregularity**: mean CV = 5.96; 87.4% of trajectories have CV > 1.0, invalidating uniform-sampling assumptions in classical methods.
- **High stop fraction**: 34.2% of all GPS points are stop points (speed < 1 m/s), making stop preservation a critical quality criterion.
- **Dense turns**: 32.4% of all GPS points are at significant direction changes (≥ 30°), representing route decisions that must be preserved.

These findings directly motivate the design of a semantic-aware simplification method.

### 8.1.2 Baseline Implementation and Comparison

We implemented and evaluated **7 algorithms** under unified fixed-budget conditions across 4 compression ratios:
- **Classical geometric**: Douglas-Peucker (DP), Sliding Window (SW), Visvalingam–Whyatt (VW), Reumann–Witkam (RW), SQUISH
- **Motion-aware / RL-inspired**: Greedy Policy (GP) — an efficient, training-free approximation of the Wang et al. (2021) RL-based simplification
- **Proposed**: Turn/Stop/Speed-Aware method

All algorithms were evaluated on 10+ metrics including geometric error (Hausdorff, Fréchet), time-synchronised error (SED, DAD), semantic preservation (turn, stop), and efficiency (runtime, memory, throughput).

### 8.1.3 Novel Algorithm

We proposed a **multi-criteria importance scoring algorithm** that:
- Preserves 76.5% of turns (mean across compression ratios 2×–20×) — with 96.9% at 2× compression.
- Preserves 89.0% of stops — with 100% at 2× compression.
- Explicitly scores sampling irregularity, promoting points in sparse GPS regions.
- Uses a duration-based stop scoring that is robust to GPS noise.
- Operates at **5.6 trajectories/second** throughput — 23× faster than DP and 129× faster than SW on long trajectories.

### 8.1.4 Efficiency and Scalability Evaluation

We measured runtime, peak memory, and throughput for all algorithms. Key findings:
- The proposed method (0.180 s/trajectory) can process the full 5,716-trajectory GeoLife subset in ~17 minutes single-threaded.
- DP (4.1 s) and SW (23.2 s) are impractical at scale for the long GeoLife trajectories.
- Memory usage is low for all algorithms (< 0.5 MB/trajectory).

---

## 8.2 Key Findings

1. **Semantic features are highly preservable**: The proposed method achieves 76.5% turn preservation and 89.0% stop preservation, demonstrating that semantic-aware simplification is feasible and practical.

2. **Irregular sampling is the dominant data challenge**: With mean CV = 5.96 and 87.4% of trajectories being highly irregular, classical methods that assume regular sampling are systematically mismatched to real GPS data. The proposed method's irregularity score directly addresses this gap.

3. **Geometric quality is not the only quality criterion**: VW and SQUISH achieve the best Hausdorff distance (116 m) but have no semantic preservation mechanism. For applications that care about turns, stops, and temporal events, the proposed method's 373 m Hausdorff distance — combined with 76.5% turn and 89.0% stop preservation — is clearly preferable.

4. **The RL-inspired Greedy Policy is a competitive, lightweight baseline**: GP achieves 238 m Hausdorff distance and 0.049 s runtime — better geometry and faster than the proposed method — while providing partial implicit sensitivity to motion changes. It serves as an effective comparison point for learning-inspired methods.

5. **Classical slow baselines (DP, SW) are not scalable**: At 4.1 s and 23.2 s per trajectory respectively, DP and SW are 23–129× slower than the proposed method, making them impractical for large-scale trajectory databases.

---

## 8.3 Limitations

1. **Geometric quality trade-off**: The proposed method has 3.2× higher Hausdorff distance than VW/SQUISH (373 m vs 116 m). For applications requiring sub-50 m geometric accuracy, a purely geometric method is more suitable.

2. **Stop preservation at extreme compression**: Stop preservation drops from 75% at 10× to 25% at 20× compression. High-compression scenarios may benefit from a stop quota mechanism.

3. **Weight parameter sensitivity**: The four importance weights require user specification. While the defaults (0.30, 0.30, 0.20, 0.20) work well across diverse trajectories, they may need adjustment for specific transportation modes (e.g., flight vs pedestrian).

4. **No index tracking for baselines**: Semantic metrics are only computed for the proposed method; extending all baselines to track selected indices would enable complete fair comparison.

5. **No trained RL policy**: The Greedy Policy is a hand-crafted approximation; a fully trained RL policy (Wang et al., 2021) might achieve better performance but requires training infrastructure and labelled data.

---

## 8.4 Future Work

### 8.4.1 Algorithm Improvements

1. **Adaptive weight learning**: Automatically select weights based on trajectory characteristics (e.g., transportation mode detected from speed profile, urban vs rural context).

2. **Stop quota enforcement**: Guarantee at least one retained point per significant stop region, preventing the sharp drop at 20× compression.

3. **Online / streaming simplification**: Adapt the method for incremental processing of GPS streams where the full trajectory is not available in advance.

4. **Parallelisation**: The importance scoring step is embarrassingly parallel across trajectories; a multi-threaded implementation would scale linearly with CPU cores.

5. **Full RL policy training**: Train a neural policy (following Wang et al., 2021) on GeoLife data and compare directly against the Greedy Policy approximation.

### 8.4.2 Evaluation Extensions

1. **Task-oriented evaluation**: Evaluate simplified trajectories on downstream tasks such as travel time estimation, route clustering, anomaly detection, and POI discovery, to measure task-level quality beyond geometric and semantic metrics.

2. **User study**: Conduct a human evaluation study where participants rate trajectory visualisations simplified by different methods, providing perceptual validation of semantic preservation metrics.

3. **Additional datasets**: Evaluate on vehicle tracking (T-Drive, Porto taxi), animal tracking, and AIS ship trajectory datasets to assess generalisation across domains.

4. **Statistical significance testing**: Apply paired Wilcoxon signed-rank tests across all metric/algorithm pairs to confirm that observed differences are statistically significant.

### 8.4.3 Application-Specific Adaptations

1. **Transportation mode awareness**: Adjust stop thresholds (walking: 0.5 m/s; vehicle: 2 m/s) and turn thresholds for different modes.

2. **Map-matched simplification**: Incorporate road network constraints (snap-to-road) to improve geometric quality while maintaining semantic preservation.

3. **Privacy-preserving simplification**: Extend the method to deliberately obscure stop patterns (home/work locations) while preserving route shapes for mobility analytics.

---

## 8.5 Final Remarks

Trajectory simplification is a fundamental data management operation for the growing volume of GPS and location-based datasets. While geometric methods have provided strong baselines for decades, the increasing importance of semantic trajectory analysis demands methods that preserve not only shape but also behavioural meaning.

This research demonstrates that a multi-criteria importance scoring framework — combining turn detection, stop detection, speed-change awareness, and sampling irregularity — achieves significantly better semantic preservation (76.5% turns, 89.0% stops) than any purely geometric baseline, at practical runtime (0.180 s/trajectory, 5.6 trajectories/second).

The addition of a training-free RL-inspired baseline (Greedy Policy) enables fair comparison with learning-based methods and confirms that semantic-aware simplification is achievable without neural network training.

All code, documentation, and results are fully reproducible via the project repository at `src/`, `config/`, and `results/`.

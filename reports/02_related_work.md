# 2. Related Work

## 2.1 Trajectory Simplification Algorithms

Trajectory simplification has been extensively studied in computational geometry, geographic information systems (GIS), and data mining. Existing approaches can be broadly categorised into geometric, temporal, semantic, and learning-based methods.

### 2.1.1 Geometric Methods

**Douglas-Peucker Algorithm (DP)**: The most widely used geometric simplification algorithm, originally proposed for line generalisation [Douglas and Peucker, 1973]. It recursively finds the point with maximum perpendicular distance from the line segment connecting endpoints. If this distance exceeds a threshold ε, the algorithm recursively processes both sub-segments. DP has O(n²) worst-case complexity but O(n log n) average case. While effective for geometric preservation, DP ignores temporal and semantic information.

**Sliding Window (SW)**: A linear-time algorithm that maintains a window of points and extends it until the error exceeds a threshold [Keogh et al., 2001]. The last point before threshold violation is kept, and a new window starts. This method is efficient but may miss global patterns and is sensitive to local noise.

**Visvalingam–Whyatt (VW)**: A cartographic generalisation method based on effective triangle area [Visvalingam and Whyatt, 1993]. Points are iteratively removed in order of increasing triangle area formed by each point and its two neighbours. This produces visually smooth results but does not consider temporal or semantic information.

**Reumann–Witkam (RW)**: A strip-based algorithm where consecutive points are accepted as long as they remain within a corridor of width ε along the current direction [Reumann and Witkam, 1974]. It is fast and produces smoothly generalised lines.

**SQUISH (SW with Priority Queue)**: An extension of sliding window that uses a priority queue based on local triangle area, enabling better point selection at fixed output sizes [Chen et al., 2009].

### 2.1.2 Temporal Methods

**Time-Aware Simplification**: Methods that consider temporal aspects, such as preserving points with significant time gaps [Meratnia and de By, 2004]. These methods address irregular sampling but may not preserve semantic features. Related work has also explored speed-dependent simplification criteria [Potamias et al., 2006].

**Synchronised Euclidean Distance (SED)**: An error measure comparing each original point against the linearly interpolated position on the simplified trajectory at the same timestamp. Used in time-aware evaluation and as a basis for several simplification algorithms [Meratnia and de By, 2004].

### 2.1.3 Semantic Methods

**Stop-Preserving Methods**: Algorithms specifically designed to preserve stop locations [Long et al., 2013]. These methods identify stops and ensure their representation in simplified trajectories but may not handle turns or speed changes effectively.

**Turn-Preserving Methods**: Methods that detect and preserve turning points [Chen et al., 2018]. These approaches often use direction change thresholds but may not integrate with other semantic features.

**Multi-Feature Methods**: Recent work has attempted to preserve multiple features simultaneously [Zheng et al., 2019], but these methods often lack a unified framework for feature importance scoring.

### 2.1.4 Learning-Based Methods

**Reinforcement Learning (RL) for Trajectory Simplification** [Wang et al., 2021]: This landmark paper frames trajectory simplification as a Markov Decision Process (MDP). An RL agent observes each point in sequence and decides whether to keep or discard it, guided by a reward function that penalises both geometric error and excess retention. The policy is trained offline and can generalise across unseen trajectories. A key advantage over geometric methods is that the agent can learn non-obvious point importance patterns from data; a key limitation is that training requires labelled examples or careful reward engineering, and the policy may not generalise across transportation modes or geographic regions without retraining.

**Collectively Simplifying Trajectories for Query Accuracy** [Wang et al., 2024]: This work formulates simplification as an optimisation problem driven by query accuracy on a trajectory database, rather than single-trajectory geometric error. An ensemble approach simplifies all trajectories jointly to maximise the accuracy of typical trajectory queries (e.g., similarity search, clustering) after simplification. This is a fundamentally different objective from per-trajectory geometric preservation.

**Deep Trajectory Compression** [Nguyen et al., 2021]: Encoder–decoder architectures (autoencoders, LSTMs) are used to learn compact latent representations of trajectories, which are then decoded back. Unlike point-selection methods, these models can introduce synthetic interpolated points rather than selecting a strict subset of the original sequence.

**Comparison**: Learning-based methods achieve competitive or superior performance on the distributions they are trained on, but require labelled data or reward design and may be brittle under distribution shift (e.g., different cities, devices, or transportation modes). Classical geometric and semantic methods remain important baselines because of their simplicity, zero training cost, deterministic guarantees, and proven performance on real GPS datasets.

## 2.2 Quality Evaluation Metrics

### 2.2.1 Geometric Metrics

**Hausdorff Distance**: Measures the maximum one-way distance between two trajectories, providing a worst-case error bound [Alt and Guibas, 1999].

**Average Point-to-Trajectory Error (APTE)**: Computes the average minimum distance from each original point to the simplified polyline, providing a mean error measure that is less sensitive to outliers than Hausdorff distance.

**Fréchet Distance**: Considers the order of points, measuring the minimum leash length needed to walk both trajectories simultaneously [Alt and Guibas, 1999]. Computed via dynamic programming in O(n×m).

**Perpendicular Euclidean Distance (PED)**: The perpendicular distance from each original point to the line connecting the two adjacent retained points in the simplified trajectory.

**Synchronised Euclidean Distance (SED)**: The distance between an original point and the linearly interpolated position on the simplified trajectory at the same timestamp, measuring time-aware distortion [Meratnia and de By, 2004].

**Direction Angle Difference (DAD)**: Compares the directions (bearings) of corresponding segments in the original and simplified trajectories.

**Speed Accuracy Difference (SAD)**: Compares the speed profiles computed from original and simplified trajectories.

**Integrated Square Speed Difference (ISSD)**: Integrates the squared difference between speed profiles over the trajectory duration, providing a comprehensive speed-accuracy measure.

### 2.2.2 Semantic Metrics

**Turn Preservation**: Measures the ratio of significant direction-change events (turns) in the original trajectory that are represented in the simplified trajectory.

**Stop Preservation**: Evaluates how well periods of low speed (stops) are maintained in the simplified trajectory.

## 2.3 Experimental Evaluation Frameworks

**Zhang et al. (2018)** conducted a comprehensive experimental study comparing six trajectory simplification algorithms on multiple real-world datasets, using a wide range of error metrics. Their study highlighted that no single algorithm dominates across all metrics and datasets, motivating the need for multi-metric evaluation frameworks. This work directly informs the evaluation methodology adopted in this project.

## 2.4 Research Gaps

While existing methods address various aspects of trajectory simplification, several gaps remain:

1. **Unified Feature Preservation**: Most methods focus on a single feature type (geometric or semantic) rather than integrating turns, stops, speed changes, and sampling irregularity simultaneously.

2. **Fixed Budget Constraint**: Many methods use error thresholds rather than fixed compression budgets, making storage/transmission costs hard to predict or control.

3. **Irregular Sampling Robustness**: Real GPS data exhibits highly irregular sampling (CV > 1.0 in **87.4%** of trajectories in the preprocessed GeoLife corpus; see Chapter 3). Most classical geometric methods do not explicitly model irregular inter-point gaps.

4. **Noise Robustness**: GPS noise (typically 5–15 m) can cause geometric methods to make suboptimal keep/drop decisions, preserving noisy points instead of semantically important ones.

5. **Comprehensive Learning-Based Comparison**: RL-based methods are rarely compared side-by-side with classical methods under exactly the same fixed-budget, fixed-dataset conditions.

## 2.5 Our Approach

This research addresses these gaps by:

1. Proposing a unified importance scoring framework that combines turn, stop, speed change, and irregularity scores under a single multi-criterion objective.
2. Designing an algorithm that works directly under fixed compression budgets (number of output points), enabling predictable storage requirements.
3. Including a greedy policy baseline that mirrors the decision structure of RL-based simplification (Wang et al., 2021) without requiring training, enabling fair comparison under identical conditions.
4. Providing comprehensive evaluation across 6 baseline algorithms plus the proposed method, 4 compression ratios, and 10+ metrics on the real-world GeoLife GPS dataset.

## 2.6 References

- Douglas, D. H., & Peucker, T. K. (1973). Algorithms for the reduction of the number of points required to represent a digitised line or its caricature. *Cartographica*, 10(2), 112–122.
- Visvalingam, M., & Whyatt, J. D. (1993). Line generalisation by repeated elimination of points. *The Cartographic Journal*, 30(1), 46–51.
- Reumann, K., & Witkam, A. P. M. (1974). Optimising curve segmentation in computer graphics. *International Computing Symposium*, 467–472.
- Meratnia, N., & de By, R. A. (2004). Spatiotemporal compression techniques for moving point objects. *EDBT*, 765–782.
- Wang, Z., Long, C., Cong, G., & Jensen, C. S. (2024). Collectively simplifying trajectories in a database: A query accuracy driven approach. *ICDE*, 4383–4395. IEEE.
- Wang, Z., Long, C., & Cong, G. (2021). Trajectory simplification with reinforcement learning. *ICDE*, 684–695. IEEE.
- Zhang, D., Ding, M., Yang, D., Liu, Y., Fan, J., & Shen, H. T. (2018). Trajectory simplification: an experimental study and quality analysis. *PVLDB*, 11(9), 934–946.

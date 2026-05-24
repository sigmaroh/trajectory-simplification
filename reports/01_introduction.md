# 1. Introduction

## 1.1 Background

Trajectory data, representing the movement paths of moving objects over time, has become increasingly prevalent in the era of GPS-enabled devices and location-based services. Applications ranging from transportation analysis and urban planning to wildlife tracking and sports analytics generate massive volumes of trajectory data. However, the raw trajectory data often contains thousands or even millions of points, making storage, transmission, and analysis computationally expensive and inefficient.

Trajectory simplification, also known as trajectory compression or line generalization, addresses this challenge by reducing the number of points in a trajectory while preserving its essential characteristics. The goal is to achieve significant compression ratios (e.g., 5x to 20x) while maintaining acceptable levels of geometric accuracy and semantic meaning.

## 1.2 Problem Statement

> All figures in this thesis were produced by running the project codebase scripts (`src/experiments/generate_plots.py`, `src/experiments/generate_dataset_plots.py`) on the real preprocessed GeoLife GPS dataset stored in `data/processed/trajectories.pkl`. No synthetic data was used for evaluation.



Traditional trajectory simplification algorithms, such as Douglas-Peucker (DP) and Sliding Window (SW), primarily focus on geometric error minimization. While these methods are effective for regularly sampled, noise-free trajectories, they face significant challenges when dealing with:

1. **Irregular Sampling**: Real-world GPS trajectories often exhibit irregular sampling intervals due to device limitations, signal loss, or power-saving modes. Points in sparse regions may represent unique information that should be preserved.

2. **Noise**: GPS measurements inherently contain noise, which can cause traditional geometric methods to make suboptimal decisions about which points to retain.

3. **Semantic Importance**: Trajectories contain semantically important features such as:
   - **Turns**: Direction changes that indicate route decisions or navigation events
   - **Stops**: Periods of low or zero speed that may represent significant events (e.g., waiting at traffic lights, visiting locations)
   - **Speed Changes**: Acceleration and deceleration patterns that reflect behavior changes

These semantic features may not align with geometric error minimization, leading to their loss during simplification.

## 1.3 Research Objectives

This research aims to develop a novel trajectory simplification algorithm that:

1. Preserves important semantic features (turns, stops, speed changes) under fixed compression budgets
2. Handles irregular sampling patterns effectively
3. Is robust to noise in GPS measurements
4. Achieves competitive or superior performance compared to baseline methods across multiple evaluation metrics

## 1.4 Contributions

The main contributions of this work are:

1. **Novel Algorithm**: A turn/speed/stop-aware trajectory simplification method that combines geometric and semantic importance scoring under a fixed compression budget, directly addressing the concrete weakness of instability under irregular sampling and noise.

2. **RL-Inspired Baseline**: A training-free Greedy Policy simplification algorithm that mirrors the per-point decision structure of Wang et al. (2021) reinforcement learning approach, enabling fair comparison with learning-based methods without requiring offline training.

3. **Comprehensive Evaluation**: Extensive experimental evaluation on real-world GeoLife GPS data comparing **7 algorithms** (six baselines plus the proposed method) across 4 compression ratios and **10+ evaluation metrics** spanning geometric, time-synchronised, semantic, and efficiency dimensions.

4. **Dataset Characterisation**: Comprehensive analysis of 5,716 GeoLife trajectories, quantifying sampling irregularity (mean CV = 5.96), stop density (34.2% of points), and turn frequency (32.4% of points) — properties that directly motivate the proposed approach.

5. **Scalability Analysis**: Runtime, memory, and throughput measurements for all algorithms, demonstrating that the proposed method is 23× faster than Douglas-Peucker and 129× faster than Sliding Window on long real-world trajectories.

6. **Reproducible Pipeline**: Preprocessing exports both **pickle** and **CSV** (`trajectories_points.csv`, `trajectories_index.csv`); experiments and figures are driven by documented CLI scripts; configuration constants live in `src/utils/config.py` (mirrored in `config/experiment_config.yaml`).

## 1.5 Thesis Structure

This thesis is organized as follows:

- **Chapter 2**: Related Work - Review of existing trajectory simplification algorithms and evaluation metrics
- **Chapter 3**: Dataset Analysis - Analysis of the GeoLife GPS dataset, including sampling irregularity, noise characteristics, and feature distribution
- **Chapter 4**: Methodology - Detailed description of baseline algorithms and the proposed method
- **Chapter 5**: Evaluation Metrics - Description of geometric and semantic preservation metrics
- **Chapter 6**: Experiments - Experimental setup, results, and analysis
- **Chapter 7**: Results and Discussion - Detailed analysis of results, trade-offs, and limitations
- **Chapter 8**: Conclusion and Future Work - Summary of contributions and directions for future research


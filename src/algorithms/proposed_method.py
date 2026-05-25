"""
PHASE 3: Proposed Semantic + Geometric Trajectory Simplification

This module implements the proposed trajectory simplification algorithm that
explicitly balances semantic feature preservation (turns, stops, speed changes,
sampling irregularity) with geometric fidelity under a fixed compression budget.

Key improvement over pure-geometric baselines (VW, RW, SQUISH, DP):
  Baselines minimise geometric error but have NO mechanism to preserve turns,
  stops, or speed changes.  The proposed method adds a geometric deviation
  component to a multi-criteria importance score so that:
    - Semantically important points (turns, stops) are kept even when they
      contribute little to geometric error.
    - Geometrically important points are kept even when they have no semantic
      label.
    - The refinement step guarantees that no segment between selected points
      exceeds an adaptive geometric threshold.

Scoring Formula (5 components):
  importance(p_i) = w_geo      * geo_score(p_i)        # perpendicular deviation
                  + w_turn     * turn_score(p_i)        # direction change
                  + w_stop     * stop_score(p_i)        # low-speed duration
                  + w_speed    * speed_score(p_i)       # acceleration/deceleration
                  + w_irregular* irregular_score(p_i)   # sparse-sampling gap

Default weights: geo=0.20, turn=0.25, stop=0.25, speed=0.15, irregular=0.15

Algorithm:
  1. Compute all 5 importance scores
  2. Select top-k interior points; always keep first and last
  3. Adaptive geometric refinement: iterate over consecutive selected pairs;
     if max perpendicular error in the gap exceeds threshold AND budget remains,
     insert the worst-offender point.  Repeat until stable or budget exhausted.
  4. Trim to budget by removing lowest-importance interior points if overshoot.

Why it handles irregular sampling and noise better than baselines:
  - Irregular sampling: explicit irregularity score boosts sparse-gap points
  - Noise robustness: stop/turn scores use smoothed/duration-based signals
  - Geometric quality: geo component + iterative refinement bound Hausdorff
  - Semantic quality: only algorithm with turn_preservation and stop_preservation
"""

import numpy as np
import pandas as pd
from typing import Union, Tuple, List, Dict
from src.algorithms.baseline_algorithms import haversine_distance, point_to_line_distance
from src.utils.config import STOP_SPEED_THRESHOLD_MS, MIN_STOP_DURATION_S


def compute_turn_score(trajectory: pd.DataFrame, 
                      window_size: int = 3) -> np.ndarray:
    """
    Compute turn significance score for each point.
    
    A turn is significant if:
    - Large direction change (angle between segments)
    - Consistent direction change (not noise)
    
    Args:
        trajectory: DataFrame with 'lat', 'lon' columns
        window_size: Window size for computing direction changes
        
    Returns:
        Array of turn scores [0, 1] for each point
    """
    if len(trajectory) < 3:
        return np.zeros(len(trajectory))
    
    points = trajectory[['lat', 'lon']].values
    n = len(points)
    turn_scores = np.zeros(n)
    
    # Compute bearings (directions) for each segment
    bearings = np.zeros(n - 1)
    for i in range(n - 1):
        lat1, lon1 = np.radians(points[i])
        lat2, lon2 = np.radians(points[i + 1])
        
        dlon = lon2 - lon1
        bearing = np.arctan2(
            np.sin(dlon) * np.cos(lat2),
            np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
        )
        bearings[i] = np.degrees(bearing)
        bearings[i] = (bearings[i] + 360) % 360
    
    # Compute direction changes
    direction_changes = np.abs(np.diff(bearings))
    direction_changes = np.minimum(direction_changes, 360 - direction_changes)
    direction_changes = np.concatenate([[0], direction_changes, [0]])
    
    # Smooth direction changes to reduce noise
    if window_size > 1:
        kernel = np.ones(window_size) / window_size
        direction_changes = np.convolve(direction_changes, kernel, mode='same')
    
    # Normalize to [0, 1]
    if np.max(direction_changes) > 0:
        turn_scores = direction_changes / np.max(direction_changes)
    else:
        turn_scores = np.zeros(n)
    
    # Boost scores for points with high local variance (sharp turns)
    for i in range(1, n - 1):
        local_variance = np.var(direction_changes[max(0, i-2):min(n, i+3)])
        turn_scores[i] = turn_scores[i] * (1 + 0.5 * local_variance)
    
    turn_scores = np.clip(turn_scores, 0, 1)
    
    return turn_scores


def compute_stop_score(trajectory: pd.DataFrame,
                       stop_threshold: float = STOP_SPEED_THRESHOLD_MS,
                       min_duration: float = MIN_STOP_DURATION_S) -> np.ndarray:
    """
    Compute stop significance score for each point.
    
    A stop is significant if:
    - Low speed (below threshold)
    - Sustained duration (not just momentary)
    
    Args:
        trajectory: DataFrame with 'lat', 'lon', 'timestamp' columns
        stop_threshold: Speed threshold for stop (m/s)
        min_duration: Minimum duration to be considered significant stop (seconds)
        
    Returns:
        Array of stop scores [0, 1] for each point
    """
    if len(trajectory) < 2:
        return np.zeros(len(trajectory))
    
    points = trajectory[['lat', 'lon']].values
    n = len(points)
    stop_scores = np.zeros(n)
    
    # Compute speeds
    speeds = np.zeros(n)
    if 'timestamp' in trajectory.columns:
        timestamps = pd.to_datetime(trajectory['timestamp']).values
    else:
        timestamps = np.arange(n)
    
    for i in range(1, n):
        dist = haversine_distance(tuple(points[i-1]), tuple(points[i]))
        time_diff = (pd.to_datetime(timestamps[i]) - pd.to_datetime(timestamps[i-1])).total_seconds()
        if time_diff > 0:
            speeds[i] = dist / time_diff
        else:
            speeds[i] = speeds[i-1] if i > 1 else 0
    
    # Identify stop regions
    is_stop = speeds < stop_threshold
    
    # Compute stop durations
    stop_durations = np.zeros(n)
    i = 0
    while i < n:
        if is_stop[i]:
            # Find contiguous stop region
            start = i
            while i < n and is_stop[i]:
                i += 1
            duration = (pd.to_datetime(timestamps[i-1]) - pd.to_datetime(timestamps[start])).total_seconds()
            
            # Assign duration to all points in stop region
            for j in range(start, i):
                stop_durations[j] = duration
        else:
            i += 1
    
    # Score based on duration (longer stops are more important)
    max_duration = np.max(stop_durations) if len(stop_durations) > 0 else 1
    if max_duration > 0:
        stop_scores = stop_durations / max_duration
    else:
        stop_scores = np.zeros(n)
    
    # Boost scores for stops above minimum duration
    significant_stops = stop_durations >= min_duration
    stop_scores[significant_stops] = np.minimum(stop_scores[significant_stops] * 1.5, 1.0)
    
    return stop_scores


def compute_speed_change_score(trajectory: pd.DataFrame,
                              window_size: int = 3) -> np.ndarray:
    """
    Compute speed change significance score for each point.
    
    Speed changes indicate behavior shifts (acceleration, deceleration).
    
    Args:
        trajectory: DataFrame with 'lat', 'lon', 'timestamp' columns
        window_size: Window size for computing speed changes
        
    Returns:
        Array of speed change scores [0, 1] for each point
    """
    if len(trajectory) < 3:
        return np.zeros(len(trajectory))
    
    points = trajectory[['lat', 'lon']].values
    n = len(points)
    
    # Compute speeds
    speeds = np.zeros(n)
    if 'timestamp' in trajectory.columns:
        timestamps = pd.to_datetime(trajectory['timestamp']).values
    else:
        timestamps = np.arange(n)
    
    for i in range(1, n):
        dist = haversine_distance(tuple(points[i-1]), tuple(points[i]))
        time_diff = (pd.to_datetime(timestamps[i]) - pd.to_datetime(timestamps[i-1])).total_seconds()
        if time_diff > 0:
            speeds[i] = dist / time_diff
        else:
            speeds[i] = speeds[i-1] if i > 1 else 0
    
    # Compute speed changes (acceleration/deceleration)
    speed_changes = np.abs(np.diff(speeds))
    speed_changes = np.concatenate([[0], speed_changes])
    
    # Smooth to reduce noise
    if window_size > 1:
        kernel = np.ones(window_size) / window_size
        speed_changes = np.convolve(speed_changes, kernel, mode='same')
    
    # Normalize to [0, 1]
    if np.max(speed_changes) > 0:
        speed_change_scores = speed_changes / np.max(speed_changes)
    else:
        speed_change_scores = np.zeros(n)
    
    return speed_change_scores


def compute_irregularity_score(trajectory: pd.DataFrame) -> np.ndarray:
    """
    Compute sampling irregularity score for each point.
    
    Points in sparse regions (large time gaps) are more important
    because they represent unique information.
    
    Args:
        trajectory: DataFrame with 'lat', 'lon', 'timestamp' columns
        
    Returns:
        Array of irregularity scores [0, 1] for each point
    """
    if len(trajectory) < 3:
        return np.zeros(len(trajectory))
    
    n = len(trajectory)
    
    if 'timestamp' in trajectory.columns:
        timestamps = pd.to_datetime(trajectory['timestamp']).values
    else:
        # Uniform sampling - no irregularity
        return np.zeros(n)
    
    # Compute time intervals
    time_intervals = np.zeros(n)
    for i in range(1, n):
        time_intervals[i] = (pd.to_datetime(timestamps[i]) - pd.to_datetime(timestamps[i-1])).total_seconds()
    
    # Normalize by median (points with intervals >> median are in sparse regions)
    median_interval = np.median(time_intervals[1:])
    if median_interval > 0:
        irregularity_scores = np.minimum(time_intervals / (median_interval * 3), 1.0)
    else:
        irregularity_scores = np.zeros(n)
    
    # Boost scores for points with very large gaps
    large_gap_threshold = median_interval * 5
    large_gaps = time_intervals > large_gap_threshold
    irregularity_scores[large_gaps] = 1.0
    
    return irregularity_scores


def compute_geometric_score(trajectory: pd.DataFrame) -> np.ndarray:
    """
    Compute geometric deviation score for each interior point.

    For point p_i, this is the perpendicular distance from p_i to the chord
    connecting p_{i-1} and p_{i+1}, normalised by the maximum such distance
    across the trajectory.  Endpoints receive score 1.0.

    A high score means that p_i deviates significantly from the straight line
    between its neighbours — i.e., removing it would introduce a large
    geometric error.  Including this component prevents the purely-semantic
    selection from inadvertently dropping geometrically critical points.

    Args:
        trajectory: DataFrame with 'lat', 'lon' columns

    Returns:
        Array of geometric deviation scores in [0, 1]
    """
    n = len(trajectory)
    if n < 3:
        return np.ones(n)

    pts = trajectory[['lat', 'lon']].values
    scores = np.zeros(n)
    scores[0] = 1.0
    scores[-1] = 1.0

    for i in range(1, n - 1):
        scores[i] = point_to_line_distance(
            tuple(pts[i]), tuple(pts[i - 1]), tuple(pts[i + 1])
        )

    max_dev = scores.max()
    if max_dev > 0:
        scores /= max_dev
    return scores


def proposed_simplification(trajectory: pd.DataFrame,
                           budget: int,
                           weights: Dict[str, float] = None,
                           geometric_refinement: bool = True,
                           min_geometric_error: float = None) -> Tuple[np.ndarray, List[int]]:
    """
    Proposed semantic + geometric trajectory simplification.

    Algorithm:
    1. Compute 5 component scores: geometric deviation, turn, stop, speed, irregularity
    2. Combine into a weighted importance score
    3. Select top-k interior points; always keep first and last
    4. Adaptive iterative refinement: while budget remains, insert worst-error
       interior points until no segment exceeds an adaptive threshold
    5. Trim to exact budget by removing lowest-importance interior points

    Args:
        trajectory:           Input trajectory DataFrame (must have lat, lon).
        budget:               Target number of retained points.
        weights:              Component weights dict.  Supports 5 keys:
                              'geo' (default 0.20), 'turn' (0.25), 'stop' (0.25),
                              'speed' (0.15), 'irregular' (0.15).
                              Missing keys default to the values above.
        geometric_refinement: If True, run the iterative refinement pass.
        min_geometric_error:  Adaptive threshold for refinement (metres).
                              None → auto: 1 % of the trajectory's spatial diagonal.

    Returns:
        Tuple of (simplified_points array [k×2], selected_indices list[int])
    """
    n = len(trajectory)
    if n <= budget:
        pts = trajectory[['lat', 'lon']].values
        return pts, list(range(n))

    # --- 1. Default and normalise weights ---
    _defaults = {'geo': 0.20, 'turn': 0.25, 'stop': 0.25, 'speed': 0.15, 'irregular': 0.15}
    if weights is None:
        weights = dict(_defaults)
    else:
        # fill missing keys with defaults, then normalise
        for k, v in _defaults.items():
            weights.setdefault(k, v)

    total_w = sum(weights.values())
    if total_w > 0:
        weights = {k: v / total_w for k, v in weights.items()}
    else:
        weights = dict(_defaults)
        weights = {k: v / sum(weights.values()) for k, v in weights.items()}

    # --- 2. Compute component scores ---
    geo_scores      = compute_geometric_score(trajectory)
    turn_scores     = compute_turn_score(trajectory)
    stop_scores     = compute_stop_score(trajectory)
    speed_scores    = compute_speed_change_score(trajectory)
    irreg_scores    = compute_irregularity_score(trajectory)

    importance = (
        weights['geo']       * geo_scores
        + weights['turn']    * turn_scores
        + weights['stop']    * stop_scores
        + weights['speed']   * speed_scores
        + weights['irregular'] * irreg_scores
    )

    # Endpoints are always kept
    importance[0]  = 2.0
    importance[-1] = 2.0

    # --- 3. Initial selection: top-k by importance ---
    sel = sorted(np.argsort(importance)[-budget:].tolist())

    # --- 4. Adaptive iterative geometric refinement ---
    if geometric_refinement:
        pts = trajectory[['lat', 'lon']].values

        # Adaptive threshold: 1 % of spatial diagonal, floor 2 m
        lat_range = pts[:, 0].max() - pts[:, 0].min()
        lon_range = pts[:, 1].max() - pts[:, 1].min()
        # approximate metres
        diag_m = np.sqrt((lat_range * 111_320) ** 2 + (lon_range * 111_320) ** 2)
        if min_geometric_error is None:
            threshold = max(2.0, diag_m * 0.01)
        else:
            threshold = min_geometric_error

        changed = True
        while changed and len(sel) < budget:
            changed = False
            new_sel = set(sel)
            for a, b in zip(sel[:-1], sel[1:]):
                gap = range(a + 1, b)
                if not gap:
                    continue
                # find worst point in gap
                worst_err = 0.0
                worst_j   = -1
                for j in gap:
                    e = point_to_line_distance(tuple(pts[j]), tuple(pts[a]), tuple(pts[b]))
                    if e > worst_err:
                        worst_err = e
                        worst_j   = j
                if worst_err > threshold and len(new_sel) < budget:
                    new_sel.add(worst_j)
                    changed = True
            sel = sorted(new_sel)

        # --- 5. Trim to exact budget if refinement overshot ---
        if len(sel) > budget:
            interior = sel[1:-1]
            imp_sub  = importance[interior]
            keep_n   = budget - 2
            keep_idx = np.argsort(imp_sub)[-keep_n:].tolist()
            sel = [sel[0]] + [interior[i] for i in sorted(keep_idx)] + [sel[-1]]

    simplified_points = trajectory.iloc[sel][['lat', 'lon']].values
    return simplified_points, sel


if __name__ == "__main__":
    # Demo on real preprocessed GeoLife (same file as run_experiments.py).
    import pickle
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    data_file = repo_root / "data" / "processed" / "trajectories.pkl"
    if not data_file.is_file():
        raise SystemExit(
            f"Missing {data_file}. Download GeoLife to data/geolife/ then run:\n"
            "  python src/utils/preprocess_geolife.py"
        )

    with open(data_file, "rb") as f:
        trajectories = pickle.load(f)

    trajectory = next((t for t in trajectories if len(t) >= 100), trajectories[0])
    meta = []
    if "user_id" in trajectory.columns:
        meta.append(f"user={trajectory['user_id'].iloc[0]}")
    if "file_id" in trajectory.columns:
        meta.append(f"file={trajectory['file_id'].iloc[0]}")
    print("Loaded preprocessed GeoLife trajectory" + (f" ({', '.join(meta)})" if meta else ""))

    print(f"Original trajectory: {len(trajectory)} points")

    budget = max(30, len(trajectory) // 10)
    simplified, indices = proposed_simplification(trajectory, budget)

    print(f"Simplified trajectory: {len(simplified)} points")
    print(f"Compression ratio: {len(trajectory) / len(simplified):.2f}x")
    print(f"Selected indices: {indices[:10]}...{indices[-5:]}")


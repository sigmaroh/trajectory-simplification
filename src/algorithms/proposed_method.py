"""
PHASE 3: Proposed Turn/Speed/Stop-Aware Trajectory Simplification

This module implements a novel trajectory simplification algorithm that preserves
important trajectory features (turns, stops, speed changes) under a fixed compression budget.

Mathematical Intuition:
- Traditional methods (RDP, Sliding Window) focus on geometric error
- Real trajectories have semantic importance: turns indicate direction changes,
  stops indicate significant events, speed changes indicate behavior shifts
- Our method assigns importance scores to points based on:
  1. Turn significance (direction change)
  2. Stop significance (low speed duration)
  3. Speed change significance (acceleration/deceleration)
  4. Sampling irregularity (points in sparse regions are more important)

Scoring Formula:
  importance(p_i) = w_turn * turn_score(p_i) + 
                     w_stop * stop_score(p_i) + 
                     w_speed * speed_change_score(p_i) + 
                     w_irregular * irregularity_score(p_i)

Algorithm:
  1. Compute importance scores for all points
  2. Select top-k points by importance
  3. Refine selection to ensure geometric quality
  4. Return simplified trajectory

Why it handles irregular sampling and noise better:
- Irregular sampling: Points in sparse regions get higher importance
- Noise: Speed/direction changes are more robust than pure geometric distance
- Stops: Explicitly preserved even if geometrically close to neighbors
- Turns: Preserved even if within error threshold of straight line
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


def proposed_simplification(trajectory: pd.DataFrame,
                           budget: int,
                           weights: Dict[str, float] = None,
                           geometric_refinement: bool = True,
                           min_geometric_error: float = 5.0) -> Tuple[np.ndarray, List[int]]:
    """
    Proposed turn/speed/stop-aware trajectory simplification.
    
    Algorithm:
    1. Compute importance scores for all points
    2. Select top-k points by importance
    3. Optionally refine to ensure geometric quality
    4. Return simplified trajectory
    
    Args:
        trajectory: Input trajectory DataFrame
        budget: Target number of points (compression budget)
        weights: Dictionary with weights for different components:
                 {'turn': 0.3, 'stop': 0.3, 'speed': 0.2, 'irregular': 0.2}
        geometric_refinement: Whether to refine selection based on geometric error
        min_geometric_error: Minimum geometric error threshold for refinement
        
    Returns:
        Tuple of (simplified_points, selected_indices)
    """
    if len(trajectory) <= budget:
        points = trajectory[['lat', 'lon']].values
        return points, list(range(len(trajectory)))
    
    # Default weights
    if weights is None:
        weights = {
            'turn': 0.3,
            'stop': 0.3,
            'speed': 0.2,
            'irregular': 0.2
        }
    
    # Normalize weights (if all weights are 0, fall back to uniform importance)
    total_weight = sum(weights.values())
    if total_weight > 0:
        weights = {k: v / total_weight for k, v in weights.items()}
    else:
        # geo_only or similar: no semantic scoring, rely on geometric refinement
        weights = {k: 0.0 for k in weights}
    
    # Compute component scores
    turn_scores = compute_turn_score(trajectory)
    stop_scores = compute_stop_score(trajectory)
    speed_scores = compute_speed_change_score(trajectory)
    irregular_scores = compute_irregularity_score(trajectory)
    
    # Combine scores
    importance_scores = (
        weights['turn'] * turn_scores +
        weights['stop'] * stop_scores +
        weights['speed'] * speed_scores +
        weights['irregular'] * irregular_scores
    )
    
    # Always include first and last points
    importance_scores[0] = 1.0
    importance_scores[-1] = 1.0
    
    # Select top-k points
    top_k = budget
    selected_indices = np.argsort(importance_scores)[-top_k:]
    selected_indices = sorted(selected_indices)
    
    # Geometric refinement: ensure no large geometric errors
    if geometric_refinement:
        points = trajectory[['lat', 'lon']].values
        refined_indices = [selected_indices[0]]
        
        for i in range(1, len(selected_indices)):
            prev_idx = refined_indices[-1]
            curr_idx = selected_indices[i]
            
            # Check maximum error in segment
            max_error = 0
            for j in range(prev_idx + 1, curr_idx):
                error = point_to_line_distance(
                    tuple(points[j]),
                    tuple(points[prev_idx]),
                    tuple(points[curr_idx])
                )
                max_error = max(max_error, error)
            
            # If error is acceptable, keep point
            # Otherwise, add intermediate point if needed
            if max_error <= min_geometric_error or curr_idx == selected_indices[-1]:
                refined_indices.append(curr_idx)
            else:
                # Find point with maximum error in segment
                max_error_idx = prev_idx + 1
                max_error_val = 0
                for j in range(prev_idx + 1, curr_idx):
                    error = point_to_line_distance(
                        tuple(points[j]),
                        tuple(points[prev_idx]),
                        tuple(points[curr_idx])
                    )
                    if error > max_error_val:
                        max_error_val = error
                        max_error_idx = j
                
                # Add point with max error if budget allows
                if len(refined_indices) < budget - 1:
                    refined_indices.append(max_error_idx)
                refined_indices.append(curr_idx)
        
        selected_indices = sorted(list(set(refined_indices)))
        
        # If we exceeded budget, remove least important points (except first/last)
        if len(selected_indices) > budget:
            middle_indices = selected_indices[1:-1]
            importance_subset = importance_scores[middle_indices]
            keep_count = budget - 2
            top_middle = np.argsort(importance_subset)[-keep_count:]
            selected_indices = [selected_indices[0]] + [middle_indices[i] for i in top_middle] + [selected_indices[-1]]
            selected_indices = sorted(selected_indices)
    
    # Extract simplified trajectory
    simplified_points = trajectory.iloc[selected_indices][['lat', 'lon']].values
    
    return simplified_points, selected_indices


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


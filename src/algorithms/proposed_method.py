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
from scipy import stats
from src.algorithms.baseline_algorithms import haversine_distance, point_to_line_distance, _haversine_batch, _point_to_line_distance_batch


# ---------------------------------------------------------------------------
# Private vectorised helpers
# ---------------------------------------------------------------------------

def _timestamp_diffs_s(timestamps) -> np.ndarray:
    """
    Convert a timestamps array to an (n-1,) array of time differences in seconds.
    Works for numpy datetime64 arrays, pandas DatetimeIndex, and integer arrays
    (integers are treated as nanoseconds since epoch, matching pd.to_datetime behaviour).
    """
    ns = np.asarray(pd.to_datetime(timestamps), dtype='datetime64[ns]').astype(np.int64)
    return np.diff(ns) * 1e-9  # nanoseconds → seconds


def _compute_speeds(points: np.ndarray, timestamps) -> np.ndarray:
    """
    Compute per-point speed (m/s) using vectorised haversine + timestamp diffs.
    Returns array of length n; speed[0] = 0.
    Propagates previous speed for zero-duration intervals (matches original behaviour).
    """
    n = len(points)
    if n < 2:
        return np.zeros(n)

    dists = _haversine_batch(
        points[:-1, 0], points[:-1, 1],
        points[1:,  0], points[1:,  1],
    )
    time_diffs = _timestamp_diffs_s(timestamps)

    speeds = np.zeros(n)
    valid = time_diffs > 0
    speeds[1:] = np.where(valid, dists / np.where(valid, time_diffs, 1.0), 0.0)
    for i in range(1, n):
        if time_diffs[i - 1] <= 0:
            speeds[i] = speeds[i - 1] if i > 1 else 0.0
    return speeds


# ---------------------------------------------------------------------------
# Public scoring functions
# ---------------------------------------------------------------------------

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

    # Vectorised bearing computation (replaces per-segment Python loop)
    lat1 = np.radians(points[:-1, 0]); lon1 = np.radians(points[:-1, 1])
    lat2 = np.radians(points[1:,  0]); lon2 = np.radians(points[1:,  1])
    dlon = lon2 - lon1
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    bearings = (np.degrees(np.arctan2(y, x)) + 360.0) % 360.0  # (n-1,)

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
        local_variance = np.var(direction_changes[max(0, i - 2):min(n, i + 3)])
        turn_scores[i] = turn_scores[i] * (1 + 0.5 * local_variance)
    
    turn_scores = np.clip(turn_scores, 0, 1)
    
    return turn_scores


def compute_stop_score(trajectory: pd.DataFrame,
                       stop_threshold: float = 1.0,
                       min_duration: float = 30.0) -> np.ndarray:
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

    if 'timestamp' in trajectory.columns:
        timestamps = pd.to_datetime(trajectory['timestamp']).values
    else:
        timestamps = np.arange(n)

    # Vectorised speed computation
    speeds = _compute_speeds(points, timestamps)

    # Pre-compute timestamp ns array for fast duration queries
    ts_ns = np.asarray(pd.to_datetime(timestamps), dtype='datetime64[ns]').astype(np.int64)

    # Identify stop regions using numpy diff (replaces nested Python loop)
    is_stop = speeds < stop_threshold
    changes = np.diff(np.concatenate([[False], is_stop, [False]]).astype(np.int8))
    starts = np.where(changes == 1)[0]
    ends   = np.where(changes == -1)[0]   # exclusive end indices

    stop_durations = np.zeros(n)
    for s, e in zip(starts, ends):
        duration = (ts_ns[e - 1] - ts_ns[s]) * 1e-9  # seconds
        stop_durations[s:e] = duration

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

    if 'timestamp' in trajectory.columns:
        timestamps = pd.to_datetime(trajectory['timestamp']).values
    else:
        timestamps = np.arange(n)

    # Vectorised speed computation
    speeds = _compute_speeds(points, timestamps)
    
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
    
    if 'timestamp' not in trajectory.columns:
        # Uniform sampling - no irregularity
        return np.zeros(n)

    timestamps = pd.to_datetime(trajectory['timestamp']).values

    # Vectorised timestamp differences (replaces per-point pd.to_datetime loop)
    time_intervals = np.zeros(n)
    time_intervals[1:] = _timestamp_diffs_s(timestamps)
    
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
    
    # Normalize weights
    total_weight = sum(weights.values())
    weights = {k: v / total_weight for k, v in weights.items()}
    
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
            
            # Vectorised: check maximum error in segment prev_idx → curr_idx
            interior = points[prev_idx + 1:curr_idx]
            if len(interior) > 0:
                dists = _point_to_line_distance_batch(
                    interior,
                    tuple(points[prev_idx]),
                    tuple(points[curr_idx]),
                )
                max_error = float(dists.max())
                max_error_idx = prev_idx + 1 + int(np.argmax(dists))
            else:
                max_error = 0.0
                max_error_idx = prev_idx + 1
            
            # If error is acceptable, keep point
            # Otherwise, add intermediate point if needed
            if max_error <= min_geometric_error or curr_idx == selected_indices[-1]:
                refined_indices.append(curr_idx)
            else:
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
    # Example usage
    import pandas as pd
    
    # Create sample trajectory with turns and stops
    n = 200
    trajectory = pd.DataFrame({
        'lat': np.concatenate([
            np.linspace(0, 1, n//4),
            np.linspace(1, 1, n//4),  # Stop region
            np.linspace(1, 2, n//4),
            np.linspace(2, 2, n//4)   # Another stop
        ]) + np.random.normal(0, 0.01, n),
        'lon': np.concatenate([
            np.linspace(0, 0, n//4),
            np.linspace(0, 1, n//4),  # Turn
            np.linspace(1, 1, n//4),
            np.linspace(1, 2, n//4)   # Turn
        ]) + np.random.normal(0, 0.01, n),
        'timestamp': pd.date_range('2023-01-01', periods=n, freq='30s')
    })
    
    print(f"Original trajectory: {len(trajectory)} points")
    
    # Simplify with proposed method
    budget = 30
    simplified, indices = proposed_simplification(trajectory, budget)
    
    print(f"Simplified trajectory: {len(simplified)} points")
    print(f"Compression ratio: {len(trajectory) / len(simplified):.2f}x")
    print(f"Selected indices: {indices[:10]}...{indices[-5:]}")

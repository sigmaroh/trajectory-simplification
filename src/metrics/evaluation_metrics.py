"""
PHASE 4: Evaluation Metrics for Trajectory Simplification

This module implements various metrics to evaluate the quality of simplified trajectories:

1. Hausdorff Distance: Maximum distance between trajectories
2. Average Point-to-Trajectory Error: Mean distance from original points to simplified trajectory
3. Frechet Distance: Similarity measure considering order of points
4. Turn Preservation Metric: How well turns are preserved
5. Stop Preservation Metric: How well stops are preserved
6. PED (Perpendicular Euclidean Distance): Mean perpendicular error to simplified segments
7. DAD (Direction-Aware Distance): Mean heading deviation
8. SED (Synchronized Euclidean Distance): Mean time-synchronized spatial error
9. SAD (Speed-Aware Distance): Mean speed difference under time synchronization
10. ISSD (Integrated Synchronized Spatial Distance): Time-integrated synchronized error

Formulas:
- Hausdorff: H(A,B) = max(h(A,B), h(B,A)) where h(A,B) = max_{a in A} min_{b in B} d(a,b)
- Average PTE: (1/n) * sum_{i=1}^n min_{p in S} d(original_i, p)
- Frechet: Minimum leash length needed to walk both trajectories simultaneously
- Turn Preservation: Ratio of preserved turns
- Stop Preservation: Ratio of preserved stops
"""

import numpy as np
import pandas as pd
from typing import Union, Tuple, List, Dict
from scipy.spatial.distance import cdist
from src.algorithms.baseline_algorithms import haversine_distance, point_to_line_distance
from src.utils.config import STOP_SPEED_THRESHOLD_MS, TURN_THRESHOLD_DEG, MIN_STOP_DURATION_S


def hausdorff_distance(original: np.ndarray, simplified: np.ndarray) -> float:
    """
    Compute Hausdorff distance between two trajectories.
    
    Hausdorff distance is the maximum distance from any point in one trajectory
    to the nearest point in the other trajectory.
    
    Formula:
        H(A,B) = max(h(A,B), h(B,A))
        where h(A,B) = max_{a in A} min_{b in B} d(a,b)
    
    Args:
        original: Original trajectory points (N x 2) with (lat, lon)
        simplified: Simplified trajectory points (M x 2) with (lat, lon)
        
    Returns:
        Hausdorff distance in meters
    """
    if len(original) == 0 or len(simplified) == 0:
        return float('inf')
    
    # Convert to approximate metric coordinates for distance calculation
    # For small distances, we can use simple approximation
    # For better accuracy, we use Haversine for each pair
    
    # Compute distance from each original point to nearest simplified point
    def min_dist_to_trajectory(point, trajectory):
        """Find minimum distance from point to any point in trajectory."""
        min_dist = float('inf')
        for traj_point in trajectory:
            dist = haversine_distance(tuple(point), tuple(traj_point))
            min_dist = min(min_dist, dist)
        return min_dist
    
    # h(original, simplified)
    h_orig_simpl = 0
    for orig_point in original:
        dist = min_dist_to_trajectory(orig_point, simplified)
        h_orig_simpl = max(h_orig_simpl, dist)
    
    # h(simplified, original)
    h_simpl_orig = 0
    for simpl_point in simplified:
        dist = min_dist_to_trajectory(simpl_point, original)
        h_simpl_orig = max(h_simpl_orig, dist)
    
    # Hausdorff distance
    hausdorff = max(h_orig_simpl, h_simpl_orig)
    
    return hausdorff


def average_point_to_trajectory_error(original: np.ndarray, 
                                      simplified: np.ndarray) -> float:
    """
    Compute average point-to-trajectory error.
    
    For each point in the original trajectory, find the minimum distance
    to any point or segment in the simplified trajectory, then average.
    
    Formula:
        APTE = (1/n) * sum_{i=1}^n min_{p in S} d(original_i, p)
    
    Args:
        original: Original trajectory points (N x 2)
        simplified: Simplified trajectory points (M x 2)
        
    Returns:
        Average error in meters
    """
    if len(original) == 0 or len(simplified) == 0:
        return float('inf')
    
    errors = []
    
    for orig_point in original:
        min_dist = float('inf')
        
        # Check distance to points
        for simpl_point in simplified:
            dist = haversine_distance(tuple(orig_point), tuple(simpl_point))
            min_dist = min(min_dist, dist)
        
        # Check distance to line segments
        for i in range(len(simplified) - 1):
            dist = point_to_line_distance(
                tuple(orig_point),
                tuple(simplified[i]),
                tuple(simplified[i + 1])
            )
            min_dist = min(min_dist, dist)
        
        errors.append(min_dist)
    
    return np.mean(errors)


def frechet_distance(original: np.ndarray, simplified: np.ndarray) -> float:
    """
    Compute discrete Frechet distance between two trajectories.
    
    Frechet distance is the minimum leash length needed to walk both trajectories
    simultaneously, where one person walks along the original and another along
    the simplified trajectory.
    
    Algorithm: Dynamic programming
    - F(i,j) = max(d(orig[i], simpl[j]), 
                   min(F(i-1,j), F(i,j-1), F(i-1,j-1)))
    
    Args:
        original: Original trajectory points (N x 2)
        simplified: Simplified trajectory points (M x 2)
        
    Returns:
        Frechet distance in meters
    """
    if len(original) == 0 or len(simplified) == 0:
        return float('inf')
    
    n, m = len(original), len(simplified)
    
    # Precompute distance matrix
    dist_matrix = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            dist_matrix[i, j] = haversine_distance(
                tuple(original[i]),
                tuple(simplified[j])
            )
    
    # Dynamic programming table
    F = np.zeros((n, m))
    F[0, 0] = dist_matrix[0, 0]
    
    # Initialize first row
    for j in range(1, m):
        F[0, j] = max(F[0, j-1], dist_matrix[0, j])
    
    # Initialize first column
    for i in range(1, n):
        F[i, 0] = max(F[i-1, 0], dist_matrix[i, 0])
    
    # Fill table
    for i in range(1, n):
        for j in range(1, m):
            F[i, j] = max(
                dist_matrix[i, j],
                min(F[i-1, j], F[i, j-1], F[i-1, j-1])
            )
    
    return F[n-1, m-1]


def turn_preservation_metric(original: pd.DataFrame,
                            simplified: pd.DataFrame,
                            original_indices: List[int],
                            turn_threshold: float = TURN_THRESHOLD_DEG) -> Tuple[float, Dict]:
    """
    Compute turn preservation metric.
    
    Measures how well turns in the original trajectory are preserved
    in the simplified trajectory.
    
    Args:
        original: Original trajectory DataFrame
        simplified: Simplified trajectory DataFrame
        original_indices: Indices of original points that were kept
        turn_threshold: Direction change threshold for turn (degrees)
        
    Returns:
        Tuple of (preservation_ratio, metrics_dict)
    """
    from src.utils.geolife_loader import compute_trajectory_properties
    
    # Compute turns in original trajectory
    orig_props = compute_trajectory_properties(original)
    
    if 'direction_changes' not in orig_props:
        return 0.0, {}
    
    # Identify turns in original
    orig_turns = orig_props['direction_changes'] > turn_threshold
    orig_turn_indices = np.where(orig_turns)[0].tolist()
    
    if len(orig_turn_indices) == 0:
        return 1.0, {'original_turns': 0, 'preserved_turns': 0, 'preservation_ratio': 1.0}
    
    # Check which turns are preserved (within small window of selected points)
    preserved_turns = 0
    window_size = max(1, len(original) // len(simplified))
    
    for turn_idx in orig_turn_indices:
        # Check if there's a selected point near this turn
        for sel_idx in original_indices:
            if abs(sel_idx - turn_idx) <= window_size:
                preserved_turns += 1
                break
    
    preservation_ratio = preserved_turns / len(orig_turn_indices)
    
    metrics = {
        'original_turns': len(orig_turn_indices),
        'preserved_turns': preserved_turns,
        'preservation_ratio': preservation_ratio
    }
    
    return preservation_ratio, metrics


def stop_preservation_metric(original: pd.DataFrame,
                            simplified: pd.DataFrame,
                            original_indices: List[int],
                            stop_threshold: float = STOP_SPEED_THRESHOLD_MS,
                            min_duration: float = MIN_STOP_DURATION_S) -> Tuple[float, Dict]:
    """
    Compute stop preservation metric.
    
    Measures how well stops in the original trajectory are preserved
    in the simplified trajectory.
    
    Args:
        original: Original trajectory DataFrame
        simplified: Simplified trajectory DataFrame
        original_indices: Indices of original points that were kept
        stop_threshold: Speed threshold for stop (m/s)
        min_duration: Minimum duration for significant stop (seconds)
        
    Returns:
        Tuple of (preservation_ratio, metrics_dict)
    """
    from src.utils.geolife_loader import compute_trajectory_properties
    
    # Compute stops in original trajectory
    orig_props = compute_trajectory_properties(original)
    
    if 'speeds' not in orig_props:
        return 0.0, {}
    
    # Identify stop regions
    speeds = orig_props['speeds']
    is_stop = speeds < stop_threshold
    
    # Find contiguous stop regions
    stop_regions = []
    i = 0
    while i < len(is_stop):
        if is_stop[i]:
            start = i
            while i < len(is_stop) and is_stop[i]:
                i += 1
            end = i - 1
            
            # Check duration
            if 'timestamp' in original.columns:
                duration = (pd.to_datetime(original['timestamp'].iloc[end]) - 
                           pd.to_datetime(original['timestamp'].iloc[start])).total_seconds()
            else:
                duration = end - start  # Assume 1 second per point
            
            if duration >= min_duration:
                stop_regions.append((start, end))
        else:
            i += 1
    
    if len(stop_regions) == 0:
        return 1.0, {'original_stops': 0, 'preserved_stops': 0, 'preservation_ratio': 1.0}
    
    # Check which stops are preserved
    preserved_stops = 0
    window_size = max(1, len(original) // len(simplified))
    
    for start, end in stop_regions:
        # Check if there's a selected point in this stop region
        stop_center = (start + end) // 2
        for sel_idx in original_indices:
            if start <= sel_idx <= end or abs(sel_idx - stop_center) <= window_size:
                preserved_stops += 1
                break
    
    preservation_ratio = preserved_stops / len(stop_regions)
    
    metrics = {
        'original_stops': len(stop_regions),
        'preserved_stops': preserved_stops,
        'preservation_ratio': preservation_ratio
    }
    
    return preservation_ratio, metrics


def compression_ratio(original: np.ndarray, simplified: np.ndarray) -> float:
    """
    Compute compression ratio.
    
    Args:
        original: Original trajectory
        simplified: Simplified trajectory
        
    Returns:
        Compression ratio (original_size / simplified_size)
    """
    if len(simplified) == 0:
        return float('inf')
    return len(original) / len(simplified)


def extract_time_seconds(original: pd.DataFrame) -> np.ndarray:
    """Extract monotonic time values in seconds for synchronization."""
    if 'timestamp' in original.columns:
        ts = pd.to_datetime(original['timestamp'])
        # Relative seconds to improve numerical stability.
        time_sec = (ts - ts.iloc[0]).dt.total_seconds().to_numpy(dtype=float)
    else:
        time_sec = np.arange(len(original), dtype=float)
    return time_sec


def make_monotonic(values: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Ensure strictly increasing values for interpolation."""
    out = values.astype(float).copy()
    for i in range(1, len(out)):
        if out[i] <= out[i - 1]:
            out[i] = out[i - 1] + eps
    return out


def bearing_degrees(start: np.ndarray, end: np.ndarray) -> float:
    """Compute bearing from start(lat, lon) to end(lat, lon) in degrees [0, 360)."""
    lat1, lon1 = np.radians(start[0]), np.radians(start[1])
    lat2, lon2 = np.radians(end[0]), np.radians(end[1])
    dlon = lon2 - lon1
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    bearing = np.degrees(np.arctan2(y, x))
    return float((bearing + 360.0) % 360.0)


def angular_diff_deg(a: float, b: float) -> float:
    """Smallest absolute difference between two angles in degrees."""
    d = abs(a - b) % 360.0
    return float(min(d, 360.0 - d))


def synchronized_positions(original: pd.DataFrame,
                            simplified: np.ndarray,
                            original_indices: List[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Interpolate simplified trajectory positions at original timestamps.

    Returns:
        query_time_sec: Original time axis (N,)
        sync_points: Interpolated simplified points aligned to query_time_sec (N, 2)
        simplified_time_sec: Time axis used for simplified points (M,)
    """
    query_time_sec = extract_time_seconds(original)
    query_time_sec = make_monotonic(query_time_sec)

    if original_indices is not None and len(original_indices) == len(simplified):
        idx = np.clip(np.asarray(original_indices, dtype=int), 0, len(original) - 1)
        simplified_time_sec = query_time_sec[idx]
    else:
        simplified_time_sec = np.linspace(query_time_sec[0], query_time_sec[-1], len(simplified), dtype=float)

    simplified_time_sec = make_monotonic(simplified_time_sec)
    lat_interp = np.interp(query_time_sec, simplified_time_sec, simplified[:, 0])
    lon_interp = np.interp(query_time_sec, simplified_time_sec, simplified[:, 1])
    sync_points = np.column_stack([lat_interp, lon_interp])

    return query_time_sec, sync_points, simplified_time_sec


def perpendicular_euclidean_distance(original: np.ndarray, simplified: np.ndarray) -> float:
    """
    PED: Mean perpendicular distance from original points to simplified segments.
    """
    if len(original) == 0 or len(simplified) < 2:
        return float('inf')

    errors = []
    for point in original:
        min_dist = float('inf')
        for i in range(len(simplified) - 1):
            dist = point_to_line_distance(tuple(point), tuple(simplified[i]), tuple(simplified[i + 1]))
            min_dist = min(min_dist, dist)
        errors.append(min_dist)
    return float(np.mean(errors))


def synchronized_euclidean_distance(original: pd.DataFrame,
                                    simplified: np.ndarray,
                                    original_indices: List[int] = None) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    SED: Mean Euclidean distance between original and time-synchronized simplified points.
    """
    if len(original) == 0 or len(simplified) == 0:
        return float('inf'), np.array([]), np.array([]), np.empty((0, 2))

    original_points = original[['lat', 'lon']].to_numpy(dtype=float)
    query_time_sec, sync_points, _ = synchronized_positions(original, simplified, original_indices)
    dists = np.array(
        [haversine_distance(tuple(original_points[i]), tuple(sync_points[i])) for i in range(len(original_points))],
        dtype=float
    )
    return float(np.mean(dists)), dists, query_time_sec, sync_points


def direction_aware_distance(original: pd.DataFrame,
                             synchronized_points: np.ndarray) -> float:
    """
    DAD: Mean heading difference (degrees) between original and synchronized simplified trajectories.
    """
    original_points = original[['lat', 'lon']].to_numpy(dtype=float)
    if len(original_points) < 2 or len(synchronized_points) < 2:
        return 0.0

    heading_errors = []
    for i in range(len(original_points) - 1):
        if haversine_distance(tuple(original_points[i]), tuple(original_points[i + 1])) < 1e-6:
            continue
        if haversine_distance(tuple(synchronized_points[i]), tuple(synchronized_points[i + 1])) < 1e-6:
            continue

        b_orig = bearing_degrees(original_points[i], original_points[i + 1])
        b_sync = bearing_degrees(synchronized_points[i], synchronized_points[i + 1])
        heading_errors.append(angular_diff_deg(b_orig, b_sync))

    if not heading_errors:
        return 0.0
    return float(np.mean(heading_errors))


def speed_aware_distance(original: pd.DataFrame,
                         synchronized_points: np.ndarray,
                         query_time_sec: np.ndarray) -> float:
    """
    SAD: Mean absolute speed difference (m/s) between original and synchronized simplified trajectories.
    """
    original_points = original[['lat', 'lon']].to_numpy(dtype=float)
    if len(original_points) < 2 or len(synchronized_points) < 2 or len(query_time_sec) < 2:
        return 0.0

    speed_errors = []
    for i in range(len(original_points) - 1):
        dt = query_time_sec[i + 1] - query_time_sec[i]
        if dt <= 0:
            continue
        v_orig = haversine_distance(tuple(original_points[i]), tuple(original_points[i + 1])) / dt
        v_sync = haversine_distance(tuple(synchronized_points[i]), tuple(synchronized_points[i + 1])) / dt
        speed_errors.append(abs(v_orig - v_sync))

    if not speed_errors:
        return 0.0
    return float(np.mean(speed_errors))


def integrated_synchronized_spatial_distance(instantaneous_distances: np.ndarray,
                                             query_time_sec: np.ndarray) -> float:
    """
    ISSD: Time-integrated synchronized spatial distance (meter*second).
    """
    if len(instantaneous_distances) == 0:
        return 0.0
    if len(query_time_sec) != len(instantaneous_distances):
        return float(np.sum(instantaneous_distances))
    return float(np.trapezoid(instantaneous_distances, query_time_sec))


def compute_all_metrics(original: pd.DataFrame,
                       simplified: np.ndarray,
                       original_indices: List[int] = None) -> Dict:
    """
    Compute all evaluation metrics for a simplified trajectory.
    
    Args:
        original: Original trajectory DataFrame
        simplified: Simplified trajectory points (N x 2)
        original_indices: Indices of original points (if available)
        
    Returns:
        Dictionary with all metrics
    """
    original_points = original[['lat', 'lon']].values
    
    # Geometric metrics
    hausdorff = hausdorff_distance(original_points, simplified)
    apte = average_point_to_trajectory_error(original_points, simplified)
    frechet = frechet_distance(original_points, simplified)
    ped = perpendicular_euclidean_distance(original_points, simplified)
    sed, sed_series, query_time_sec, sync_points = synchronized_euclidean_distance(
        original, simplified, original_indices
    )
    dad = direction_aware_distance(original, sync_points)
    sad = speed_aware_distance(original, sync_points, query_time_sec)
    issd = integrated_synchronized_spatial_distance(sed_series, query_time_sec)
    
    # Compression
    comp_ratio = compression_ratio(original_points, simplified)
    
    metrics = {
        'hausdorff_distance': hausdorff,
        'average_pte': apte,
        'frechet_distance': frechet,
        'ped': ped,
        'dad': dad,
        'sed': sed,
        'sad': sad,
        'issd': issd,
        'compression_ratio': comp_ratio,
        'original_points': len(original_points),
        'simplified_points': len(simplified)
    }
    
    # Feature preservation metrics (if indices available)
    if original_indices is not None:
        # Create simplified DataFrame for feature metrics
        simplified_df = pd.DataFrame(simplified, columns=['lat', 'lon'])
        if 'timestamp' in original.columns:
            simplified_df['timestamp'] = original.iloc[original_indices]['timestamp'].values
        
        turn_pres, turn_metrics = turn_preservation_metric(
            original, simplified_df, original_indices
        )
        stop_pres, stop_metrics = stop_preservation_metric(
            original, simplified_df, original_indices
        )
        
        metrics.update({
            'turn_preservation': turn_pres,
            'stop_preservation': stop_pres,
            **turn_metrics,
            **stop_metrics
        })
    
    return metrics


if __name__ == "__main__":
    """
    Verify evaluation metrics on real GeoLife GPS trajectories.

    Loads preprocessed GeoLife data from data/processed/trajectories.pkl and
    tests all metrics across three algorithms (VW, Greedy Policy, Proposed) at
    5× compression, plus an identity check confirming zero error when simplified
    equals the original.
    """
    import sys, os, pickle
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.algorithms.baseline_algorithms import simplify_with_budget
    from src.algorithms.proposed_method import proposed_simplification

    DATA_FILE = Path("data/processed/trajectories.pkl")
    if not DATA_FILE.exists():
        print(f"[ERROR] {DATA_FILE} not found. Run: python src/utils/preprocess_geolife.py")
        sys.exit(1)

    print(f"Loading trajectories from {DATA_FILE} ...")
    with open(DATA_FILE, "rb") as f:
        all_trajs = pickle.load(f)

    # Use 3 short trajectories for fast but real evaluation
    trajs = sorted(all_trajs[:60], key=len)[:3]
    print(f"Selected {len(trajs)} trajectories, sizes: {[len(t) for t in trajs]}\n")

    ALGORITHMS = [
        ("vw",            lambda t, b: (simplify_with_budget(t, 'vw', b),           None)),
        ("greedy_policy", lambda t, b: (simplify_with_budget(t, 'greedy_policy', b), None)),
        ("proposed",      lambda t, b: proposed_simplification(t, b)),
    ]
    COMPRESSION_RATIO = 5.0

    print(f"{'Algorithm':<18} {'Traj':>5} {'N_orig':>6} {'N_simp':>6} "
          f"{'Hausdorff(m)':>13} {'APTE(m)':>9} {'Fréchet(m)':>11} "
          f"{'SED(m)':>8} {'DAD(°)':>7} "
          f"{'TurnPres':>9} {'StopPres':>9} {'RT(s)':>7}")
    print("-" * 118)

    import time, tracemalloc
    for traj in trajs:
        budget = max(2, int(len(traj) / COMPRESSION_RATIO))
        for name, fn in ALGORITHMS:
            tracemalloc.start()
            t0 = time.time()
            simp, idx = fn(traj, budget)
            elapsed = time.time() - t0
            tracemalloc.stop()

            m = compute_all_metrics(traj, simp, idx)
            tp = m.get("turn_preservation")
            sp = m.get("stop_preservation")
            tp_str = f"{tp:>9.3f}" if tp is not None else f"{'—':>9}"
            sp_str = f"{sp:>9.3f}" if sp is not None else f"{'—':>9}"
            print(
                f"{name:<18} {len(traj):>5} {m['original_points']:>6} {m['simplified_points']:>6} "
                f"{m['hausdorff_distance']:>13.2f} {m['average_pte']:>9.3f} {m['frechet_distance']:>11.2f} "
                f"{m['sed']:>8.2f} {m['dad']:>7.2f} "
                f"{tp_str} {sp_str} {elapsed:>7.3f}"
            )

    # ── Identity check: original == simplified → geometric error == 0 ──────
    print()
    print("Identity check (simplified == original with indices → all geometric errors ≈ 0):")
    traj = trajs[0]
    pts  = traj[['lat', 'lon']].values
    idx  = list(range(len(traj)))
    m    = compute_all_metrics(traj, pts, idx)
    geometric_keys = ['hausdorff_distance', 'average_pte', 'frechet_distance', 'ped', 'sed']
    all_zero = True
    for key in geometric_keys:
        val = m.get(key, float('nan'))
        status = "✓" if abs(val) < 1e-3 else "✗"
        if abs(val) >= 1e-3:
            all_zero = False
        print(f"  {status} {key}: {val:.6f} m")
    print(f"  {'✓' if all_zero else '✗'} turn_preservation = {m.get('turn_preservation', 'N/A')}")
    print(f"  {'✓' if all_zero else '✗'} stop_preservation = {m.get('stop_preservation', 'N/A')}")


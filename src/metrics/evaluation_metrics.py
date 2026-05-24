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


# ---------------------------------------------------------------------------
# Private vectorised helpers – not part of the public API.
# ---------------------------------------------------------------------------

def _haversine_batch(lats1: np.ndarray, lons1: np.ndarray,
                     lats2: np.ndarray, lons2: np.ndarray) -> np.ndarray:
    """Vectorised Haversine distance (degrees → metres) for arrays of points."""
    lat1 = np.radians(lats1); lon1 = np.radians(lons1)
    lat2 = np.radians(lats2); lon2 = np.radians(lons2)
    dlat = lat2 - lat1; dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 6371000.0 * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _haversine_matrix(orig: np.ndarray, simpl: np.ndarray) -> np.ndarray:
    """
    Compute an (n, m) Haversine distance matrix in one vectorised call.
    orig:  (n, 2) array of (lat, lon) in degrees
    simpl: (m, 2) array of (lat, lon) in degrees
    """
    lat1 = np.radians(orig[:, 0, np.newaxis])    # (n, 1)
    lon1 = np.radians(orig[:, 1, np.newaxis])    # (n, 1)
    lat2 = np.radians(simpl[np.newaxis, :, 0])   # (1, m)
    lon2 = np.radians(simpl[np.newaxis, :, 1])   # (1, m)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 6371000.0 * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))  # (n, m)


def _point_to_line_matrix(pts: np.ndarray,
                           seg_starts: np.ndarray,
                           seg_ends: np.ndarray) -> np.ndarray:
    """
    Compute perpendicular distances from n points to m line segments in one
    vectorised call, using the same projection + Haversine logic as the
    scalar point_to_line_distance function.

    pts:        (n, 2) – (lat, lon)
    seg_starts: (m, 2) – (lat, lon) of segment start points
    seg_ends:   (m, 2) – (lat, lon) of segment end points
    Returns:    (n, m) distance matrix in metres
    """
    P = pts[:, np.newaxis, :]          # (n, 1, 2)
    A = seg_starts[np.newaxis, :, :]   # (1, m, 2)
    B = seg_ends[np.newaxis, :, :]     # (1, m, 2)

    dx = B[..., 1] - A[..., 1]        # (1, m) lon component
    dy = B[..., 0] - A[..., 0]        # (1, m) lat component
    dxp = P[..., 1] - A[..., 1]       # (n, m)
    dyp = P[..., 0] - A[..., 0]       # (n, m)

    denom = dx * dx + dy * dy          # (1, m)
    safe_denom = np.where(denom > 1e-30, denom, 1.0)
    t = np.clip((dxp * dx + dyp * dy) / safe_denom, 0.0, 1.0)  # (n, m)

    lat_c = A[..., 0] + t * dy        # (n, m)
    lon_c = A[..., 1] + t * dx        # (n, m)

    # Degenerate segments (start == end): project to the start point
    degenerate = denom < 1e-30        # (1, m)
    lat_c = np.where(degenerate, A[..., 0], lat_c)
    lon_c = np.where(degenerate, A[..., 1], lon_c)

    lat1 = np.radians(P[..., 0])      # (n, 1)
    lon1 = np.radians(P[..., 1])      # (n, 1)
    lat2 = np.radians(lat_c)          # (n, m)
    lon2 = np.radians(lon_c)          # (n, m)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 6371000.0 * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))  # (n, m)


def _bearings_batch(starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """
    Vectorised forward bearing for arrays of (start, end) point pairs.
    starts, ends: (k, 2) in degrees.  Returns (k,) array in degrees [0, 360).
    """
    lat1 = np.radians(starts[:, 0]); lon1 = np.radians(starts[:, 1])
    lat2 = np.radians(ends[:, 0]);   lon2 = np.radians(ends[:, 1])
    dlon = lon2 - lon1
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return (np.degrees(np.arctan2(y, x)) + 360.0) % 360.0


# ---------------------------------------------------------------------------
# Public scalar helpers – imported by other modules; do not rename or remove.
# ---------------------------------------------------------------------------

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

    # (n, m) Haversine matrix – one vectorised call replaces O(n*m) Python calls
    dist_mat = _haversine_matrix(
        np.asarray(original,   dtype=float),
        np.asarray(simplified, dtype=float),
    )
    h_orig_simpl = float(dist_mat.min(axis=1).max())
    h_simpl_orig = float(dist_mat.min(axis=0).max())
    return max(h_orig_simpl, h_simpl_orig)


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

    orig = np.asarray(original,   dtype=float)
    simpl = np.asarray(simplified, dtype=float)

    # Point-to-point distances: (n, m)
    pt_dists = _haversine_matrix(orig, simpl)
    min_pt = pt_dists.min(axis=1)   # (n,)

    # Point-to-segment distances: (n, m-1)
    if len(simpl) >= 2:
        seg_dists = _point_to_line_matrix(orig, simpl[:-1], simpl[1:])
        min_seg = seg_dists.min(axis=1)   # (n,)
        min_dists = np.minimum(min_pt, min_seg)
    else:
        min_dists = min_pt

    return float(np.mean(min_dists))


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

    orig = np.asarray(original,   dtype=float)
    simpl = np.asarray(simplified, dtype=float)
    n, m = len(orig), len(simpl)

    # Vectorised distance matrix (replaces O(n*m) scalar haversine calls)
    dist_matrix = _haversine_matrix(orig, simpl)

    # Dynamic programming (inherently sequential; initialise boundary rows
    # with np.maximum.accumulate to avoid two Python loops)
    F = np.empty((n, m), dtype=float)
    F[0, 0] = dist_matrix[0, 0]
    F[0, 1:] = np.maximum.accumulate(dist_matrix[0, 1:])
    F[1:, 0] = np.maximum.accumulate(dist_matrix[1:, 0])

    for i in range(1, n):
        for j in range(1, m):
            F[i, j] = max(
                dist_matrix[i, j],
                min(F[i - 1, j], F[i, j - 1], F[i - 1, j - 1])
            )

    return float(F[n - 1, m - 1])


def turn_preservation_metric(original: pd.DataFrame,
                            simplified: pd.DataFrame,
                            original_indices: List[int],
                            turn_threshold: float = 30.0) -> Tuple[float, Dict]:
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
                            stop_threshold: float = 1.0,
                            min_duration: float = 30.0) -> Tuple[float, Dict]:
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


def _extract_time_seconds(original: pd.DataFrame) -> np.ndarray:
    """Extract monotonic time values in seconds for synchronization."""
    if 'timestamp' in original.columns:
        ts = pd.to_datetime(original['timestamp'])
        # Relative seconds to improve numerical stability.
        time_sec = (ts - ts.iloc[0]).dt.total_seconds().to_numpy(dtype=float)
    else:
        time_sec = np.arange(len(original), dtype=float)
    return time_sec


def _make_monotonic(values: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Ensure strictly increasing values for interpolation."""
    out = values.astype(float).copy()
    for i in range(1, len(out)):
        if out[i] <= out[i - 1]:
            out[i] = out[i - 1] + eps
    return out


def _bearing_degrees(start: np.ndarray, end: np.ndarray) -> float:
    """Compute bearing from start(lat, lon) to end(lat, lon) in degrees [0, 360)."""
    lat1, lon1 = np.radians(start[0]), np.radians(start[1])
    lat2, lon2 = np.radians(end[0]), np.radians(end[1])
    dlon = lon2 - lon1
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    bearing = np.degrees(np.arctan2(y, x))
    return float((bearing + 360.0) % 360.0)


def _angular_diff_deg(a: float, b: float) -> float:
    """Smallest absolute difference between two angles in degrees."""
    d = abs(a - b) % 360.0
    return float(min(d, 360.0 - d))


def _synchronized_positions(original: pd.DataFrame,
                            simplified: np.ndarray,
                            original_indices: List[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Interpolate simplified trajectory positions at original timestamps.

    Returns:
        query_time_sec: Original time axis (N,)
        sync_points: Interpolated simplified points aligned to query_time_sec (N, 2)
        simplified_time_sec: Time axis used for simplified points (M,)
    """
    query_time_sec = _extract_time_seconds(original)
    query_time_sec = _make_monotonic(query_time_sec)

    if original_indices is not None and len(original_indices) == len(simplified):
        idx = np.clip(np.asarray(original_indices, dtype=int), 0, len(original) - 1)
        simplified_time_sec = query_time_sec[idx]
    else:
        simplified_time_sec = np.linspace(query_time_sec[0], query_time_sec[-1], len(simplified), dtype=float)

    simplified_time_sec = _make_monotonic(simplified_time_sec)
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

    orig = np.asarray(original,   dtype=float)
    simpl = np.asarray(simplified, dtype=float)

    # (n, m-1) matrix of point-to-segment distances – one vectorised call
    seg_dists = _point_to_line_matrix(orig, simpl[:-1], simpl[1:])
    return float(np.mean(seg_dists.min(axis=1)))


def synchronized_euclidean_distance(original: pd.DataFrame,
                                    simplified: np.ndarray,
                                    original_indices: List[int] = None) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    SED: Mean Euclidean distance between original and time-synchronized simplified points.
    """
    if len(original) == 0 or len(simplified) == 0:
        return float('inf'), np.array([]), np.array([]), np.empty((0, 2))

    original_points = original[['lat', 'lon']].to_numpy(dtype=float)
    query_time_sec, sync_points, _ = _synchronized_positions(original, simplified, original_indices)

    # Vectorised pairwise haversine (one call replaces n scalar calls)
    dists = _haversine_batch(
        original_points[:, 0], original_points[:, 1],
        sync_points[:, 0],     sync_points[:, 1],
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

    starts_orig = original_points[:-1]
    ends_orig   = original_points[1:]
    starts_sync = synchronized_points[:-1]
    ends_sync   = synchronized_points[1:]

    # Filter out near-stationary segments (vectorised distance check)
    d_orig = _haversine_batch(starts_orig[:, 0], starts_orig[:, 1],
                               ends_orig[:, 0],   ends_orig[:, 1])
    d_sync = _haversine_batch(starts_sync[:, 0], starts_sync[:, 1],
                               ends_sync[:, 0],   ends_sync[:, 1])
    valid = (d_orig >= 1e-6) & (d_sync >= 1e-6)

    if not np.any(valid):
        return 0.0

    # Vectorised bearing computation
    b_orig = _bearings_batch(starts_orig[valid], ends_orig[valid])
    b_sync = _bearings_batch(starts_sync[valid], ends_sync[valid])

    diff = np.abs(b_orig - b_sync) % 360.0
    diff = np.minimum(diff, 360.0 - diff)
    return float(np.mean(diff))


def speed_aware_distance(original: pd.DataFrame,
                         synchronized_points: np.ndarray,
                         query_time_sec: np.ndarray) -> float:
    """
    SAD: Mean absolute speed difference (m/s) between original and synchronized simplified trajectories.
    """
    original_points = original[['lat', 'lon']].to_numpy(dtype=float)
    if len(original_points) < 2 or len(synchronized_points) < 2 or len(query_time_sec) < 2:
        return 0.0

    dt = np.diff(query_time_sec)
    valid = dt > 0
    if not np.any(valid):
        return 0.0

    # Vectorised distances for valid segments
    d_orig = _haversine_batch(
        original_points[:-1][valid, 0], original_points[:-1][valid, 1],
        original_points[1:][valid, 0],  original_points[1:][valid, 1],
    )
    d_sync = _haversine_batch(
        synchronized_points[:-1][valid, 0], synchronized_points[:-1][valid, 1],
        synchronized_points[1:][valid, 0],  synchronized_points[1:][valid, 1],
    )
    v_orig = d_orig / dt[valid]
    v_sync = d_sync / dt[valid]
    return float(np.mean(np.abs(v_orig - v_sync)))


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
    # Example usage
    import pandas as pd
    
    # Create sample trajectories
    n = 100
    original = pd.DataFrame({
        'lat': np.linspace(0, 1, n) + np.random.normal(0, 0.01, n),
        'lon': np.linspace(0, 1, n) + np.random.normal(0, 0.01, n),
        'timestamp': pd.date_range('2023-01-01', periods=n, freq='1min')
    })
    
    # Simplified (every 5th point)
    simplified = original.iloc[::5][['lat', 'lon']].values
    indices = list(range(0, n, 5))
    
    # Compute metrics
    metrics = compute_all_metrics(original, simplified, indices)
    
    print("Evaluation Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

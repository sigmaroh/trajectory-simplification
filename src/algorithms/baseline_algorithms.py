"""
PHASE 2: Baseline Trajectory Simplification Algorithms

This module implements standard baseline algorithms for trajectory simplification:
1. Douglas-Peucker (RDP) - Geometric distance-based
2. Sliding Window - Local error threshold
3. Uniform Sampling - Fixed interval sampling
4. Adaptive Threshold - Dynamic error threshold based on speed
5. Visvalingam-Whyatt (VW) - Effective-area based
6. Reumann-Witkam (RW) - Strip/corridor based
7. SQUISH - Priority-based point removal

Each algorithm has different strengths and weaknesses:
- RDP: Good for geometric preservation, but ignores temporal/speed information
- Sliding Window: Handles local variations, but may miss global patterns
- Uniform Sampling: Simple and fast, but ignores trajectory shape
- Adaptive: Considers speed, but may be sensitive to noise
"""

import numpy as np
from typing import List, Tuple, Union
import pandas as pd


# ---------------------------------------------------------------------------
# Private vectorised helpers – not part of the public API.
#
# These replicate the scalar logic of haversine_distance and
# point_to_line_distance but operate on numpy arrays so that the inner
# distance loops of every algorithm can be expressed as a single numpy
# call instead of iterating point-by-point in Python.
# ---------------------------------------------------------------------------

def _haversine_batch(lats1: np.ndarray, lons1: np.ndarray,
                     lats2: np.ndarray, lons2: np.ndarray) -> np.ndarray:
    """Vectorised Haversine distance (degrees → metres) for arrays of points."""
    lat1 = np.radians(lats1)
    lon1 = np.radians(lons1)
    lat2 = np.radians(lats2)
    lon2 = np.radians(lons2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 6371000.0 * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _point_to_line_distance_batch(pts: np.ndarray,
                                   line_start: Tuple[float, float],
                                   line_end: Tuple[float, float]) -> np.ndarray:
    """
    Perpendicular distances from an array of (lat, lon) points to a line
    segment, using exactly the same projection + Haversine logic as the
    scalar point_to_line_distance function.
    """
    lat1, lon1 = line_start
    lat2, lon2 = line_end
    dx = lon2 - lon1
    dy = lat2 - lat1
    if dx == 0 and dy == 0:
        return _haversine_batch(
            pts[:, 0], pts[:, 1],
            np.full(len(pts), lat1, dtype=float),
            np.full(len(pts), lon1, dtype=float),
        )
    denom = dx * dx + dy * dy
    dx_p = pts[:, 1] - lon1
    dy_p = pts[:, 0] - lat1
    t = np.clip((dx_p * dx + dy_p * dy) / denom, 0.0, 1.0)
    return _haversine_batch(pts[:, 0], pts[:, 1], lat1 + t * dy, lon1 + t * dx)


# ---------------------------------------------------------------------------
# Public scalar helpers – imported by other modules; do not rename or remove.
# ---------------------------------------------------------------------------

def haversine_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """
    Compute Haversine distance between two (lat, lon) points in meters.
    
    Args:
        p1: (latitude, longitude) tuple
        p2: (latitude, longitude) tuple
        
    Returns:
        Distance in meters
    """
    lat1, lon1 = np.radians(p1[0]), np.radians(p1[1])
    lat2, lon2 = np.radians(p2[0]), np.radians(p2[1])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    earth_radius = 6371000  # meters
    return earth_radius * c


def point_to_line_distance(point: Tuple[float, float], 
                          line_start: Tuple[float, float],
                          line_end: Tuple[float, float]) -> float:
    """
    Compute perpendicular distance from point to line segment.
    
    Args:
        point: (lat, lon) point
        line_start: (lat, lon) line start
        line_end: (lat, lon) line end
        
    Returns:
        Distance in meters
    """
    # Convert to approximate metric coordinates for distance calculation
    # Using simple approximation (works for small distances)
    lat1, lon1 = line_start
    lat2, lon2 = line_end
    lat_p, lon_p = point
    
    # Vector from start to end
    dx = lon2 - lon1
    dy = lat2 - lat1
    
    # Vector from start to point
    dx_p = lon_p - lon1
    dy_p = lat_p - lat1
    
    # Project point onto line
    if dx == 0 and dy == 0:
        # Line is a point
        return haversine_distance(point, line_start)
    
    t = (dx_p * dx + dy_p * dy) / (dx * dx + dy * dy)
    t = np.clip(t, 0, 1)
    
    # Closest point on line
    lat_closest = lat1 + t * dy
    lon_closest = lon1 + t * dx
    
    return haversine_distance(point, (lat_closest, lon_closest))


def douglas_peucker(trajectory: Union[pd.DataFrame, np.ndarray],
                   epsilon: float,
                   indices: bool = False) -> Union[np.ndarray, List[int]]:
    """
    Douglas-Peucker (RDP) algorithm for trajectory simplification.
    
    Algorithm:
    1. Find the point with maximum distance from line between first and last
    2. If max distance > epsilon, recursively simplify both segments
    3. Otherwise, return endpoints only
    
    Complexity: O(n²) worst case, O(n log n) average
    When it fails: Irregular sampling, speed variations, stops
    
    Args:
        trajectory: DataFrame with 'lat', 'lon' columns or array of (lat, lon)
        epsilon: Maximum allowed distance error (meters)
        indices: If True, return indices instead of points
        
    Returns:
        Simplified trajectory or list of indices
    """
    if isinstance(trajectory, pd.DataFrame):
        points = trajectory[['lat', 'lon']].values
    else:
        points = np.array(trajectory)
    
    if len(points) <= 2:
        if indices:
            return list(range(len(points)))
        return points
    
    def rdp_recursive(start_idx: int, end_idx: int) -> List[int]:
        """Recursive RDP implementation."""
        if end_idx - start_idx <= 1:
            return [start_idx, end_idx]
        
        # Vectorised: compute all interior distances in one numpy call
        interior = points[start_idx + 1:end_idx]
        dists = _point_to_line_distance_batch(
            interior,
            tuple(points[start_idx]),
            tuple(points[end_idx]),
        )
        max_idx_rel = int(np.argmax(dists))
        max_dist = float(dists[max_idx_rel])
        max_idx = start_idx + 1 + max_idx_rel
        
        # If max distance > epsilon, recursively simplify
        if max_dist > epsilon:
            left = rdp_recursive(start_idx, max_idx)
            right = rdp_recursive(max_idx, end_idx)
            return left[:-1] + right  # Remove duplicate max_idx
        else:
            return [start_idx, end_idx]
    
    indices_list = rdp_recursive(0, len(points) - 1)
    indices_list = sorted(list(set(indices_list)))  # Remove duplicates and sort
    
    if indices:
        return indices_list
    
    return points[indices_list]


def sliding_window(trajectory: Union[pd.DataFrame, np.ndarray],
                  epsilon: float,
                  indices: bool = False) -> Union[np.ndarray, List[int]]:
    """
    Sliding Window algorithm for trajectory simplification.
    
    Algorithm:
    1. Start with first point
    2. Extend window until error exceeds threshold
    3. Keep last point before threshold, start new window
    
    Complexity: O(n)
    When it fails: Global patterns, long straight segments with noise
    
    Args:
        trajectory: DataFrame with 'lat', 'lon' columns or array of (lat, lon)
        epsilon: Maximum allowed distance error (meters)
        indices: If True, return indices instead of points
        
    Returns:
        Simplified trajectory or list of indices
    """
    if isinstance(trajectory, pd.DataFrame):
        points = trajectory[['lat', 'lon']].values
    else:
        points = np.array(trajectory)
    
    if len(points) <= 2:
        if indices:
            return list(range(len(points)))
        return points
    
    indices_list = [0]
    start_idx = 0
    
    for i in range(2, len(points)):
        # Vectorised: check all interior points against line (start_idx → i)
        interior = points[start_idx + 1:i]
        if len(interior) > 0:
            dists = _point_to_line_distance_batch(
                interior,
                tuple(points[start_idx]),
                tuple(points[i]),
            )
            max_error = float(dists.max())
        else:
            max_error = 0.0

        if max_error > epsilon:
            # Keep previous point, start new window
            indices_list.append(i - 1)
            start_idx = i - 1
    
    # Always include last point
    if indices_list[-1] != len(points) - 1:
        indices_list.append(len(points) - 1)
    
    if indices:
        return indices_list
    
    return points[indices_list]


def uniform_sampling(trajectory: Union[pd.DataFrame, np.ndarray],
                    num_points: int,
                    indices: bool = False) -> Union[np.ndarray, List[int]]:
    """
    Uniform sampling: select points at regular intervals.
    
    Algorithm:
    1. Compute step size: (n-1) / (k-1)
    2. Sample points at regular indices
    
    Complexity: O(n)
    When it fails: Important points may be skipped, ignores trajectory shape
    
    Args:
        trajectory: DataFrame with 'lat', 'lon' columns or array of (lat, lon)
        num_points: Target number of points in simplified trajectory
        indices: If True, return indices instead of points
        
    Returns:
        Simplified trajectory or list of indices
    """
    if isinstance(trajectory, pd.DataFrame):
        points = trajectory[['lat', 'lon']].values
    else:
        points = np.array(trajectory)
    
    if len(points) <= num_points:
        if indices:
            return list(range(len(points)))
        return points
    
    # Compute indices for uniform sampling
    indices_list = np.linspace(0, len(points) - 1, num_points, dtype=int).tolist()
    indices_list = sorted(list(set(indices_list)))  # Remove duplicates
    
    if indices:
        return indices_list
    
    return points[indices_list]


def adaptive_threshold(trajectory: Union[pd.DataFrame, np.ndarray],
                      base_epsilon: float,
                      speed_weight: float = 0.5,
                      indices: bool = False) -> Union[np.ndarray, List[int]]:
    """
    Adaptive threshold algorithm: adjusts error threshold based on speed.
    
    Algorithm:
    1. Compute speed at each point
    2. Adjust epsilon based on speed: higher speed -> larger threshold
    3. Apply sliding window with adaptive threshold
    
    Complexity: O(n)
    When it fails: Noise in speed measurements, abrupt speed changes
    
    Args:
        trajectory: DataFrame with 'lat', 'lon', 'timestamp' columns or array
        base_epsilon: Base error threshold (meters)
        speed_weight: Weight for speed-based adjustment (0-1)
        indices: If True, return indices instead of points
        
    Returns:
        Simplified trajectory or list of indices
    """
    if isinstance(trajectory, pd.DataFrame):
        points = trajectory[['lat', 'lon']].values
        if 'timestamp' in trajectory.columns:
            timestamps = pd.to_datetime(trajectory['timestamp']).values
        else:
            # Use uniform time if no timestamp
            timestamps = np.arange(len(trajectory))
    else:
        points = np.array(trajectory)
        timestamps = np.arange(len(trajectory))
    
    if len(points) <= 2:
        if indices:
            return list(range(len(points)))
        return points
    
    # Vectorised speed computation (replaces per-point Python loop)
    dists_consec = _haversine_batch(
        points[:-1, 0], points[:-1, 1],
        points[1:,  0], points[1:,  1],
    )
    # Convert timestamps to int64 nanoseconds for vectorised diff
    _ts_ns = np.asarray(pd.to_datetime(timestamps), dtype='datetime64[ns]').astype(np.int64)
    time_diffs_s = np.diff(_ts_ns) * 1e-9  # nanoseconds → seconds

    speeds = np.zeros(len(points))
    valid = time_diffs_s > 0
    speeds[1:] = np.where(valid, dists_consec / np.where(valid, time_diffs_s, 1.0), 0.0)
    # Propagate previous speed for zero-duration intervals (original behaviour)
    for i in range(1, len(points)):
        if time_diffs_s[i - 1] <= 0:
            speeds[i] = speeds[i - 1] if i > 1 else 0.0
    
    # Normalize speeds to [0, 1]
    if np.max(speeds) > 0:
        speeds_norm = speeds / np.max(speeds)
    else:
        speeds_norm = np.zeros_like(speeds)
    
    # Adaptive epsilon: higher speed -> larger threshold
    adaptive_epsilons = base_epsilon * (1 + speed_weight * speeds_norm)
    
    # Apply sliding window with adaptive threshold (vectorised inner loop)
    indices_list = [0]
    start_idx = 0
    
    for i in range(2, len(points)):
        # Use average epsilon for the segment
        avg_epsilon = float(adaptive_epsilons[start_idx:i + 1].mean())

        # Vectorised: check all interior points in one batch call
        interior = points[start_idx + 1:i]
        if len(interior) > 0:
            dists = _point_to_line_distance_batch(
                interior,
                tuple(points[start_idx]),
                tuple(points[i]),
            )
            max_error = float(dists.max())
        else:
            max_error = 0.0
        
        if max_error > avg_epsilon:
            indices_list.append(i - 1)
            start_idx = i - 1
    
    if indices_list[-1] != len(points) - 1:
        indices_list.append(len(points) - 1)
    
    if indices:
        return indices_list
    
    return points[indices_list]


def visvalingam_whyatt(trajectory: Union[pd.DataFrame, np.ndarray],
                       num_points: int,
                       indices: bool = False) -> Union[np.ndarray, List[int]]:
    """
    Visvalingam-Whyatt simplification using effective triangle areas.

    Args:
        trajectory: DataFrame with 'lat', 'lon' columns or array of (lat, lon)
        num_points: Target number of points
        indices: If True, return indices

    Returns:
        Simplified trajectory or selected indices
    """
    if isinstance(trajectory, pd.DataFrame):
        points = trajectory[['lat', 'lon']].values
    else:
        points = np.array(trajectory)

    n = len(points)
    if n <= num_points:
        idx = list(range(n))
        return idx if indices else points
    if num_points <= 2:
        idx = [0, n - 1]
        return idx if indices else points[idx]

    active = list(range(n))

    while len(active) > num_points:
        # Vectorised: compute triangle areas for all interior positions at once
        arr = np.array(active)
        p1 = points[arr[:-2]]   # previous neighbours
        p2 = points[arr[1:-1]]  # interior candidates
        p3 = points[arr[2:]]    # next neighbours
        areas = np.abs(
            p1[:, 1] * (p2[:, 0] - p3[:, 0]) +
            p2[:, 1] * (p3[:, 0] - p1[:, 0]) +
            p3[:, 1] * (p1[:, 0] - p2[:, 0])
        ) / 2.0
        # areas[k] corresponds to active position k+1 (endpoints excluded)
        min_pos = int(np.argmin(areas)) + 1
        if min_pos is None:
            break
        active.pop(min_pos)

    active = sorted(active)
    if indices:
        return active
    return points[active]


def reumann_witkam(trajectory: Union[pd.DataFrame, np.ndarray],
                   epsilon: float,
                   indices: bool = False) -> Union[np.ndarray, List[int]]:
    """
    Reumann-Witkam simplification with a fixed strip width epsilon.

    Args:
        trajectory: DataFrame with 'lat', 'lon' columns or array of (lat, lon)
        epsilon: Strip width threshold (meters)
        indices: If True, return indices

    Returns:
        Simplified trajectory or selected indices
    """
    if isinstance(trajectory, pd.DataFrame):
        points = trajectory[['lat', 'lon']].values
    else:
        points = np.array(trajectory)

    n = len(points)
    if n <= 2:
        idx = list(range(n))
        return idx if indices else points

    selected = [0]
    anchor = 0

    while anchor < n - 1:
        if anchor + 1 >= n:
            break

        line_start = tuple(points[anchor])
        line_end = tuple(points[anchor + 1])
        candidate = anchor + 2
        last_inside = anchor + 1

        while candidate < n:
            dist = point_to_line_distance(tuple(points[candidate]), line_start, line_end)
            if dist <= epsilon:
                last_inside = candidate
                candidate += 1
            else:
                break

        if last_inside <= anchor:
            last_inside = anchor + 1

        selected.append(last_inside)
        anchor = last_inside

    if selected[-1] != n - 1:
        selected.append(n - 1)

    selected = sorted(list(set(selected)))
    if indices:
        return selected
    return points[selected]


def squish(trajectory: Union[pd.DataFrame, np.ndarray],
           num_points: int,
           indices: bool = False) -> Union[np.ndarray, List[int]]:
    """
    SQUISH-style priority removal using local triangle areas.

    Args:
        trajectory: DataFrame with 'lat', 'lon' columns or array of (lat, lon)
        num_points: Target number of points
        indices: If True, return indices

    Returns:
        Simplified trajectory or selected indices
    """
    if isinstance(trajectory, pd.DataFrame):
        points = trajectory[['lat', 'lon']].values
    else:
        points = np.array(trajectory)

    n = len(points)
    if n <= num_points:
        idx = list(range(n))
        return idx if indices else points
    if num_points <= 2:
        idx = [0, n - 1]
        return idx if indices else points[idx]

    active = list(range(n))

    while len(active) > num_points:
        # Vectorised: compute interior triangle areas (endpoints preserved)
        arr = np.array(active)
        if len(arr) <= 2:
            break
        p1 = points[arr[:-2]]
        p2 = points[arr[1:-1]]
        p3 = points[arr[2:]]
        interior_areas = np.abs(
            p1[:, 1] * (p2[:, 0] - p3[:, 0]) +
            p2[:, 1] * (p3[:, 0] - p1[:, 0]) +
            p3[:, 1] * (p1[:, 0] - p2[:, 0])
        ) / 2.0
        # interior_areas[k] → active position k+1 (endpoints always preserved)
        remove_pos = int(np.argmin(interior_areas)) + 1
        if remove_pos == 0 or remove_pos == len(active) - 1:
            break
        active.pop(remove_pos)

    active = sorted(active)
    if indices:
        return active
    return points[active]


def simplify_with_budget(trajectory: Union[pd.DataFrame, np.ndarray],
                        algorithm: str,
                        budget: int,
                        **kwargs) -> Union[np.ndarray, List[int]]:
    """
    Simplify trajectory to a fixed budget (number of points).
    
    Uses binary search to find appropriate parameters for algorithms
    that don't directly support budget constraints.
    
    Args:
        trajectory: Input trajectory
        algorithm: Algorithm name ('rdp', 'sliding_window', 'uniform', 'adaptive')
        budget: Target number of points
        **kwargs: Additional algorithm-specific parameters
        
    Returns:
        Simplified trajectory
    """
    if isinstance(trajectory, pd.DataFrame):
        n = len(trajectory)
    else:
        n = len(trajectory)
    
    if n <= budget:
        if isinstance(trajectory, pd.DataFrame):
            return trajectory[['lat', 'lon']].values
        return np.array(trajectory)
    
    algorithm_key = algorithm.lower().replace(" ", "_")
    algorithm_aliases = {
        'original': 'original',
        'rdp': 'rdp',
        'dp': 'rdp',
        'douglas-peucker': 'rdp',
        'douglas_peucker': 'rdp',
        'sliding_window': 'sliding_window',
        'sliding-window': 'sliding_window',
        'sw': 'sliding_window',
        'uniform': 'uniform',
        'adaptive': 'adaptive',
        'visvalingam-whyatt': 'vw',
        'visvalingam_whyatt': 'vw',
        'vw': 'vw',
        'reumann-witkam': 'rw',
        'reumann_witkam': 'rw',
        'rw': 'rw',
        'squish': 'squish',
    }
    algorithm = algorithm_aliases.get(algorithm_key, algorithm_key)

    if algorithm == 'original':
        if isinstance(trajectory, pd.DataFrame):
            return trajectory[['lat', 'lon']].values
        return np.array(trajectory)
    
    if algorithm == 'uniform':
        return uniform_sampling(trajectory, budget, indices=False)
    
    elif algorithm == 'rdp':
        # Binary search for epsilon (widened upper bound handles long trajectories)
        epsilon_min, epsilon_max = 0, 100_000  # metres
        best_result = None
        
        for _ in range(20):  # Max 20 iterations
            epsilon = (epsilon_min + epsilon_max) / 2
            result = douglas_peucker(trajectory, epsilon, indices=True)
            
            if len(result) <= budget:
                best_result = result
                epsilon_max = epsilon
            else:
                epsilon_min = epsilon
            
            if len(result) == budget:  # exact hit – no need to continue
                break
        
        if best_result is None:
            best_result = result
        
        if isinstance(trajectory, pd.DataFrame):
            return trajectory.iloc[best_result][['lat', 'lon']].values
        return np.array(trajectory)[best_result]
    
    elif algorithm == 'sliding_window':
        # Binary search for epsilon
        epsilon_min, epsilon_max = 0, 100_000
        best_result = None
        
        for _ in range(20):
            epsilon = (epsilon_min + epsilon_max) / 2
            result = sliding_window(trajectory, epsilon, indices=True)
            
            if len(result) <= budget:
                best_result = result
                epsilon_max = epsilon
            else:
                epsilon_min = epsilon
            
            if len(result) == budget:
                break
        
        if best_result is None:
            best_result = result
        
        if isinstance(trajectory, pd.DataFrame):
            return trajectory.iloc[best_result][['lat', 'lon']].values
        return np.array(trajectory)[best_result]
    
    elif algorithm == 'adaptive':
        base_epsilon = kwargs.get('base_epsilon', 10.0)
        speed_weight = kwargs.get('speed_weight', 0.5)
        
        # Binary search for base_epsilon
        epsilon_min, epsilon_max = 0, 100_000
        best_result = None
        
        for _ in range(20):
            base_eps = (epsilon_min + epsilon_max) / 2
            result = adaptive_threshold(trajectory, base_eps, speed_weight, indices=True)
            
            if len(result) <= budget:
                best_result = result
                epsilon_max = base_eps
            else:
                epsilon_min = base_eps
            
            if len(result) == budget:
                break
        
        if best_result is None:
            best_result = result
        
        if isinstance(trajectory, pd.DataFrame):
            return trajectory.iloc[best_result][['lat', 'lon']].values
        return np.array(trajectory)[best_result]

    elif algorithm == 'vw':
        return visvalingam_whyatt(trajectory, budget, indices=False)

    elif algorithm == 'squish':
        return squish(trajectory, budget, indices=False)

    elif algorithm == 'rw':
        epsilon_min, epsilon_max = 0, 100_000
        best_result = None

        for _ in range(20):
            epsilon = (epsilon_min + epsilon_max) / 2
            result = reumann_witkam(trajectory, epsilon, indices=True)

            if len(result) <= budget:
                best_result = result
                epsilon_max = epsilon
            else:
                epsilon_min = epsilon

            if len(result) == budget:
                break

        if best_result is None:
            best_result = result

        if isinstance(trajectory, pd.DataFrame):
            return trajectory.iloc[best_result][['lat', 'lon']].values
        return np.array(trajectory)[best_result]
    
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


if __name__ == "__main__":
    # Example usage
    import pandas as pd
    
    # Create sample trajectory
    n = 100
    trajectory = pd.DataFrame({
        'lat': np.linspace(0, 1, n) + np.random.normal(0, 0.01, n),
        'lon': np.linspace(0, 1, n) + np.random.normal(0, 0.01, n),
        'timestamp': pd.date_range('2023-01-01', periods=n, freq='1min')
    })
    
    print(f"Original trajectory: {len(trajectory)} points")
    
    # Test each algorithm
    budget = 20
    
    rdp_result = simplify_with_budget(trajectory, 'rdp', budget)
    print(f"RDP result: {len(rdp_result)} points")
    
    sw_result = simplify_with_budget(trajectory, 'sliding_window', budget)
    print(f"Sliding Window result: {len(sw_result)} points")
    
    uniform_result = simplify_with_budget(trajectory, 'uniform', budget)
    print(f"Uniform result: {len(uniform_result)} points")
    
    adaptive_result = simplify_with_budget(trajectory, 'adaptive', budget, base_epsilon=10.0)
    print(f"Adaptive result: {len(adaptive_result)} points")

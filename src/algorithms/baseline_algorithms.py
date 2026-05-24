"""
PHASE 2: Baseline Trajectory Simplification Algorithms

This module implements standard baseline algorithms for trajectory simplification:
1. Douglas-Peucker (RDP) - Geometric distance-based
2. Sliding Window - Local error threshold
3. Visvalingam-Whyatt (VW) - Effective-area based
4. Reumann-Witkam (RW) - Strip/corridor based
5. SQUISH - Priority-based point removal
6. Greedy Policy (RL-inspired) - Sequential keep/drop via local value function
   Inspired by: Wang et al. (2021). Trajectory simplification with reinforcement
   learning. ICDE 2021, 684-695. IEEE.

Each algorithm has different strengths and weaknesses:
- RDP: Good for geometric preservation, but ignores temporal/speed information
- Sliding Window: Handles local variations, but may miss global patterns
- Greedy Policy: Balances geometric deviation and motion change signal
"""

import numpy as np
from typing import Callable, List, Optional, Tuple, Union
import pandas as pd

from src.utils.config import (
    EARTH_RADIUS_M,
    BINARY_SEARCH_ITERATIONS,
    BINARY_SEARCH_EPS_MIN,
    BINARY_SEARCH_EPS_MAX,
    BINARY_SEARCH_TOLERANCE,
)


def trajectory_to_points(trajectory: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
    """Normalise a trajectory input to an (N, 2) numpy array of [lat, lon]."""
    if isinstance(trajectory, pd.DataFrame):
        return trajectory[['lat', 'lon']].values
    return np.array(trajectory)


def select_points(trajectory: Union[pd.DataFrame, np.ndarray],
                  selected_indices: Optional[List[int]] = None) -> np.ndarray:
    """Return all trajectory points, or a subset selected by index."""
    points = trajectory_to_points(trajectory)
    if selected_indices is None:
        return points
    return points[selected_indices]


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

    return EARTH_RADIUS_M * c


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
    points = trajectory_to_points(trajectory)
    
    if len(points) <= 2:
        if indices:
            return list(range(len(points)))
        return points
    
    def rdp_recursive(start_idx: int, end_idx: int) -> List[int]:
        """Recursive RDP implementation."""
        if end_idx - start_idx <= 1:
            return [start_idx]
        
        # Find point with maximum distance
        max_dist = 0
        max_idx = start_idx
        
        for i in range(start_idx + 1, end_idx):
            dist = point_to_line_distance(
                tuple(points[i]),
                tuple(points[start_idx]),
                tuple(points[end_idx])
            )
            if dist > max_dist:
                max_dist = dist
                max_idx = i
        
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
    points = trajectory_to_points(trajectory)
    
    if len(points) <= 2:
        if indices:
            return list(range(len(points)))
        return points
    
    indices_list = [0]
    start_idx = 0
    
    for i in range(2, len(points)):
        # Check if adding point i violates error constraint
        max_error = 0
        for j in range(start_idx + 1, i):
            dist = point_to_line_distance(
                tuple(points[j]),
                tuple(points[start_idx]),
                tuple(points[i])
            )
            max_error = max(max_error, dist)
        
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
    points = trajectory_to_points(trajectory)

    n = len(points)
    if n <= num_points:
        idx = list(range(n))
        return idx if indices else points
    if num_points <= 2:
        idx = [0, n - 1]
        return idx if indices else points[idx]

    active = list(range(n))

    def triangle_area(i_prev: int, i_curr: int, i_next: int) -> float:
        p1 = points[i_prev]
        p2 = points[i_curr]
        p3 = points[i_next]
        return abs(
            p1[1] * (p2[0] - p3[0]) +
            p2[1] * (p3[0] - p1[0]) +
            p3[1] * (p1[0] - p2[0])
        ) / 2.0

    while len(active) > num_points:
        min_area = np.inf
        min_pos = None
        for pos in range(1, len(active) - 1):
            area = triangle_area(active[pos - 1], active[pos], active[pos + 1])
            if area < min_area:
                min_area = area
                min_pos = pos
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
    points = trajectory_to_points(trajectory)

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
    points = trajectory_to_points(trajectory)

    n = len(points)
    if n <= num_points:
        idx = list(range(n))
        return idx if indices else points
    if num_points <= 2:
        idx = [0, n - 1]
        return idx if indices else points[idx]

    active = list(range(n))

    def priority(pos: int) -> float:
        # Endpoints are always preserved.
        if pos == 0 or pos == len(active) - 1:
            return np.inf
        i_prev, i_curr, i_next = active[pos - 1], active[pos], active[pos + 1]
        p1, p2, p3 = points[i_prev], points[i_curr], points[i_next]
        area = abs(
            p1[1] * (p2[0] - p3[0]) +
            p2[1] * (p3[0] - p1[0]) +
            p3[1] * (p1[0] - p2[0])
        ) / 2.0
        return area

    while len(active) > num_points:
        priorities = [priority(pos) for pos in range(len(active))]
        remove_pos = int(np.argmin(priorities))
        if remove_pos == 0 or remove_pos == len(active) - 1:
            break
        active.pop(remove_pos)

    active = sorted(active)
    if indices:
        return active
    return points[active]


def greedy_policy_simplification(
        trajectory: Union[pd.DataFrame, np.ndarray],
        num_points: int,
        alpha: float = 0.5,
        indices: bool = False) -> Union[np.ndarray, List[int]]:
    """
    Greedy sequential-policy simplification (RL-inspired baseline).

    Inspired by Wang et al. (2021) "Trajectory Simplification with Reinforcement
    Learning" (ICDE 2021), which frames simplification as a Markov Decision
    Process where an agent sequentially decides whether to keep each point.

    Here we implement a deterministic greedy policy: every interior point is
    scored by a value function that combines geometric deviation and motion-change
    signal, and the top-(num_points - 2) interior points are retained together
    with the mandatory endpoints.

    Value function for interior point p_i:
        v(i) = alpha       * geo_dev(i)
             + (1 - alpha) * motion_change(i)

    where:
        geo_dev(i)       = perpendicular distance from p_i to line(p_{i-1}, p_{i+1}),
                           normalised to [0, 1] over the trajectory.
        motion_change(i) = 0.5 * norm_bearing_change(i) + 0.5 * norm_speed_change(i),
                           using one-sided finite differences; normalised to [0, 1].

    Complexity: O(n)
    Weakness: Greedy scores ignore global context; may miss long flat segments.

    Args:
        trajectory: DataFrame with 'lat', 'lon' (and optionally 'timestamp')
                    or Nx2 array of (lat, lon).
        num_points: Target number of points (including endpoints).
        alpha:      Weight for geometric deviation vs. motion signal (0–1).
        indices:    If True, return selected indices instead of point array.

    Returns:
        Simplified trajectory array or list of selected indices.
    """
    points = trajectory_to_points(trajectory)
    n = len(points)

    if n <= num_points:
        idx = list(range(n))
        return idx if indices else points

    if num_points <= 2:
        idx = [0, n - 1]
        return idx if indices else points[idx]

    # ------------------------------------------------------------------
    # Geometric deviation: perpendicular distance to chord p_{i-1}→p_{i+1}
    # ------------------------------------------------------------------
    geo_dev = np.zeros(n)
    for i in range(1, n - 1):
        geo_dev[i] = point_to_line_distance(
            tuple(points[i]),
            tuple(points[i - 1]),
            tuple(points[i + 1])
        )
    max_geo = geo_dev.max()
    if max_geo > 0:
        geo_dev /= max_geo

    # ------------------------------------------------------------------
    # Motion-change signal: bearing change + speed change
    # ------------------------------------------------------------------
    bearings = np.zeros(n)
    speeds = np.zeros(n)

    if isinstance(trajectory, pd.DataFrame) and 'timestamp' in trajectory.columns:
        timestamps = pd.to_datetime(trajectory['timestamp']).values
        use_time = True
    else:
        use_time = False

    for i in range(1, n):
        lat1, lon1 = np.radians(points[i - 1])
        lat2, lon2 = np.radians(points[i])
        dlon = lon2 - lon1
        bearing = np.arctan2(
            np.sin(dlon) * np.cos(lat2),
            np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
        )
        bearings[i] = (np.degrees(bearing) + 360) % 360

        dist = haversine_distance(tuple(points[i - 1]), tuple(points[i]))
        if use_time:
            dt = (pd.Timestamp(timestamps[i]) - pd.Timestamp(timestamps[i - 1])).total_seconds()
            speeds[i] = dist / dt if dt > 0 else (speeds[i - 1] if i > 1 else 0)
        else:
            speeds[i] = dist  # proxy when no timestamps

    bearing_changes = np.zeros(n)
    speed_changes = np.zeros(n)
    for i in range(1, n - 1):
        bc = abs(bearings[i + 1] - bearings[i])
        bearing_changes[i] = min(bc, 360 - bc)
        speed_changes[i] = abs(speeds[i + 1] - speeds[i])

    max_bc = bearing_changes.max()
    max_sc = speed_changes.max()
    if max_bc > 0:
        bearing_changes /= max_bc
    if max_sc > 0:
        speed_changes /= max_sc

    motion_change = 0.5 * bearing_changes + 0.5 * speed_changes

    # ------------------------------------------------------------------
    # Combined value function
    # ------------------------------------------------------------------
    value = alpha * geo_dev + (1.0 - alpha) * motion_change

    # Endpoints are mandatory
    interior_idx = np.arange(1, n - 1)
    keep_count = num_points - 2
    top_interior = interior_idx[np.argsort(value[interior_idx])[-keep_count:]]
    selected = sorted([0] + top_interior.tolist() + [n - 1])

    if indices:
        return selected
    return points[selected]


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
        algorithm: Algorithm name ('rdp', 'sliding_window', 'vw', 'squish', 'rw', 'greedy_policy', ...)
        budget: Target number of points
        **kwargs: Additional algorithm-specific parameters
        
    Returns:
        Simplified trajectory
    """
    n = len(trajectory)
    
    if n <= budget:
        return select_points(trajectory)
    
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
        'visvalingam-whyatt': 'vw',
        'visvalingam_whyatt': 'vw',
        'vw': 'vw',
        'reumann-witkam': 'rw',
        'reumann_witkam': 'rw',
        'rw': 'rw',
        'squish': 'squish',
        'greedy_policy': 'greedy_policy',
        'greedy-policy': 'greedy_policy',
        'rl_inspired': 'greedy_policy',
        'rl': 'greedy_policy',
    }
    algorithm = algorithm_aliases.get(algorithm_key, algorithm_key)

    if algorithm == 'original':
        return select_points(trajectory)

    def search_budget_indices(search_fn: Callable[[float], List[int]]) -> List[int]:
        """Binary-search epsilon to reach the target budget point count."""
        epsilon_min, epsilon_max = BINARY_SEARCH_EPS_MIN, BINARY_SEARCH_EPS_MAX
        best_result: Optional[List[int]] = None
        last_result: List[int] = []

        for _ in range(BINARY_SEARCH_ITERATIONS):
            epsilon = (epsilon_min + epsilon_max) / 2
            result = search_fn(epsilon)
            last_result = result

            if len(result) <= budget:
                best_result = result
                epsilon_max = epsilon
            else:
                epsilon_min = epsilon

            if abs(len(result) - budget) <= BINARY_SEARCH_TOLERANCE:
                break

        return best_result if best_result is not None else last_result

    if algorithm == 'rdp':
        selected_indices = search_budget_indices(
            lambda epsilon: douglas_peucker(trajectory, epsilon, indices=True)
        )
        return select_points(trajectory, selected_indices)

    elif algorithm == 'sliding_window':
        selected_indices = search_budget_indices(
            lambda epsilon: sliding_window(trajectory, epsilon, indices=True)
        )
        return select_points(trajectory, selected_indices)

    elif algorithm == 'vw':
        return visvalingam_whyatt(trajectory, budget, indices=False)

    elif algorithm == 'squish':
        return squish(trajectory, budget, indices=False)

    elif algorithm == 'rw':
        selected_indices = search_budget_indices(
            lambda epsilon: reumann_witkam(trajectory, epsilon, indices=True)
        )
        return select_points(trajectory, selected_indices)

    elif algorithm == 'greedy_policy':
        alpha = kwargs.get('alpha', 0.5)
        return greedy_policy_simplification(trajectory, budget, alpha=alpha, indices=False)

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


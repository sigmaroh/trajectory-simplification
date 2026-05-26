"""
PHASE 2: Baseline Trajectory Simplification Algorithms

This module implements standard baseline algorithms for trajectory simplification:
1. Douglas-Peucker (RDP) - Geometric distance-based
2. Visvalingam-Whyatt (VW) - Effective-area based
3. Reumann-Witkam (RW) - Strip/corridor based
4. SQUISH - Priority-based point removal
5. Greedy Policy (RL-inspired) - Training-free proxy with fixed value function
6. RL DQN Policy - Full DQN agent (Wang et al., ICDE 2021); pre-train via:
   python -m src.algorithms.rl_policy --epochs 50
7. Uniform Sampling (US) - Fixed interval point selection
8. Adaptive Threshold (AT) - Speed-adaptive sliding window

Each algorithm has different strengths and weaknesses:
- RDP: Good for geometric preservation, but ignores temporal/speed information
- Greedy Policy: Balances geometric deviation and motion change signal
- Uniform Sampling: Simple and fast, but ignores trajectory shape
- Adaptive Threshold: Considers speed dynamics, adapts error threshold locally
"""

import heapq
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


def _ped_to_segment(points: np.ndarray, start_idx: int, end_idx: int) -> Tuple[np.ndarray, np.ndarray]:
    """Perpendicular distances for interior indices to the chord start_idx..end_idx."""
    interior = np.arange(start_idx + 1, end_idx, dtype=int)
    if len(interior) == 0:
        return interior, np.array([], dtype=float)

    lat1, lon1 = points[start_idx]
    lat2, lon2 = points[end_idx]
    lat_p = points[interior, 0]
    lon_p = points[interior, 1]

    dx = lon2 - lon1
    dy = lat2 - lat1
    denom = dx * dx + dy * dy

    if denom == 0:
        dists = np.array([
            haversine_distance((float(lat_p[i]), float(lon_p[i])), (float(lat1), float(lon1)))
            for i in range(len(interior))
        ])
        return interior, dists

    t = ((lon_p - lon1) * dx + (lat_p - lat1) * dy) / denom
    t = np.clip(t, 0.0, 1.0)
    lat_c = lat1 + t * dy
    lon_c = lon1 + t * dx

    lat1r = np.radians(lat_p)
    lon1r = np.radians(lon_p)
    lat2r = np.radians(lat_c)
    lon2r = np.radians(lon_c)
    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2) ** 2
    dists = EARTH_RADIUS_M * 2 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    return interior, dists


def _max_ped_on_segment(points: np.ndarray, start_idx: int, end_idx: int) -> Tuple[int, float]:
    """Return (index, distance) of the interior point farthest from chord start..end."""
    interior, dists = _ped_to_segment(points, start_idx, end_idx)
    if len(interior) == 0:
        return start_idx, 0.0
    k = int(np.argmax(dists))
    return int(interior[k]), float(dists[k])


def _segment_max_ped(points: np.ndarray, start_idx: int, end_idx: int) -> float:
    """Maximum perpendicular distance from interior points to chord start..end."""
    _, dists = _ped_to_segment(points, start_idx, end_idx)
    if len(dists) == 0:
        return 0.0
    return float(dists.max())


def _compute_speeds(trajectory: pd.DataFrame, n: int) -> np.ndarray:
    """Per-point speed (m/s) from timestamps; ones if timestamps are unavailable."""
    speeds = np.ones(n)
    if 'timestamp' not in trajectory.columns:
        return speeds

    ts = pd.to_datetime(trajectory['timestamp']).values.astype('datetime64[ns]')
    dt = np.diff(ts).astype('timedelta64[ms]').astype(float) / 1000.0
    dt = np.insert(dt, 0, 0.0)

    lat = np.radians(trajectory['lat'].values)
    lon = np.radians(trajectory['lon'].values)
    dlat = np.zeros(n)
    dlon = np.zeros(n)
    dlat[1:] = lat[1:] - lat[:-1]
    dlon[1:] = lon[1:] - lon[:-1]
    cos_lat = np.cos(lat)
    cos_prev = np.empty(n)
    cos_prev[0] = cos_lat[0]
    cos_prev[1:] = cos_lat[:-1]
    a = (np.sin(dlat / 2) ** 2
         + cos_prev * cos_lat * np.sin(dlon / 2) ** 2)
    dist = EARTH_RADIUS_M * 2 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    valid = dt > 0
    speeds[valid] = dist[valid] / dt[valid]
    return speeds


def douglas_peucker(trajectory: Union[pd.DataFrame, np.ndarray],
                   epsilon: float,
                   indices: bool = False) -> Union[np.ndarray, List[int]]:
    """
    Douglas-Peucker (RDP) algorithm for trajectory simplification.
    
    Algorithm:
    1. Find the point with maximum distance from line between first and last
    2. If max distance > epsilon, recursively simplify both segments
    3. Otherwise, return endpoints only
    
    Complexity: O(n log n) typical, O(n²) worst case (iterative stack, vectorised segment scans)
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

    n = len(points)
    keep = {0, n - 1}
    stack: List[Tuple[int, int]] = [(0, n - 1)]

    while stack:
        start_idx, end_idx = stack.pop()
        if end_idx - start_idx <= 1:
            continue

        max_idx, max_dist = _max_ped_on_segment(points, start_idx, end_idx)
        if max_dist > epsilon:
            keep.add(max_idx)
            stack.append((start_idx, max_idx))
            stack.append((max_idx, end_idx))

    indices_list = sorted(keep)
    
    if indices:
        return indices_list
    
    return points[indices_list]



def uniform_sampling(trajectory: Union[pd.DataFrame, np.ndarray],
                     num_points: int,
                     indices: bool = False) -> Union[np.ndarray, List[int]]:
    """
    Uniform Sampling: select points at evenly-spaced indices.

    Complexity: O(n)
    Strengths: Extremely fast, predictable compression.
    Weaknesses: Ignores trajectory shape, may skip important points.

    Args:
        trajectory: DataFrame with 'lat', 'lon' columns or array of (lat, lon)
        num_points: Target number of points to keep
        indices: If True, return indices instead of points

    Returns:
        Simplified trajectory or list of indices
    """
    points = trajectory_to_points(trajectory)
    n = len(points)

    if n <= num_points:
        idx = list(range(n))
    else:
        # Always include first and last; distribute the rest evenly
        idx = list(np.linspace(0, n - 1, num_points, dtype=int))
        idx = sorted(set(idx))

    if indices:
        return idx
    return points[idx]


def adaptive_threshold(trajectory: Union[pd.DataFrame, np.ndarray],
                       epsilon: float,
                       speed_weight: float = 0.5,
                       indices: bool = False) -> Union[np.ndarray, List[int]]:
    """
    Adaptive Threshold: sliding-window with a speed-adaptive error threshold.

    The effective tolerance at each step is scaled by local speed so that
    high-speed segments get a tighter threshold (more points kept) and
    low-speed segments get a looser one.

    Complexity: O(n) (speed-adaptive corridor, same scan pattern as Reumann-Witkam)
    Weaknesses: Sensitive to speed noise; does not explicitly preserve stops/turns.
        trajectory: DataFrame with 'lat', 'lon', and optionally 'timestamp'
        epsilon: Base error threshold (metres)
        speed_weight: How strongly speed modulates the threshold (0 = uniform)
        indices: If True, return indices instead of points

    Returns:
        Simplified trajectory or list of indices
    """
    points = trajectory_to_points(trajectory)
    n = len(points)

    if n <= 2:
        idx = list(range(n))
        return idx if indices else points[idx]

    speeds = _compute_speeds(trajectory, n) if isinstance(trajectory, pd.DataFrame) else np.ones(n)
    max_speed = speeds.max() if speeds.max() > 0 else 1.0
    norm_speeds = speeds / max_speed

    indices_list = [0]
    anchor = 0

    while anchor < n - 1:
        line_start = tuple(points[anchor])
        line_end = tuple(points[anchor + 1])
        candidate = anchor + 2
        last_inside = anchor + 1

        while candidate < n:
            local_speed = norm_speeds[candidate]
            adaptive_eps = epsilon * (1.0 - speed_weight * local_speed)
            adaptive_eps = max(adaptive_eps, epsilon * 0.1)
            dist = point_to_line_distance(tuple(points[candidate]), line_start, line_end)
            if dist <= adaptive_eps:
                last_inside = candidate
                candidate += 1
            else:
                break

        if last_inside <= anchor:
            last_inside = anchor + 1

        indices_list.append(last_inside)
        anchor = last_inside

    if indices_list[-1] != n - 1:
        indices_list.append(n - 1)

    if indices:
        return indices_list
    return points[indices_list]


def _triangle_area_xy(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """Effective triangle area for Visvalingam-Whyatt (planar lat/lon)."""
    return abs(
        p1[1] * (p2[0] - p3[0]) +
        p2[1] * (p3[0] - p1[0]) +
        p3[1] * (p1[0] - p2[0])
    ) / 2.0


def _simplify_by_min_triangle_area(points: np.ndarray, num_points: int) -> List[int]:
    """
    Remove points with the smallest effective triangle area until ``num_points`` remain.

    Shared by Visvalingam-Whyatt and SQUISH (same priority rule in this project).
    Complexity: O(n log n).
    """
    n = len(points)
    if n <= num_points:
        return list(range(n))
    if num_points <= 2:
        return [0, n - 1]

    prev_idx = np.arange(n) - 1
    next_idx = np.arange(n) + 1
    prev_idx[0] = -1
    next_idx[-1] = n
    removed = np.zeros(n, dtype=bool)
    areas = np.full(n, np.inf)

    def area_at(i: int) -> float:
        if i <= 0 or i >= n - 1:
            return np.inf
        p = prev_idx[i]
        q = next_idx[i]
        return _triangle_area_xy(points[p], points[i], points[q])

    heap: List[Tuple[float, int]] = []
    for i in range(1, n - 1):
        areas[i] = area_at(i)
        heapq.heappush(heap, (areas[i], i))

    remaining = n
    while remaining > num_points and heap:
        area, i = heapq.heappop(heap)
        if removed[i] or area != areas[i]:
            continue
        if i <= 0 or i >= n - 1:
            continue

        p = prev_idx[i]
        q = next_idx[i]
        removed[i] = True
        remaining -= 1
        next_idx[p] = q
        prev_idx[q] = p

        for j in (p, q):
            if 0 < j < n - 1 and not removed[j]:
                areas[j] = area_at(j)
                heapq.heappush(heap, (areas[j], j))

    return np.flatnonzero(~removed).tolist()


def visvalingam_whyatt(trajectory: Union[pd.DataFrame, np.ndarray],
                       num_points: int,
                       indices: bool = False) -> Union[np.ndarray, List[int]]:
    """
    Visvalingam-Whyatt simplification using effective triangle areas.

    Uses a min-heap over point areas so budget-based simplification is
    O(n log n) instead of the naive O(n²) scan-and-remove loop.

    Args:
        trajectory: DataFrame with 'lat', 'lon' columns or array of (lat, lon)
        num_points: Target number of points
        indices: If True, return indices

    Returns:
        Simplified trajectory or selected indices
    """
    points = trajectory_to_points(trajectory)
    n = len(points)
    kept = _simplify_by_min_triangle_area(points, num_points)
    if indices:
        return kept
    return points[kept]


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

    Uses the same min-area heap removal as Visvalingam-Whyatt (O(n log n)).

    Args:
        trajectory: DataFrame with 'lat', 'lon' columns or array of (lat, lon)
        num_points: Target number of points
        indices: If True, return indices

    Returns:
        Simplified trajectory or selected indices
    """
    points = trajectory_to_points(trajectory)
    kept = _simplify_by_min_triangle_area(points, num_points)
    if indices:
        return kept
    return points[kept]


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
    if n > 2:
        for i in range(1, n - 1):
            geo_dev[i] = point_to_line_distance(
                tuple(points[i]),
                tuple(points[i - 1]),
                tuple(points[i + 1])
            )
    max_geo = geo_dev.max()
    if max_geo > 0:
        geo_dev /= max_geo

    bearings = np.zeros(n)
    speeds = np.zeros(n)
    if isinstance(trajectory, pd.DataFrame):
        speeds = _compute_speeds(trajectory, n)
        use_time = 'timestamp' in trajectory.columns
    else:
        use_time = False

    lat = np.radians(points[:, 0])
    lon = np.radians(points[:, 1])
    dlon = np.zeros(n)
    dlon[1:] = lon[1:] - lon[:-1]
    lat2 = lat[1:]
    lat1 = lat[:-1]
    bearing = np.arctan2(
        np.sin(dlon[1:]) * np.cos(lat2),
        np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon[1:])
    )
    bearings[1:] = (np.degrees(bearing) + 360) % 360

    if not use_time:
        seg_dist = np.zeros(n)
        for i in range(1, n):
            seg_dist[i] = haversine_distance(tuple(points[i - 1]), tuple(points[i]))
        speeds = seg_dist

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
        algorithm: Algorithm name ('rdp', 'us', 'at', 'vw', 'squish', 'rw', 'greedy_policy', ...)
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
        'uniform_sampling': 'us',
        'uniform-sampling': 'us',
        'us': 'us',
        'adaptive_threshold': 'at',
        'adaptive-threshold': 'at',
        'at': 'at',
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
        'rl_dqn': 'rl_dqn',
        'rl': 'rl_dqn',
        'dqn': 'rl_dqn',
    }
    algorithm = algorithm_aliases.get(algorithm_key, algorithm_key)

    if algorithm == 'original':
        return select_points(trajectory)

    def _pad_indices_to_budget(sel: List[int], pts: np.ndarray) -> List[int]:
        """Pad sel to exactly `budget` by re-inserting the highest-PED excluded points."""
        sel_set = set(sel)
        n_pts = len(pts)
        if len(sel_set) >= budget:
            interior = [i for i in sorted(sel_set) if i not in (0, n_pts - 1)]
            scored = [
                (point_to_line_distance(tuple(pts[i]),
                                      tuple(pts[max(0, i - 1)]),
                                      tuple(pts[min(n_pts - 1, i + 1)])),
                 i)
                for i in interior
            ]
            keep = [j for _, j in heapq.nlargest(budget - 2, scored, key=lambda x: x[0])]
            return sorted([0, n_pts - 1] + keep)

        sel_sorted = sorted(sel_set)
        candidates: List[tuple] = []
        for a, b in zip(sel_sorted[:-1], sel_sorted[1:]):
            pa, pb = tuple(pts[a]), tuple(pts[b])
            for j in range(a + 1, b):
                if j not in sel_set:
                    d = point_to_line_distance(tuple(pts[j]), pa, pb)
                    candidates.append((d, j))

        needed = budget - len(sel_set)
        for _, j in heapq.nlargest(needed, candidates, key=lambda x: x[0]):
            sel_set.add(j)
        return sorted(sel_set)

    def search_budget_indices(search_fn: Callable[[float], List[int]]) -> List[int]:
        """Find an epsilon whose result has <= budget points, then pad/trim to exact budget."""
        pts = trajectory_to_points(trajectory)

        probe_epsilons = [
            BINARY_SEARCH_EPS_MAX,
            500.0, 200.0, 100.0, 50.0, 20.0, 10.0, 5.0, 2.0, 1.0,
            0.5, 0.2, 0.1, 0.05, 0.02, 0.01,
            BINARY_SEARCH_EPS_MIN,
        ]
        seen = set()
        best_result: Optional[List[int]] = None

        for epsilon in probe_epsilons:
            if epsilon in seen:
                continue
            seen.add(epsilon)
            result = search_fn(epsilon)
            if len(result) <= budget:
                best_result = result
                break

        if best_result is None:
            best_result = search_fn(BINARY_SEARCH_EPS_MIN)

        # Fine bisection only when still above budget after probes
        if len(best_result) > budget:
            epsilon_min, epsilon_max = BINARY_SEARCH_EPS_MIN, BINARY_SEARCH_EPS_MAX
            for _ in range(BINARY_SEARCH_ITERATIONS):
                epsilon = (epsilon_min + epsilon_max) / 2
                result = search_fn(epsilon)
                if len(result) <= budget:
                    best_result = result
                    epsilon_max = epsilon
                else:
                    epsilon_min = epsilon
                if abs(len(result) - budget) <= BINARY_SEARCH_TOLERANCE:
                    best_result = result if len(result) <= budget else best_result
                    break

        raw = best_result if best_result is not None else search_fn(BINARY_SEARCH_EPS_MIN)
        if len(raw) != budget:
            raw = _pad_indices_to_budget(raw, pts)
        return raw

    if algorithm == 'rdp':
        selected_indices = search_budget_indices(
            lambda epsilon: douglas_peucker(trajectory, epsilon, indices=True)
        )
        return select_points(trajectory, selected_indices)

    elif algorithm == 'us':
        return select_points(trajectory, uniform_sampling(trajectory, budget, indices=True))

    elif algorithm == 'at':
        speed_weight = kwargs.get('speed_weight', 0.5)
        selected_indices = search_budget_indices(
            lambda epsilon: adaptive_threshold(trajectory, epsilon,
                                              speed_weight=speed_weight, indices=True)
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

    elif algorithm == 'rl_dqn':
        from src.algorithms.rl_policy import get_or_load_model
        weights = kwargs.get('weights_path', None)
        model   = get_or_load_model(weights)
        return model.simplify(trajectory, budget, indices=False)

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
    
    vw_result = simplify_with_budget(trajectory, 'vw', budget)
    print(f"Visvalingam-Whyatt result: {len(vw_result)} points")


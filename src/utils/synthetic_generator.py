"""
PHASE 7: Synthetic Trajectory Generator

This module generates synthetic trajectories for scalability testing and
controlled experiments. Supports:
- Variable trajectory lengths
- Irregular sampling patterns
- Noise injection
- Turn patterns
- Stop regions
- Speed variations
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict
from datetime import datetime, timedelta


def generate_synthetic_trajectory(n_points: int,
                                 irregular_sampling: bool = True,
                                 noise_level: float = 0.01,
                                 include_turns: bool = True,
                                 include_stops: bool = True,
                                 speed_variation: bool = True,
                                 seed: int = None) -> pd.DataFrame:
    """
    Generate a synthetic trajectory with specified properties.
    
    Args:
        n_points: Number of points in trajectory
        irregular_sampling: Whether to use irregular sampling intervals
        noise_level: Standard deviation of noise (degrees)
        include_turns: Whether to include turn patterns
        include_stops: Whether to include stop regions
        speed_variation: Whether to vary speed
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with columns: lat, lon, timestamp
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Base trajectory: spiral or random walk
    trajectory_type = np.random.choice(['spiral', 'random_walk', 'grid'])
    
    if trajectory_type == 'spiral':
        # Spiral trajectory
        t = np.linspace(0, 4 * np.pi, n_points)
        base_lat = 40.0 + 0.1 * t * np.cos(t)
        base_lon = -74.0 + 0.1 * t * np.sin(t)
        
    elif trajectory_type == 'random_walk':
        # Random walk
        steps = np.random.randn(n_points, 2) * 0.01
        base_lat = 40.0 + np.cumsum(steps[:, 0])
        base_lon = -74.0 + np.cumsum(steps[:, 1])
        
    else:  # grid
        # Grid pattern
        side_length = int(np.sqrt(n_points))
        x = np.linspace(0, 1, side_length)
        y = np.linspace(0, 1, side_length)
        xx, yy = np.meshgrid(x, y)
        base_lat = 40.0 + xx.flatten()[:n_points] * 0.1
        base_lon = -74.0 + yy.flatten()[:n_points] * 0.1
    
    # Add turns if requested
    if include_turns:
        # Add sharp turns at random points
        n_turns = max(1, n_points // 20)
        turn_indices = np.random.choice(n_points - 2, n_turns, replace=False) + 1
        
        for turn_idx in turn_indices:
            # Create a turn by rotating the direction
            angle = np.random.uniform(30, 90)  # degrees
            angle_rad = np.radians(angle)
            
            # Compute direction vector
            if turn_idx < n_points - 1:
                dx = base_lon[turn_idx] - base_lon[turn_idx - 1]
                dy = base_lat[turn_idx] - base_lat[turn_idx - 1]
                
                # Rotate
                cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
                dx_new = dx * cos_a - dy * sin_a
                dy_new = dx * sin_a + dy * cos_a
                
                # Apply rotation to subsequent points
                for i in range(turn_idx, min(turn_idx + 10, n_points)):
                    if i < n_points - 1:
                        base_lat[i + 1] = base_lat[i] + dy_new * 0.01
                        base_lon[i + 1] = base_lon[i] + dx_new * 0.01
    
    # Add stops if requested
    if include_stops:
        n_stops = max(1, n_points // 30)
        stop_indices = np.random.choice(n_points - 10, n_stops, replace=False)
        
        for stop_idx in stop_indices:
            # Create stop region (points stay in same location)
            stop_duration = np.random.randint(5, 20)
            for i in range(stop_idx, min(stop_idx + stop_duration, n_points)):
                base_lat[i] = base_lat[stop_idx]
                base_lon[i] = base_lon[stop_idx]
    
    # Add noise
    lat = base_lat + np.random.normal(0, noise_level, n_points)
    lon = base_lon + np.random.normal(0, noise_level, n_points)
    
    # Generate timestamps
    if irregular_sampling:
        # Irregular sampling: varying intervals
        base_interval = 30  # seconds
        intervals = np.random.exponential(base_interval, n_points - 1)
        intervals = np.maximum(intervals, 1)  # Minimum 1 second
    else:
        # Regular sampling
        intervals = np.ones(n_points - 1) * 30
    
    start_time = datetime(2023, 1, 1, 0, 0, 0)
    timestamps = [start_time]
    for interval in intervals:
        timestamps.append(timestamps[-1] + timedelta(seconds=interval))
    
    # Create DataFrame
    trajectory = pd.DataFrame({
        'lat': lat,
        'lon': lon,
        'timestamp': timestamps
    })
    
    return trajectory


def generate_trajectory_batch(sizes: List[int],
                             n_per_size: int = 5,
                             **kwargs) -> List[pd.DataFrame]:
    """
    Generate a batch of synthetic trajectories of different sizes.
    
    Args:
        sizes: List of trajectory sizes to generate
        n_per_size: Number of trajectories per size
        **kwargs: Additional arguments for generate_synthetic_trajectory
        
    Returns:
        List of trajectory DataFrames
    """
    trajectories = []
    
    for size in sizes:
        for i in range(n_per_size):
            seed = hash((size, i)) % (2**32)
            traj = generate_synthetic_trajectory(size, seed=seed, **kwargs)
            trajectories.append(traj)
    
    return trajectories


def scalability_test(algorithms: List[str],
                    trajectory_sizes: List[int],
                    compression_ratio: float = 5.0,
                    n_trajectories_per_size: int = 3) -> pd.DataFrame:
    """
    Run scalability test: measure runtime vs trajectory size.
    
    Args:
        algorithms: List of algorithm names to test
        trajectory_sizes: List of trajectory sizes to test
        compression_ratio: Compression ratio to use
        n_trajectories_per_size: Number of trajectories per size
        
    Returns:
        DataFrame with scalability results
    """
    import time
    from src.algorithms.baseline_algorithms import simplify_with_budget
    from src.algorithms.proposed_method import proposed_simplification
    
    results = []
    
    print("Running scalability test...")
    print(f"  Algorithms: {algorithms}")
    print(f"  Sizes: {trajectory_sizes}")
    print(f"  Compression: {compression_ratio}x")
    
    for size in trajectory_sizes:
        print(f"\nTesting size: {size}")
        
        # Generate trajectories
        trajectories = generate_trajectory_batch(
            [size], n_per_size=n_trajectories_per_size,
            irregular_sampling=True, noise_level=0.01
        )
        
        for traj in trajectories:
            budget = max(2, int(len(traj) / compression_ratio))
            
            for algorithm in algorithms:
                try:
                    start_time = time.time()
                    
                    if algorithm == 'proposed':
                        simplified, _ = proposed_simplification(traj, budget)
                    else:
                        simplified = simplify_with_budget(traj, algorithm, budget)
                    
                    runtime = time.time() - start_time
                    
                    results.append({
                        'algorithm': algorithm,
                        'trajectory_size': size,
                        'compression_ratio': compression_ratio,
                        'runtime_seconds': runtime,
                        'throughput_points_per_sec': size / runtime if runtime > 0 else 0
                    })
                    
                except Exception as e:
                    print(f"  Error with {algorithm} on size {size}: {e}")
    
    results_df = pd.DataFrame(results)
    return results_df


if __name__ == "__main__":
    # Example: Generate synthetic trajectories
    print("Generating synthetic trajectories...")
    
    traj1 = generate_synthetic_trajectory(100, irregular_sampling=True, include_turns=True)
    print(f"Generated trajectory 1: {len(traj1)} points")
    
    traj2 = generate_synthetic_trajectory(500, irregular_sampling=True, include_stops=True)
    print(f"Generated trajectory 2: {len(traj2)} points")
    
    # Generate batch
    batch = generate_trajectory_batch([100, 200, 500], n_per_size=2)
    print(f"Generated batch: {len(batch)} trajectories")
    
    # Scalability test
    print("\nRunning scalability test...")
    scalability_results = scalability_test(
        algorithms=['rdp', 'sliding_window', 'proposed'],
        trajectory_sizes=[100, 200, 500, 1000, 2000],
        compression_ratio=5.0,
        n_trajectories_per_size=2
    )
    
    print("\nScalability Results:")
    print(scalability_results.groupby(['algorithm', 'trajectory_size'])['runtime_seconds'].mean())


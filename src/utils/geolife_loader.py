"""
PHASE 1: GeoLife GPS Dataset Loader

This module provides functions to load and preprocess the GeoLife GPS trajectory dataset.
The GeoLife dataset contains GPS trajectories collected from 182 users over 5 years.

Dataset Structure:
- Each user has a folder: Data/XXX/Trajectory/
- Each trajectory is stored in a .plt file
- Format: latitude, longitude, 0, altitude, date, time

Reference:
Zheng, Y., Li, Q., Chen, Y., Xie, X., & Ma, W. Y. (2008). Understanding mobility based on GPS data.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

from src.utils.config import (
    EARTH_RADIUS_M,
    STOP_SPEED_THRESHOLD_MS,
    TURN_THRESHOLD_DEG,
)


class GeoLifeLoader:
    """Loader for GeoLife GPS trajectory dataset."""
    
    def __init__(self, data_dir: str = "data/geolife"):
        """
        Initialize the GeoLife loader.
        
        Args:
            data_dir: Path to the GeoLife dataset root directory
        """
        self.data_dir = Path(data_dir)
        self.trajectories = []

    def load_labels(self, user_id: str) -> pd.DataFrame:
        """
        Load transportation mode labels for a user if available.

        Args:
            user_id: User ID (e.g., "000")

        Returns:
            DataFrame with columns: start, end, mode  (empty if no label file)
        """
        label_file = self.data_dir / "Data" / user_id / "labels.txt"
        if not label_file.exists():
            return pd.DataFrame(columns=['start', 'end', 'mode'])
        try:
            labels = pd.read_csv(label_file, sep='\t',
                                 names=['start', 'end', 'mode'], skiprows=1)
            labels['start'] = pd.to_datetime(labels['start'])
            labels['end'] = pd.to_datetime(labels['end'])
            labels['mode'] = labels['mode'].str.strip().str.lower()
            return labels
        except Exception:
            return pd.DataFrame(columns=['start', 'end', 'mode'])

    def is_airplane_trajectory(self, traj: pd.DataFrame,
                               labels: pd.DataFrame) -> bool:
        """
        Return True if the trajectory overlaps with any airplane-labeled segment.

        Args:
            traj: Trajectory DataFrame with a 'timestamp' column
            labels: Labels DataFrame from load_labels()

        Returns:
            True if the trajectory should be excluded as airplane data
        """
        airplane_labels = labels[labels['mode'] == 'airplane']
        if airplane_labels.empty:
            return False
        traj_start = traj['timestamp'].min()
        traj_end = traj['timestamp'].max()
        for _, row in airplane_labels.iterrows():
            # Overlap check: two intervals overlap when start_A < end_B and end_A > start_B
            if traj_start < row['end'] and traj_end > row['start']:
                return True
        return False

    @staticmethod
    def is_airplane_by_speed(traj: pd.DataFrame,
                             speed_threshold_ms: float = 80.0) -> bool:
        """
        Fallback: return True if the trajectory's mean speed exceeds
        speed_threshold_ms (default 80 m/s ≈ 288 km/h), which is above
        any realistic ground vehicle speed in this dataset.

        Args:
            traj: Trajectory DataFrame
            speed_threshold_ms: Mean-speed cutoff in m/s

        Returns:
            True if the trajectory should be excluded as airplane data
        """
        if len(traj) < 2:
            return False
        lat = np.radians(traj['lat'].values)
        lon = np.radians(traj['lon'].values)
        dlat, dlon = np.diff(lat), np.diff(lon)
        a = np.sin(dlat / 2) ** 2 + np.cos(lat[:-1]) * np.cos(lat[1:]) * np.sin(dlon / 2) ** 2
        distances = 6371000 * 2 * np.arcsin(np.sqrt(a))
        timestamps = pd.to_datetime(traj['timestamp'])
        dt = timestamps.diff().dt.total_seconds().dropna().values
        dt = np.where(dt < 1, 1, dt)          # avoid div-by-zero
        speeds = distances / dt
        mean_speed = np.mean(speeds)
        return mean_speed > speed_threshold_ms

    def load_plt_file(self, filepath: str) -> pd.DataFrame:
        """
        Load a single .plt trajectory file.
        
        Format:
        - Columns: latitude, longitude, 0, altitude, date (YYYY-MM-DD), time (HH:MM:SS)
        - First 6 lines are metadata (skip)
        
        Args:
            filepath: Path to .plt file
            
        Returns:
            DataFrame with columns: lat, lon, alt, timestamp
        """
        try:
            # Skip first 6 lines (metadata)
            # GeoLife .plt rows contain 7 columns:
            # lat, lon, 0, altitude, date_days, date, time
            df = pd.read_csv(
                filepath,
                skiprows=6,
                header=None,
                names=['lat', 'lon', 'zero', 'alt', 'date_days', 'date', 'time']
            )
            
            # Remove zero column
            df = df.drop(columns=['zero', 'date_days'])
            
            # Combine date and time into timestamp
            df['timestamp'] = pd.to_datetime(df['date'] + ' ' + df['time'])
            df = df.drop(columns=['date', 'time'])
            
            # Sort by timestamp
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Remove duplicates
            df = df.drop_duplicates(subset=['timestamp']).reset_index(drop=True)
            
            return df
            
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return pd.DataFrame()
    
    def load_user_trajectories(self, user_id: str,
                               exclude_airplane: bool = True) -> List[pd.DataFrame]:
        """
        Load all trajectories for a specific user.

        Args:
            user_id: User ID (e.g., "000", "001")
            exclude_airplane: If True, skip trajectories identified as airplane
                              trips via label files (primary) or mean-speed
                              heuristic (fallback for unlabelled users).

        Returns:
            List of trajectory DataFrames
        """
        user_dir = self.data_dir / "Data" / user_id / "Trajectory"

        if not user_dir.exists():
            print(f"User directory not found: {user_dir}")
            return []

        labels = self.load_labels(user_id) if exclude_airplane else pd.DataFrame()
        has_labels = not labels.empty

        trajectories = []
        plt_files = sorted(user_dir.glob("*.plt"))

        for plt_file in plt_files:
            traj = self.load_plt_file(str(plt_file))
            if traj.empty or len(traj) <= 10:
                continue

            if exclude_airplane:
                if has_labels:
                    if self.is_airplane_trajectory(traj, labels):
                        continue
                else:
                    if self.is_airplane_by_speed(traj):
                        continue

            traj['user_id'] = user_id
            traj['file_id'] = plt_file.stem
            trajectories.append(traj)

        return trajectories
    
    def load_all_trajectories(self, max_users: int = None,
                             min_points: int = 50,
                             exclude_airplane: bool = True,
                             user_ids: list = None) -> List[pd.DataFrame]:
        """
        Load trajectories from all (or selected) users.

        Args:
            max_users:        Maximum number of users to load (None for all).
                              Ignored when user_ids is specified.
            min_points:       Minimum number of points per trajectory.
            exclude_airplane: Filter out airplane trajectories via label files
                              or mean-speed heuristic.
            user_ids:         Optional list of specific user IDs to load
                              (e.g. ['000', '010', '050']).  When provided,
                              max_users is ignored.

        Returns:
            List of trajectory DataFrames
        """
        data_path = self.data_dir / "Data"

        if not data_path.exists():
            print(f"Data directory not found: {data_path}")
            print("Please download GeoLife dataset and extract to data/geolife/")
            return []

        user_dirs = sorted([d for d in data_path.iterdir() if d.is_dir()])

        if user_ids is not None:
            # Normalise to zero-padded 3-digit strings
            target = set(uid.zfill(3) for uid in user_ids)
            user_dirs = [d for d in user_dirs if d.name in target]
            missing = target - {d.name for d in user_dirs}
            if missing:
                print(f"  Warning: user IDs not found in dataset: {sorted(missing)}")
        elif max_users:
            user_dirs = user_dirs[:max_users]

        all_trajectories = []

        for user_dir in user_dirs:
            user_id = user_dir.name
            trajectories = self.load_user_trajectories(user_id,
                                                       exclude_airplane=exclude_airplane)
            trajectories = [t for t in trajectories if len(t) >= min_points]
            all_trajectories.extend(trajectories)

        print(f"Loaded {len(all_trajectories)} trajectories from {len(user_dirs)} users")
        return all_trajectories


def compute_trajectory_properties(trajectory: pd.DataFrame) -> Dict:
    """
    Compute trajectory properties: sampling intervals, speed, direction changes.
    
    Args:
        trajectory: DataFrame with columns: lat, lon, timestamp
        
    Returns:
        Dictionary with computed properties
    """
    if len(trajectory) < 2:
        return {}
    
    # Convert lat/lon to radians for distance calculation
    lat_rad = np.radians(trajectory['lat'].values)
    lon_rad = np.radians(trajectory['lon'].values)
    
    # Compute time intervals (in seconds)
    timestamps = pd.to_datetime(trajectory['timestamp'])
    time_diffs = timestamps.diff().dt.total_seconds().fillna(0)
    
    # Compute distances using Haversine formula
    dlat = np.diff(lat_rad)
    dlon = np.diff(lon_rad)
    a = np.sin(dlat/2)**2 + np.cos(lat_rad[:-1]) * np.cos(lat_rad[1:]) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    distances = EARTH_RADIUS_M * c
    
    # Compute speeds (m/s)
    speeds = distances / (time_diffs[1:].values + 1e-6)  # Avoid division by zero
    speeds = np.concatenate([[0], speeds])  # First point has no speed
    
    # Compute directions (bearing in degrees)
    bearings = np.degrees(np.arctan2(
        np.sin(dlon) * np.cos(lat_rad[1:]),
        np.cos(lat_rad[:-1]) * np.sin(lat_rad[1:]) - 
        np.sin(lat_rad[:-1]) * np.cos(lat_rad[1:]) * np.cos(dlon)
    ))
    bearings = (bearings + 360) % 360  # Normalize to [0, 360]
    bearings = np.concatenate([[bearings[0]], bearings])  # First point uses first bearing
    
    # Compute direction changes (angular difference)
    direction_changes = np.abs(np.diff(bearings))
    direction_changes = np.minimum(direction_changes, 360 - direction_changes)  # Handle wrap-around
    direction_changes = np.concatenate([[0], direction_changes])
    
    # Identify stops (speed < threshold, e.g., 1 m/s ≈ 3.6 km/h)
    is_stop = speeds < STOP_SPEED_THRESHOLD_MS

    # Identify turns (direction change > threshold, e.g., 30 degrees)
    is_turn = direction_changes > TURN_THRESHOLD_DEG
    
    properties = {
        'num_points': len(trajectory),
        'time_intervals': time_diffs.values,
        'mean_interval': np.mean(time_diffs[1:].values),
        'std_interval': np.std(time_diffs[1:].values),
        'cv_interval': np.std(time_diffs[1:].values) / (np.mean(time_diffs[1:].values) + 1e-6),  # Coefficient of variation
        'total_distance': np.sum(distances),
        'distances': distances,
        'speeds': speeds,
        'mean_speed': np.mean(speeds[1:]),
        'max_speed': np.max(speeds),
        'bearings': bearings,
        'direction_changes': direction_changes,
        'mean_direction_change': np.mean(direction_changes[1:]),
        'num_stops': np.sum(is_stop),
        'stop_ratio': np.mean(is_stop),
        'num_turns': np.sum(is_turn),
        'turn_ratio': np.mean(is_turn),
        'duration': (timestamps.iloc[-1] - timestamps.iloc[0]).total_seconds()
    }
    
    return properties


def preprocess_trajectory(trajectory: pd.DataFrame, 
                         remove_outliers: bool = True,
                         max_speed: float = 200.0) -> pd.DataFrame:
    """
    Preprocess trajectory: remove outliers, filter unrealistic speeds.
    
    Args:
        trajectory: Raw trajectory DataFrame
        remove_outliers: Whether to remove outliers
        max_speed: Maximum allowed speed (m/s), default 200 m/s ≈ 720 km/h
        
    Returns:
        Cleaned trajectory DataFrame
    """
    traj = trajectory.copy()
    
    if len(traj) < 2:
        return traj
    
    # Compute properties
    props = compute_trajectory_properties(traj)
    
    if 'speeds' in props:
        # Remove points with unrealistic speeds
        valid_mask = props['speeds'] <= max_speed
        traj = traj[valid_mask].reset_index(drop=True)
    
    if remove_outliers and len(traj) > 2:
        # Remove spatial outliers (points too far from neighbors)
        lat_rad = np.radians(traj['lat'].values)
        lon_rad = np.radians(traj['lon'].values)
        
        # Compute distances to neighbors
        dlat = np.diff(lat_rad)
        dlon = np.diff(lon_rad)
        a = np.sin(dlat/2)**2 + np.cos(lat_rad[:-1]) * np.cos(lat_rad[1:]) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        distances = EARTH_RADIUS_M * c
        
        # Remove points with very large jumps (outliers)
        if len(distances) > 0:
            median_dist = np.median(distances)
            mad = np.median(np.abs(distances - median_dist))  # Median Absolute Deviation
            threshold = median_dist + 5 * mad  # 5 MAD rule
            
            valid_mask = np.ones(len(traj), dtype=bool)
            valid_mask[1:] = distances <= threshold
            traj = traj[valid_mask].reset_index(drop=True)
    
    return traj


if __name__ == "__main__":
    # Example usage
    loader = GeoLifeLoader("data/geolife")
    trajectories = loader.load_all_trajectories(max_users=5, min_points=100)
    
    if trajectories:
        print(f"\nLoaded {len(trajectories)} trajectories")
        
        # Analyze first trajectory
        traj = trajectories[0]
        props = compute_trajectory_properties(traj)
        
        print(f"\nTrajectory properties:")
        print(f"  Points: {props['num_points']}")
        print(f"  Duration: {props['duration']:.1f} seconds")
        print(f"  Total distance: {props['total_distance']:.1f} meters")
        print(f"  Mean speed: {props['mean_speed']:.2f} m/s")
        print(f"  Sampling irregularity (CV): {props['cv_interval']:.3f}")
        print(f"  Stop ratio: {props['stop_ratio']:.3f}")
        print(f"  Turn ratio: {props['turn_ratio']:.3f}")


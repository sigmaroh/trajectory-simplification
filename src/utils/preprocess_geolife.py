"""
Preprocessing script for GeoLife dataset.
Run this to preprocess and save trajectories for experiments.
"""

import sys
import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from typing import List

from src.utils.geolife_loader import GeoLifeLoader, compute_trajectory_properties, preprocess_trajectory


def trajectories_to_points_dataframe(trajectories: List[pd.DataFrame]) -> pd.DataFrame:
    """Flatten a list of trajectory DataFrames into one long-format table."""
    parts = []
    for traj_id, traj in enumerate(trajectories):
        df = traj.copy()
        df["trajectory_id"] = traj_id
        if "user_id" not in df.columns:
            df["user_id"] = "unknown"
        if "file_id" not in df.columns:
            df["file_id"] = "unknown"
        df["point_index"] = np.arange(len(df), dtype=int)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")
        preferred = [
            "trajectory_id", "user_id", "file_id", "point_index",
            "lat", "lon", "alt", "timestamp",
        ]
        cols = [c for c in preferred if c in df.columns]
        parts.append(df[cols])
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def save_trajectories_csv(
    trajectories: List[pd.DataFrame],
    output_dir: Path,
    points_filename: str = "trajectories_points.csv",
    index_filename: str = "trajectories_index.csv",
) -> None:
    """Write processed trajectories to CSV (points + per-trajectory index)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    points_df = trajectories_to_points_dataframe(trajectories)
    points_path = output_dir / points_filename
    points_df.to_csv(points_path, index=False)

    index_rows = []
    for traj_id, traj in enumerate(trajectories):
        index_rows.append({
            "trajectory_id": traj_id,
            "user_id": traj["user_id"].iloc[0] if "user_id" in traj.columns else "unknown",
            "file_id": traj["file_id"].iloc[0] if "file_id" in traj.columns else "unknown",
            "num_points": len(traj),
        })
    index_df = pd.DataFrame(index_rows)
    index_path = output_dir / index_filename
    index_df.to_csv(index_path, index=False)

    print(f"  {points_path} ({len(points_df):,} rows)")
    print(f"  {index_path} ({len(index_df):,} trajectories)")


def export_pickle_to_csv(
    pickle_path: str = "data/processed/trajectories.pkl",
    output_dir: str = "data/processed",
) -> None:
    """Export an existing trajectories.pkl to CSV without re-running preprocessing."""
    pickle_path = Path(pickle_path)
    with open(pickle_path, "rb") as f:
        trajectories = pickle.load(f)
    print(f"Loaded {len(trajectories)} trajectories from {pickle_path}")
    print(f"Writing CSV to {output_dir}/...")
    save_trajectories_csv(trajectories, Path(output_dir))


def main():
    """Preprocess GeoLife dataset and save to disk."""
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess GeoLife GPS trajectories")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/geolife",
        help="Path to GeoLife root (must contain Data/000/, Data/001/, ...)",
    )
    parser.add_argument(
        "--max-users",
        type=int,
        default=50,
        help="Maximum number of user folders to load (first N by sorted user id). "
             "Use 0 or omit limit by passing a very large number; see --all-users.",
    )
    parser.add_argument(
        "--all-users",
        action="store_true",
        help="Load every user in the dataset (ignores --max-users).",
    )
    parser.add_argument(
        "--min-points",
        type=int,
        default=100,
        help="Minimum points per trajectory when loading from .plt files",
    )
    parser.add_argument(
        "--min-points-after-clean",
        type=int,
        default=50,
        help="Minimum points required after outlier removal to keep a trajectory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Directory for trajectories.pkl and CSV outputs",
    )
    parser.add_argument(
        "--export-csv-only",
        type=str,
        default=None,
        metavar="PICKLE",
        help="Only convert an existing .pkl to CSV (path to pickle); skips loading GeoLife",
    )
    args = parser.parse_args()

    if args.export_csv_only:
        export_pickle_to_csv(args.export_csv_only, args.output_dir)
        return

    max_users = None if args.all_users else args.max_users
    if max_users == 0:
        max_users = None

    print("=" * 60)
    print("GeoLife Dataset Preprocessing")
    print("=" * 60)
    
    # Initialize loader
    loader = GeoLifeLoader(args.data_dir)
    
    # Load trajectories (airplane trips excluded via label files + speed heuristic)
    print("\nLoading trajectories (excluding airplane trips)...")
    if max_users is None:
        print("  max_users: all users")
    else:
        print(f"  max_users: {max_users}")
    trajectories = loader.load_all_trajectories(
        max_users=max_users, min_points=args.min_points,
        exclude_airplane=True
    )
    
    if not trajectories:
        print("ERROR: No trajectories loaded. Please check dataset path.")
        return
    
    print(f"Loaded {len(trajectories)} trajectories")
    
    # Preprocess each trajectory
    print("\nPreprocessing trajectories...")
    processed_trajectories = []
    trajectory_properties = []
    
    for i, traj in enumerate(trajectories):
        if (i + 1) % 10 == 0:
            print(f"  Processing {i + 1}/{len(trajectories)}...")
        
        # Preprocess
        cleaned_traj = preprocess_trajectory(traj, remove_outliers=True)
        
        if len(cleaned_traj) >= args.min_points_after_clean:
            # Compute properties
            props = compute_trajectory_properties(cleaned_traj)
            props['trajectory_id'] = i
            props['user_id'] = traj['user_id'].iloc[0] if 'user_id' in traj.columns else 'unknown'
            props['file_id'] = traj['file_id'].iloc[0] if 'file_id' in traj.columns else 'unknown'
            
            processed_trajectories.append(cleaned_traj)
            trajectory_properties.append(props)
    
    print(f"\nProcessed {len(processed_trajectories)} valid trajectories")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving to {output_dir} (pickle + CSV)...")

    with open(output_dir / "trajectories.pkl", "wb") as f:
        pickle.dump(processed_trajectories, f)
    print(f"  Pickle: {output_dir / 'trajectories.pkl'}")

    props_df = pd.DataFrame(trajectory_properties)
    props_df.to_csv(output_dir / "trajectory_properties.csv", index=False)
    print(f"  CSV:    {output_dir / 'trajectory_properties.csv'}")

    save_trajectories_csv(processed_trajectories, output_dir)
    
    print(f"\nSaved {len(processed_trajectories)} trajectories")
    print(f"Properties summary:")
    print(f"  Mean points: {props_df['num_points'].mean():.1f}")
    print(f"  Mean duration: {props_df['duration'].mean():.1f} seconds")
    print(f"  Mean distance: {props_df['total_distance'].mean():.1f} meters")
    print(f"  Mean sampling CV: {props_df['cv_interval'].mean():.3f}")
    
    print("\nPreprocessing complete!")


if __name__ == "__main__":
    main()


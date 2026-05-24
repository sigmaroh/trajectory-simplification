"""
Preprocessing script for GeoLife dataset.
Run this to preprocess and save trajectories for experiments.
"""

import sys
import os
import pickle
import pandas as pd
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.geolife_loader import GeoLifeLoader, compute_trajectory_properties, preprocess_trajectory


def main():
    """Preprocess GeoLife dataset and save to disk."""
    
    print("=" * 60)
    print("GeoLife Dataset Preprocessing")
    print("=" * 60)
    
    # Initialize loader
    loader = GeoLifeLoader("data/geolife")
    
    # Load trajectories (airplane trips excluded via label files + speed heuristic)
    print("\nLoading trajectories (excluding airplane trips)...")
    trajectories = loader.load_all_trajectories(max_users=182, min_points=100,
                                                exclude_airplane=True)
    
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
        
        if len(cleaned_traj) >= 50:  # Keep only trajectories with sufficient points
            # Compute properties
            props = compute_trajectory_properties(cleaned_traj)
            props['trajectory_id'] = i
            props['user_id'] = traj['user_id'].iloc[0] if 'user_id' in traj.columns else 'unknown'
            props['file_id'] = traj['file_id'].iloc[0] if 'file_id' in traj.columns else 'unknown'
            
            processed_trajectories.append(cleaned_traj)
            trajectory_properties.append(props)
    
    print(f"\nProcessed {len(processed_trajectories)} valid trajectories")
    
    # Save processed trajectories
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving to {output_dir}...")
    
    # Save trajectories as pickle
    with open(output_dir / "trajectories.pkl", "wb") as f:
        pickle.dump(processed_trajectories, f)
    
    # Save properties as DataFrame
    props_df = pd.DataFrame(trajectory_properties)
    props_df.to_csv(output_dir / "trajectory_properties.csv", index=False)
    
    print(f"\nSaved {len(processed_trajectories)} trajectories")
    print(f"Properties summary:")
    print(f"  Mean points: {props_df['num_points'].mean():.1f}")
    print(f"  Mean duration: {props_df['duration'].mean():.1f} seconds")
    print(f"  Mean distance: {props_df['total_distance'].mean():.1f} meters")
    print(f"  Mean sampling CV: {props_df['cv_interval'].mean():.3f}")
    
    print("\nPreprocessing complete!")


if __name__ == "__main__":
    main()


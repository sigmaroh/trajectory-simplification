"""
Generate dataset characterisation plots from real GeoLife preprocessed data.

Produces:
  results/figures/dataset_length_distribution.png
  results/figures/dataset_sampling_irregularity.png
  results/figures/dataset_speed.png
  results/figures/dataset_turns_stops.png

Usage:
    python src/experiments/generate_dataset_plots.py
    python src/experiments/generate_dataset_plots.py --data-file data/processed/trajectories.pkl
"""

import sys, os, argparse, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.config import (
    EARTH_RADIUS_M,
    STOP_SPEED_THRESHOLD_MS,
    TURN_THRESHOLD_DEG,
    MAX_VALID_INTERVAL_S,
    MAX_VALID_SPEED_MS,
)

sns.set_theme(style='whitegrid', palette='muted', font_scale=1.1)


def haversine_batch(pts: np.ndarray) -> np.ndarray:
    """Return step-lengths (metres) between consecutive rows of an (N, 2) lat/lon array.

    Uses a fully vectorised Haversine calculation — much faster than calling
    ``haversine_distance`` in a loop for large trajectories.

    Args:
        pts: (N, 2) numpy array with columns [latitude_deg, longitude_deg].

    Returns:
        (N-1,) array of distances in metres.
    """
    la1, lo1 = np.radians(pts[:-1, 0]), np.radians(pts[:-1, 1])
    la2, lo2 = np.radians(pts[1:, 0]),  np.radians(pts[1:, 1])
    dlat, dlon = la2 - la1, lo2 - lo1
    a = np.sin(dlat / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin(dlon / 2) ** 2
    return EARTH_RADIUS_M * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def compute_stats(trajectories: list, max_traj: int = 500) -> dict:
    """Compute dataset-wide statistics from a list of trajectory DataFrames.

    Args:
        trajectories: List of trajectory DataFrames with at least 'lat', 'lon'
                      columns and optionally 'timestamp'.
        max_traj:     Maximum number of trajectories to process (for speed).

    Returns:
        Dictionary with keys: n_pts, all_ivs, cvs, dists, durs,
        all_spds, stop_pcts, turn_pcts.
    """
    trajs = trajectories[:max_traj]

    # --- Length ---
    n_pts = np.array([len(t) for t in trajs])

    # --- Intervals & irregularity ---
    all_ivs, cvs = [], []
    for t in trajs:
        if 'timestamp' not in t.columns:
            continue
        ts = pd.to_datetime(t['timestamp'])
        iv = ts.diff().dt.total_seconds().dropna().values
        iv = iv[(iv > 0) & (iv < MAX_VALID_INTERVAL_S)]
        if len(iv) > 1:
            all_ivs.extend(iv.tolist())
            cvs.append(float(iv.std() / iv.mean()))
    all_ivs = np.array(all_ivs)
    cvs = np.array(cvs)

    # --- Distance & duration ---
    dists, durs = [], []
    for t in trajs:
        pts = t[['lat', 'lon']].values
        dists.append(haversine_batch(pts).sum() / 1000)
        if 'timestamp' in t.columns:
            ts = pd.to_datetime(t['timestamp'])
            durs.append((ts.iloc[-1] - ts.iloc[0]).total_seconds() / 60)
    dists, durs = np.array(dists), np.array(durs)

    # --- Speed & stops ---
    all_spds, stop_pcts = [], []
    for t in trajs:
        if 'timestamp' not in t.columns:
            continue
        pts = t[['lat', 'lon']].values
        ts = pd.to_datetime(t['timestamp']).values
        dist_m = haversine_batch(pts)
        dt = np.array(
            [(pd.Timestamp(ts[i]) - pd.Timestamp(ts[i - 1])).total_seconds()
             for i in range(1, len(pts))]
        )
        mask = dt > 0
        spd = np.where(mask, dist_m / np.where(mask, dt, 1), 0)
        clean = spd[spd < MAX_VALID_SPEED_MS]
        all_spds.extend(clean.tolist())
        stop_pcts.append(float(100 * (spd < STOP_SPEED_THRESHOLD_MS).mean()))
    all_spds = np.array(all_spds)

    # --- Turns ---
    turn_pcts = []
    for t in trajs:
        pts = t[['lat', 'lon']].values
        if len(pts) < 3:
            continue
        la1, lo1 = np.radians(pts[:-1, 0]), np.radians(pts[:-1, 1])
        la2, lo2 = np.radians(pts[1:, 0]),  np.radians(pts[1:, 1])
        dlo = lo2 - lo1
        b = np.degrees(np.arctan2(
            np.sin(dlo) * np.cos(la2),
            np.cos(la1) * np.sin(la2) - np.sin(la1) * np.cos(la2) * np.cos(dlo)
        ))
        b = (b + 360) % 360
        dc = np.abs(np.diff(b))
        dc = np.minimum(dc, 360 - dc)
        turn_pcts.append(float(100 * (dc >= TURN_THRESHOLD_DEG).mean()))
    turn_pcts = np.array(turn_pcts)

    return dict(
        n_pts=n_pts, all_ivs=all_ivs, cvs=cvs,
        dists=dists, durs=durs,
        all_spds=all_spds, stop_pcts=np.array(stop_pcts),
        turn_pcts=turn_pcts,
    )


def plot_length_distribution(s, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].hist(s['n_pts'], bins=40, color='steelblue', edgecolor='white')
    axes[0].axvline(np.median(s['n_pts']), color='red', linestyle='--',
                    label=f"Median = {int(np.median(s['n_pts']))}")
    axes[0].set_xlabel('Number of Points')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Trajectory Length (points)')
    axes[0].legend()

    clip = np.percentile(s['dists'], 99)
    axes[1].hist(s['dists'][s['dists'] <= clip], bins=40, color='coral', edgecolor='white')
    axes[1].axvline(np.median(s['dists']), color='red', linestyle='--',
                    label=f"Median = {np.median(s['dists']):.1f} km")
    axes[1].set_xlabel('Total Distance (km)')
    axes[1].set_title('Trajectory Distance')
    axes[1].legend()

    if len(s['durs']):
        clip2 = np.percentile(s['durs'], 99)
        axes[2].hist(s['durs'][s['durs'] <= clip2], bins=40, color='seagreen', edgecolor='white')
        axes[2].axvline(np.median(s['durs']), color='red', linestyle='--',
                        label=f"Median = {np.median(s['durs']):.0f} min")
        axes[2].set_xlabel('Duration (minutes)')
        axes[2].set_title('Trajectory Duration')
        axes[2].legend()

    plt.suptitle('GeoLife Dataset – Trajectory Length Distribution', fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = out_dir / 'dataset_length_distribution.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved {path}')


def plot_sampling_irregularity(s, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    clip = np.percentile(s['all_ivs'], 99)
    axes[0].hist(s['all_ivs'][s['all_ivs'] <= clip], bins=60,
                 color='steelblue', edgecolor='white')
    axes[0].axvline(np.median(s['all_ivs']), color='red', linestyle='--',
                    label=f"Median = {np.median(s['all_ivs']):.1f} s")
    axes[0].set_xlabel('Inter-point Interval (seconds)')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Sampling Interval Distribution (99th pct clip)')
    axes[0].legend()

    axes[1].hist(s['cvs'], bins=40, color='coral', edgecolor='white')
    axes[1].axvline(1.0, color='red', linestyle='--', label='CV = 1.0 (irregular threshold)')
    axes[1].set_xlabel('Coefficient of Variation (CV)')
    axes[1].set_title(f'Sampling Irregularity per Trajectory\n'
                      f'{100*(s["cvs"]>1.0).mean():.1f}% have CV > 1.0  (mean CV = {s["cvs"].mean():.2f})')
    axes[1].legend()

    plt.suptitle('GeoLife Dataset – Sampling Irregularity', fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = out_dir / 'dataset_sampling_irregularity.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved {path}')


def plot_speed(s, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    clip = np.percentile(s['all_spds'], 99)
    axes[0].hist(s['all_spds'][s['all_spds'] <= clip], bins=60,
                 color='steelblue', edgecolor='white')
    axes[0].axvline(np.median(s['all_spds']), color='red', linestyle='--',
                    label=f"Median = {np.median(s['all_spds']):.2f} m/s")
    axes[0].axvline(1.0, color='orange', linestyle=':', label='Stop threshold (1 m/s)')
    axes[0].set_xlabel('Speed (m/s)')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Speed Distribution')
    axes[0].legend()

    axes[1].hist(s['stop_pcts'], bins=40, color='tomato', edgecolor='white')
    axes[1].axvline(np.mean(s['stop_pcts']), color='red', linestyle='--',
                    label=f"Mean = {np.mean(s['stop_pcts']):.1f}%")
    axes[1].set_xlabel('Stop Points per Trajectory (%)')
    axes[1].set_title('Fraction of Stop Points (speed < 1 m/s)\nper Trajectory')
    axes[1].legend()

    plt.suptitle('GeoLife Dataset – Speed & Stop Statistics', fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = out_dir / 'dataset_speed.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved {path}')


def plot_turns_stops(s, out_dir):
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.hist(s['turn_pcts'], bins=40, color='mediumpurple', edgecolor='white', alpha=0.85,
            label='Turn points (direction change ≥ 30°)')
    ax.hist(s['stop_pcts'], bins=40, color='tomato', edgecolor='white', alpha=0.65,
            label='Stop points (speed < 1 m/s)')
    ax.axvline(np.mean(s['turn_pcts']), color='purple', linestyle='--',
               label=f"Mean turns = {np.mean(s['turn_pcts']):.1f}%")
    ax.axvline(np.mean(s['stop_pcts']), color='red', linestyle='--',
               label=f"Mean stops = {np.mean(s['stop_pcts']):.1f}%")
    ax.set_xlabel('Fraction of Points (%)')
    ax.set_ylabel('Number of Trajectories')
    ax.set_title('GeoLife Dataset – Turn and Stop Point Distribution per Trajectory')
    ax.legend()
    plt.tight_layout()

    path = out_dir / 'dataset_turns_stops.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved {path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-file', default='data/processed/trajectories.pkl')
    parser.add_argument('--output-dir', default='results/figures')
    parser.add_argument('--max-trajectories', type=int, default=500)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'Loading trajectories from {args.data_file} ...')
    with open(args.data_file, 'rb') as f:
        trajectories = pickle.load(f)
    print(f'  Loaded {len(trajectories)} trajectories. Computing stats on first {args.max_trajectories}...')

    s = compute_stats(trajectories, max_traj=args.max_trajectories)

    print('Generating dataset plots...')
    plot_length_distribution(s, out_dir)
    plot_sampling_irregularity(s, out_dir)
    plot_speed(s, out_dir)
    plot_turns_stops(s, out_dir)
    print('Done.')


if __name__ == '__main__':
    main()

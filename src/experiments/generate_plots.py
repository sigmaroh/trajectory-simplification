"""
PHASE 6: Visualization Code

This module generates all plots for the project:
- Original vs simplified trajectories
- Compression vs error curves
- Runtime vs size
- Metric comparison charts
"""

import sys
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.algorithms.baseline_algorithms import simplify_with_budget
from src.algorithms.proposed_method import proposed_simplification
from src.utils.config import PLOT_DPI, PLOT_FIGSIZE, PLOT_FONT_SIZE

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = list(PLOT_FIGSIZE)
plt.rcParams['font.size'] = PLOT_FONT_SIZE


def plot_trajectory_comparison(trajectory: pd.DataFrame,
                               algorithms: List[str],
                               compression_ratio: float,
                               output_path: str = None):
    """
    Plot original vs simplified trajectories for multiple algorithms.
    
    Args:
        trajectory: Original trajectory DataFrame
        algorithms: List of algorithm names to compare
        compression_ratio: Compression ratio to use
        output_path: Path to save figure
    """
    budget = max(2, int(len(trajectory) / compression_ratio))
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # Plot original
    ax = axes[0]
    ax.plot(trajectory['lon'], trajectory['lat'], 'b-', linewidth=2, alpha=0.7, label='Original')
    ax.scatter(trajectory['lon'].iloc[0], trajectory['lat'].iloc[0], 
              color='green', s=100, marker='o', label='Start', zorder=5)
    ax.scatter(trajectory['lon'].iloc[-1], trajectory['lat'].iloc[-1], 
              color='red', s=100, marker='s', label='End', zorder=5)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(f'Original Trajectory\n({len(trajectory)} points)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot each algorithm
    algorithm_names = {
        'none': 'None',
        'dp': 'Douglas-Peucker',
        'rdp': 'Douglas-Peucker',
        'sw': 'Sliding Window',
        'sliding_window': 'Sliding Window',
        'vw': 'Visvalingam-Whyatt',
        'rw': 'Reumann-Witkam',
        'squish': 'SQUISH',
        'greedy_policy': 'Greedy Policy (RL-inspired)',
        'proposed': 'Proposed Method'
    }
    
    for idx, algorithm in enumerate(algorithms[:5], 1):
        ax = axes[idx]
        
        try:
            if algorithm == 'proposed':
                simplified, indices = proposed_simplification(trajectory, budget)
            else:
                simplified = simplify_with_budget(trajectory, algorithm, budget)
                indices = None
            
            ax.plot(trajectory['lon'], trajectory['lat'], 'lightgray', 
                   linewidth=1, alpha=0.3, label='Original')
            ax.plot(simplified[:, 1], simplified[:, 0], 'r-', 
                   linewidth=2, alpha=0.8, label='Simplified')
            ax.scatter(simplified[:, 1], simplified[:, 0], 
                      color='red', s=50, marker='o', zorder=5)
            ax.scatter(trajectory['lon'].iloc[0], trajectory['lat'].iloc[0], 
                      color='green', s=100, marker='o', zorder=6)
            ax.scatter(trajectory['lon'].iloc[-1], trajectory['lat'].iloc[-1], 
                      color='blue', s=100, marker='s', zorder=6)
            
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            ax.set_title(f'{algorithm_names.get(algorithm, algorithm)}\n({len(simplified)} points)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{algorithm_names.get(algorithm, algorithm)}\n(Error)')
    
    # Hide unused subplots
    for idx in range(len(algorithms) + 1, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved trajectory comparison to {output_path}")
    
    plt.show()


def plot_compression_error_curves(results_df: pd.DataFrame,
                                 output_path: str = None):
    """
    Plot compression ratio vs error curves for different algorithms.
    
    Args:
        results_df: Results DataFrame from experiments
        output_path: Path to save figure
    """
    metric_specs = [
        ('hausdorff_distance', 'Hausdorff Distance (m)'),
        ('average_pte',        'Average PTE (m)'),
        ('frechet_distance',   'Fréchet Distance (m)'),
        ('ped',                'PED (m)'),
        ('sed',                'SED (m)'),
        ('dad',                'DAD (degrees)'),
        ('sad',                'SAD (m/s)'),
        ('issd',               'ISSD (m·s)'),
        ('turn_preservation',  'Turn Preservation (0–1)'),
        ('stop_preservation',  'Stop Preservation (0–1)'),
        ('runtime_seconds',    'Runtime (seconds)'),
    ]
    metrics = [(metric, label) for metric, label in metric_specs if metric in results_df.columns]
    if not metrics:
        print("No available metrics found for compression-error curves.")
        return

    n_cols = 3
    n_rows = int(np.ceil(len(metrics) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4.5 * n_rows))
    axes = np.atleast_1d(axes).flatten()
    
    # Snap compression ratios to nearest target for clean x-axis
    TARGET_CRS = [2.0, 5.0, 10.0]
    tol = 0.6

    def snap_cr(v):
        return min(TARGET_CRS, key=lambda cr: abs(v - cr))

    df_plot = results_df.copy()
    df_plot = df_plot[df_plot['compression_ratio'].apply(
        lambda v: any(abs(v - cr) <= tol for cr in TARGET_CRS))]
    df_plot['cr_label'] = df_plot['compression_ratio'].apply(snap_cr)

    x_pos    = {cr: i for i, cr in enumerate(sorted(TARGET_CRS))}
    x_labels = [f"{int(cr)}×" for cr in sorted(TARGET_CRS)]

    algo_order  = ['dp','vw','squish','rw','greedy_policy','proposed']
    algo_colors = {
        'dp':           '#FF6600',   # orange
        'vw':           '#2ECC71',   # green
        'squish':       '#E91E63',   # pink
        'rw':           '#9B59B6',   # purple
        'greedy_policy':'#FF5722',   # deep orange
        'proposed':     '#212121',   # near-black
    }

    for idx, (metric, ylabel) in enumerate(metrics):
        ax = axes[idx]
        present = [a for a in algo_order if a in df_plot['algorithm'].unique()]
        for algorithm in present:
            algo_data = df_plot[df_plot['algorithm'] == algorithm]
            grouped   = algo_data.groupby('cr_label')[metric].agg(['mean', 'std'])
            if grouped.empty:
                continue
            xs = [x_pos[cr] for cr in grouped.index]
            ax.errorbar(xs, grouped['mean'], yerr=grouped['std'],
                        label=algorithm, marker='o',
                        color=algo_colors.get(algorithm, None),
                        linewidth=2, capsize=5, markersize=7)

        ax.set_xlabel('Compression Ratio')
        ax.set_ylabel(ylabel)
        ax.set_title(f'{ylabel} vs Compression Ratio')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(len(TARGET_CRS)))
        ax.set_xticklabels(x_labels, fontsize=10)
        if metric not in ('runtime_seconds', 'dad', 'sad',
                          'turn_preservation', 'stop_preservation'):
            ax.set_yscale('log')

    for idx in range(len(metrics), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved compression-error curves to {output_path}")
    
    plt.show()


def plot_runtime_scalability(results_df: pd.DataFrame,
                            output_path: str = None):
    """
    Plot runtime vs trajectory size for scalability analysis.
    
    Args:
        results_df: Results DataFrame
        output_path: Path to save figure
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    for algorithm in results_df['algorithm'].unique():
        algo_data = results_df[results_df['algorithm'] == algorithm]
        
        # Group by trajectory size
        grouped = algo_data.groupby('trajectory_size')['runtime_seconds'].agg(['mean', 'std'])
        
        ax.errorbar(grouped.index, grouped['mean'],
                   yerr=grouped['std'],
                   label=algorithm, marker='o', linewidth=2, capsize=5)
    
    ax.set_xlabel('Trajectory Size (points)')
    ax.set_ylabel('Runtime (seconds)')
    ax.set_title('Runtime Scalability')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved runtime scalability plot to {output_path}")
    
    plt.show()


def plot_metric_comparison(results_df: pd.DataFrame,
                          compression_ratio: float,
                          output_path: str = None):
    """
    Plot metric comparison bar chart for different algorithms.
    
    Args:
        results_df: Results DataFrame
        compression_ratio: Specific compression ratio to compare
        output_path: Path to save figure
    """
    # Compression ratios are often not exact due to integer budgets.
    # Select the nearest available ratio per algorithm/trajectory group.
    ratio_series = pd.to_numeric(results_df['compression_ratio'], errors='coerce')
    valid = results_df[ratio_series.notna()].copy()
    valid['compression_ratio'] = ratio_series[ratio_series.notna()]

    group_cols = ['algorithm']
    if 'trajectory_id' in valid.columns:
        group_cols.append('trajectory_id')

    nearest_rows = []
    tolerance = max(0.5, 0.25 * compression_ratio)
    for _, group in valid.groupby(group_cols, dropna=False):
        nearest_idx = (group['compression_ratio'] - compression_ratio).abs().idxmin()
        nearest_row = valid.loc[nearest_idx]
        if abs(float(nearest_row['compression_ratio']) - compression_ratio) <= tolerance:
            nearest_rows.append(nearest_row)

    filtered = pd.DataFrame(nearest_rows)
    if filtered.empty:
        print(f"No data available to compare near compression ratio {compression_ratio}x.")
        return
    
    # Select metrics to compare
    metrics = [
        'hausdorff_distance', 'average_pte', 'frechet_distance',
        'ped', 'dad', 'sed', 'sad', 'issd',
        'turn_preservation', 'stop_preservation'
    ]
    
    # Filter to available metrics
    available_metrics = [m for m in metrics if m in filtered.columns]
    
    fig, axes = plt.subplots(1, len(available_metrics), figsize=(5*len(available_metrics), 6))
    
    if len(available_metrics) == 1:
        axes = [axes]
    
    for idx, metric in enumerate(available_metrics):
        ax = axes[idx]
        
        grouped = filtered.groupby('algorithm')[metric].agg(['mean', 'std'])
        
        x_pos = np.arange(len(grouped))
        ax.bar(x_pos, grouped['mean'], yerr=grouped['std'], 
              capsize=5, alpha=0.7, edgecolor='black')
        
        ax.set_xlabel('Algorithm')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'{metric.replace("_", " ").title()}\n(Target compression: {compression_ratio}x)')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(grouped.index, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved metric comparison to {output_path}")
    
    plt.show()


def plot_per_metric_pages(results_df: pd.DataFrame,
                          output_dir: str = None,
                          compression_ratios: list = None):
    """
    Save one PNG per metric — each shows all algorithms across compression ratios.

    Args:
        results_df: Results DataFrame
        output_dir: Directory to save individual metric figures
        compression_ratios: List of CRs to include (None = all in data)
    """
    metric_specs = [
        ('hausdorff_distance', 'Hausdorff Distance (m)'),
        ('average_pte',        'Average PTE (m)'),
        ('frechet_distance',   'Fréchet Distance (m)'),
        ('ped',                'PED (m)'),
        ('sed',                'SED (m)'),
        ('dad',                'DAD (degrees)'),
        ('sad',                'SAD (m/s)'),
        ('issd',               'ISSD (m·s)'),
        ('turn_preservation',  'Turn Preservation (0–1)'),
        ('stop_preservation',  'Stop Preservation (0–1)'),
        ('runtime_seconds',    'Runtime (seconds)'),
    ]
    metrics = [(m, l) for m, l in metric_specs if m in results_df.columns]
    if not metrics:
        print("No metrics available for per-metric pages.")
        return

    out = Path(output_dir) if output_dir else None
    if out:
        per_metric_dir = out / "per_metric"
        per_metric_dir.mkdir(parents=True, exist_ok=True)

    # Snap each row's compression_ratio to the nearest target CR label
    TARGET_CRS = compression_ratios if compression_ratios else [2.0, 5.0, 10.0]
    tol = 0.6

    df = results_df.copy()
    if compression_ratios:
        mask = df['compression_ratio'].apply(
            lambda v: any(abs(v - cr) <= tol for cr in compression_ratios))
        df = df[mask].copy()

    def snap(v):
        diffs = [(abs(v - cr), cr) for cr in TARGET_CRS]
        return min(diffs)[1]

    df['cr_label'] = df['compression_ratio'].apply(snap)

    algo_order = ['dp', 'vw', 'squish', 'rw', 'greedy_policy', 'proposed']
    algo_colors = {
        'dp':           '#FF6600',
        'vw':           '#2ECC71',
        'squish':       '#E91E63',
        'rw':           '#9B59B6',
        'greedy_policy':'#FF5722',
        'proposed':     '#212121',
    }

    x_positions = {cr: i for i, cr in enumerate(sorted(TARGET_CRS))}
    x_labels    = [f"{int(cr)}×" for cr in sorted(TARGET_CRS)]

    for metric, ylabel in metrics:
        fig, ax = plt.subplots(figsize=(10, 5))
        present = [a for a in algo_order if a in df['algorithm'].unique()]
        for algo in present:
            adf = df[df['algorithm'] == algo]
            grouped = adf.groupby('cr_label')[metric].agg(['mean', 'std'])
            if grouped.empty:
                continue
            xs = [x_positions[cr] for cr in grouped.index]
            ax.errorbar(xs, grouped['mean'], yerr=grouped['std'],
                        label=algo, marker='o',
                        color=algo_colors.get(algo, None),
                        linewidth=2, capsize=5, markersize=7)

        ax.set_xlabel('Compression Ratio')
        ax.set_ylabel(ylabel)
        ax.set_title(f'{ylabel} vs Compression Ratio\n(mean ± std across trajectories)')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(len(TARGET_CRS)))
        ax.set_xticklabels(x_labels, fontsize=11)
        if metric not in ('runtime_seconds', 'dad', 'sad',
                          'turn_preservation', 'stop_preservation'):
            ax.set_yscale('log')

        plt.tight_layout()
        if out:
            fname = per_metric_dir / f"metric_{metric}.png"
            plt.savefig(fname, dpi=200, bbox_inches='tight')
            print(f"  Saved {fname}")
        plt.close()


def generate_all_plots(results_file: str = "results/experiment_results.csv",
                      trajectories_file: str = "data/processed/trajectories.pkl",
                      output_dir: str = "results/figures"):
    """
    Generate all plots for the project.

    Produces:
    - trajectory_comparison.png
    - compression_error_curves.png         (all metrics, single combined page)
    - compression_error_curves_2x.png      (combined, 2× CR only)
    - compression_error_curves_5x.png      (combined, 5× CR only)
    - compression_error_curves_10x.png     (combined, 10× CR only)
    - per_metric/metric_<name>.png         (one page per metric, all CRs)
    - runtime_scalability.png
    - metric_comparison_<cr>x.png          (bar charts at each CR)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    results_df = pd.read_csv(results_file)

    with open(trajectories_file, 'rb') as f:
        trajectories = pickle.load(f)

    # Use only 2×, 5×, 10× CRs; drop 20× if present
    KEEP_CRS = [2.0, 5.0, 10.0]
    tol = 0.6
    cr_mask = results_df['compression_ratio'].apply(
        lambda v: any(abs(v - cr) <= tol for cr in KEEP_CRS))
    results_3cr = results_df[cr_mask].copy()

    print(f"Loaded {len(trajectories)} trajectories, {len(results_df)} total rows, "
          f"{len(results_3cr)} rows after filtering to 2×/5×/10×")
    print(f"Algorithms: {sorted(results_3cr['algorithm'].unique())}")

    print("\nGenerating plots...")

    # 1. Trajectory comparison
    print("1. Trajectory comparison (DP + VW + RW + Greedy Policy + Proposed)...")
    sample_traj = sorted(trajectories[:60], key=len)[5]
    plot_trajectory_comparison(
        sample_traj,
        ['dp', 'vw', 'rw', 'greedy_policy', 'proposed'],
        compression_ratio=5.0,
        output_path=str(output_dir / "trajectory_comparison.png")
    )

    # 2a. Combined compression-error curves (all CRs)
    print("2a. Compression-error curves (combined, all 3 CRs)...")
    plot_compression_error_curves(
        results_3cr,
        output_path=str(output_dir / "compression_error_curves.png")
    )

    # 2b. Per-CR combined pages
    for cr in KEEP_CRS:
        print(f"2b. Combined page at {int(cr)}×...")
        sub = results_3cr[abs(results_3cr['compression_ratio'] - cr) <= tol]
        if not sub.empty:
            plot_compression_error_curves(
                sub,
                output_path=str(output_dir / f"compression_error_curves_{int(cr)}x.png")
            )

    # 3. Per-metric separate pages
    print("3. Per-metric separate pages → results/figures/per_metric/...")
    plot_per_metric_pages(results_3cr, output_dir=str(output_dir),
                          compression_ratios=KEEP_CRS)

    # 4. Runtime scalability
    print("4. Runtime scalability...")
    plot_runtime_scalability(
        results_3cr,
        output_path=str(output_dir / "runtime_scalability.png")
    )

    # 5. Metric comparison bar charts per CR
    print("5. Metric comparison bar charts...")
    for cr in KEEP_CRS:
        plot_metric_comparison(
            results_3cr,
            compression_ratio=cr,
            output_path=str(output_dir / f"metric_comparison_{int(cr)}x.png")
        )

    print(f"\nAll plots saved to {output_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate visualization plots')
    parser.add_argument('--results-file', type=str,
                       default='results/experiment_results.csv',
                       help='Path to experiment results CSV')
    parser.add_argument('--trajectories-file', type=str,
                       default='data/processed/trajectories.pkl',
                       help='Path to trajectories pickle file')
    parser.add_argument('--output-dir', type=str,
                       default='results/figures',
                       help='Directory to save plots')
    
    args = parser.parse_args()
    
    generate_all_plots(
        results_file=args.results_file,
        trajectories_file=args.trajectories_file,
        output_dir=args.output_dir
    )


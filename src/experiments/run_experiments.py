"""
PHASE 5: Experiment Pipeline

This module provides a comprehensive experiment runner that:
- Tests multiple algorithms on multiple trajectories
- Evaluates at different compression ratios
- Measures runtime and memory usage
- Generates results tables automatically
- Supports batch evaluation
"""

import sys
import os
import time
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm
import tracemalloc

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.algorithms.baseline_algorithms import simplify_with_budget
from src.algorithms.proposed_method import proposed_simplification
from src.metrics.evaluation_metrics import compute_all_metrics


class ExperimentRunner:
    """Runner for trajectory simplification experiments."""
    
    def __init__(self, trajectories: List[pd.DataFrame], output_dir: str = "results"):
        """
        Initialize experiment runner.
        
        Args:
            trajectories: List of trajectory DataFrames
            output_dir: Directory to save results
        """
        self.trajectories = trajectories
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results = []
        
    def run_single_experiment(self,
                             trajectory: pd.DataFrame,
                             algorithm: str,
                             compression_ratio: float,
                             algorithm_params: Dict = None) -> Dict:
        """
        Run a single experiment: simplify one trajectory with one algorithm.
        
        Args:
            trajectory: Input trajectory
            algorithm: Algorithm name ('original', 'dp', 'squish', 'vw', 'sw', 'rw', 'proposed')
            compression_ratio: Target compression ratio (e.g., 5.0 means 5x compression)
            algorithm_params: Additional parameters for algorithm
            
        Returns:
            Dictionary with results including metrics, runtime, memory
        """
        if algorithm_params is None:
            algorithm_params = {}
        
        # Compute budget
        budget = max(2, int(len(trajectory) / compression_ratio))
        
        # Measure runtime and memory
        tracemalloc.start()
        start_time = time.time()
        
        try:
            # Run algorithm
            if algorithm == 'proposed':
                simplified, indices = proposed_simplification(
                    trajectory, budget, **algorithm_params
                )
            else:
                simplified = simplify_with_budget(
                    trajectory, algorithm, budget, **algorithm_params
                )
                # For baseline algorithms, we need to find indices
                # This is approximate - in practice, you'd track indices during simplification
                indices = None
            
            end_time = time.time()
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            runtime = end_time - start_time
            memory_mb = peak / 1024 / 1024  # Convert to MB
            
            # Compute metrics
            metrics = compute_all_metrics(trajectory, simplified, indices)
            
            # Throughput: trajectories processed per second
            throughput = 1.0 / runtime if runtime > 0 else float('inf')

            # Add experiment metadata
            result = {
                'algorithm': algorithm,
                'compression_ratio': compression_ratio,
                'budget': budget,
                'runtime_seconds': runtime,
                'memory_mb': memory_mb,
                'throughput_traj_per_sec': throughput,
                **metrics
            }
            
            return result
            
        except Exception as e:
            tracemalloc.stop()
            print(f"Error in experiment: {e}")
            return {
                'algorithm': algorithm,
                'compression_ratio': compression_ratio,
                'error': str(e)
            }
    
    def run_batch_experiments(self,
                             algorithms: List[str],
                             compression_ratios: List[float],
                             max_trajectories: int = None,
                             algorithm_params: Dict[str, Dict] = None) -> pd.DataFrame:
        """
        Run batch experiments on multiple trajectories.
        
        Args:
            algorithms: List of algorithm names to test
            compression_ratios: List of compression ratios to test
            max_trajectories: Maximum number of trajectories to test (None for all)
            algorithm_params: Dictionary mapping algorithm names to parameter dicts
            
        Returns:
            DataFrame with all results
        """
        if algorithm_params is None:
            algorithm_params = {}
        
        trajectories_to_test = self.trajectories
        if max_trajectories:
            trajectories_to_test = trajectories_to_test[:max_trajectories]
        
        total_experiments = len(trajectories_to_test) * len(algorithms) * len(compression_ratios)
        
        print(f"Running {total_experiments} experiments...")
        print(f"  Trajectories: {len(trajectories_to_test)}")
        print(f"  Algorithms: {algorithms}")
        print(f"  Compression ratios: {compression_ratios}")
        
        results = []
        
        with tqdm(total=total_experiments, desc="Experiments") as pbar:
            for traj_idx, trajectory in enumerate(trajectories_to_test):
                for algorithm in algorithms:
                    params = algorithm_params.get(algorithm, {})
                    
                    for comp_ratio in compression_ratios:
                        result = self.run_single_experiment(
                            trajectory, algorithm, comp_ratio, params
                        )
                        result['trajectory_id'] = traj_idx
                        result['trajectory_size'] = len(trajectory)
                        results.append(result)
                        
                        pbar.update(1)
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        # Save results
        results_df.to_csv(self.output_dir / "experiment_results.csv", index=False)
        
        print(f"\nExperiments complete! Results saved to {self.output_dir / 'experiment_results.csv'}")
        
        return results_df
    
    def generate_summary_table(self, results_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Generate summary table with aggregated statistics.
        
        Args:
            results_df: Results DataFrame (if None, load from file)
            
        Returns:
            Summary DataFrame with mean/std metrics per algorithm and compression ratio
        """
        if results_df is None:
            results_df = pd.read_csv(self.output_dir / "experiment_results.csv")
        
        # Group by algorithm and compression ratio
        summary_cols = [
            'hausdorff_distance', 'average_pte', 'frechet_distance',
            'ped', 'dad', 'sed', 'sad', 'issd',
            'turn_preservation', 'stop_preservation',
            'runtime_seconds', 'memory_mb', 'throughput_traj_per_sec'
        ]
        
        # Filter to available columns
        available_cols = [col for col in summary_cols if col in results_df.columns]
        
        summary = results_df.groupby(['algorithm', 'compression_ratio'])[available_cols].agg(['mean', 'std'])
        summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
        summary = summary.reset_index()
        
        # Save summary
        summary.to_csv(self.output_dir / "summary_table.csv", index=False)
        
        print(f"Summary table saved to {self.output_dir / 'summary_table.csv'}")
        
        return summary


def main():
    """Main experiment runner."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run trajectory simplification experiments')
    parser.add_argument('--max-trajectories', type=int, default=20,
                       help='Maximum number of trajectories to test')
    parser.add_argument('--compression-ratios', type=float, nargs='+',
                       default=[2.0, 5.0, 10.0, 20.0],
                       help='Compression ratios to test')
    parser.add_argument('--algorithms', type=str, nargs='+',
                       default=['original', 'dp', 'squish', 'vw', 'sw', 'rw', 'greedy_policy', 'proposed'],
                       help='Algorithms to test')
    parser.add_argument('--data-file', type=str,
                       default='data/processed/trajectories.pkl',
                       help='Path to preprocessed trajectories file')
    
    args = parser.parse_args()
    
    # Load trajectories
    print("Loading trajectories...")
    with open(args.data_file, 'rb') as f:
        trajectories = pickle.load(f)
    
    print(f"Loaded {len(trajectories)} trajectories")
    
    # Initialize runner
    runner = ExperimentRunner(trajectories, output_dir="results")
    
    # Algorithm parameters
    algorithm_params = {
        'greedy_policy': {'alpha': 0.5},
        'proposed': {'weights': {'turn': 0.3, 'stop': 0.3, 'speed': 0.2, 'irregular': 0.2}}
    }
    
    # Run experiments
    results_df = runner.run_batch_experiments(
        algorithms=args.algorithms,
        compression_ratios=args.compression_ratios,
        max_trajectories=args.max_trajectories,
        algorithm_params=algorithm_params
    )
    
    # Generate summary
    summary = runner.generate_summary_table(results_df)
    
    print("\n" + "="*60)
    print("Experiment Summary")
    print("="*60)
    print(summary.to_string())
    
    return results_df, summary


if __name__ == "__main__":
    results_df, summary = main()


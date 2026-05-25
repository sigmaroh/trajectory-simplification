"""
Large-scale efficiency / scalability experiments.

Measures runtime, peak memory (tracemalloc), and throughput for algorithms
across increasing trajectory sizes — either synthetic or the longest real
GeoLife trips in trajectories.pkl.

Examples::

    # Synthetic sizes 1k–20k (fast algorithms only)
    python -m src.experiments.run_scalability --mode synthetic \\
        --sizes 1000 2000 5000 10000 20000 \\
        --algorithms vw rw greedy_policy proposed

    # Top 10 longest real trajectories (min 5000 points)
    python -m src.experiments.run_scalability --mode geolife \\
        --min-points 5000 --top-n 10 \\
        --algorithms dp vw rw greedy_policy proposed
"""

from __future__ import annotations

import argparse
import pickle
import time
import tracemalloc
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.algorithms.baseline_algorithms import simplify_with_budget
from src.algorithms.proposed_method import proposed_simplification
from src.utils.config import DATASET
from src.utils.synthetic_generator import generate_synthetic_trajectory


def _run_one(
    trajectory: pd.DataFrame,
    algorithm: str,
    compression_ratio: float,
    algorithm_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    algorithm_params = algorithm_params or {}
    n = len(trajectory)
    budget = max(2, int(n / compression_ratio))

    tracemalloc.start()
    t0 = time.time()
    try:
        if algorithm == "proposed":
            proposed_simplification(trajectory, budget, **algorithm_params)
        else:
            simplify_with_budget(trajectory, algorithm, budget, **algorithm_params)
        runtime = time.time() - t0
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    return {
        "trajectory_size": n,
        "budget": budget,
        "runtime_seconds": runtime,
        "memory_mb": peak / 1024 / 1024,
        "throughput_traj_per_sec": 1.0 / runtime if runtime > 0 else float("inf"),
        "throughput_points_per_sec": n / runtime if runtime > 0 else float("inf"),
    }


def load_longest_geolife(
    pickle_path: Path,
    top_n: int,
    min_points: int,
) -> List[pd.DataFrame]:
    with pickle_path.open("rb") as f:
        trajectories = pickle.load(f)

    eligible = [t for t in trajectories if isinstance(t, pd.DataFrame) and len(t) >= min_points]
    eligible.sort(key=len, reverse=True)
    return eligible[:top_n]


def run_synthetic(
    sizes: List[int],
    algorithms: List[str],
    compression_ratio: float,
    repeats: int,
    algorithm_params: Dict[str, Dict],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for size in sizes:
        for rep in range(repeats):
            traj = generate_synthetic_trajectory(
                size,
                irregular_sampling=True,
                noise_level=0.01,
                include_turns=True,
                include_stops=True,
                seed=42 + size * 100 + rep,
            )
            for algo in algorithms:
                row = _run_one(traj, algo, compression_ratio, algorithm_params.get(algo))
                row.update(
                    {
                        "mode": "synthetic",
                        "algorithm": algo,
                        "compression_ratio": compression_ratio,
                        "repeat": rep,
                    }
                )
                rows.append(row)
                print(
                    f"  synthetic n={size} rep={rep} {algo}: "
                    f"{row['runtime_seconds']:.3f}s mem={row['memory_mb']:.3f}MB"
                )
    return pd.DataFrame(rows)


def run_geolife(
    pickle_path: Path,
    top_n: int,
    min_points: int,
    algorithms: List[str],
    compression_ratio: float,
    algorithm_params: Dict[str, Dict],
) -> pd.DataFrame:
    trajectories = load_longest_geolife(pickle_path, top_n, min_points)
    if not trajectories:
        raise ValueError(f"No trajectories with >= {min_points} points in {pickle_path}")

    rows: List[Dict[str, Any]] = []
    for i, traj in enumerate(trajectories):
        uid = traj["user_id"].iloc[0] if "user_id" in traj.columns else "?"
        fid = traj["file_id"].iloc[0] if "file_id" in traj.columns else "?"
        for algo in algorithms:
            row = _run_one(traj, algo, compression_ratio, algorithm_params.get(algo))
            row.update(
                {
                    "mode": "geolife",
                    "algorithm": algo,
                    "compression_ratio": compression_ratio,
                    "trajectory_rank": i,
                    "user_id": uid,
                    "file_id": fid,
                }
            )
            rows.append(row)
            print(
                f"  geolife #{i} n={len(traj)} user={uid} {algo}: "
                f"{row['runtime_seconds']:.3f}s mem={row['memory_mb']:.3f}MB"
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Large-scale scalability benchmark")
    parser.add_argument("--mode", choices=["synthetic", "geolife"], required=True)
    parser.add_argument("--pickle", default=DATASET["data_file"])
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=["rw", "greedy_policy", "vw", "proposed"],
    )
    parser.add_argument("--compression-ratio", type=float, default=5.0)
    parser.add_argument("--sizes", type=int, nargs="+", default=[1000, 2000, 5000, 10000, 20000])
    parser.add_argument("--repeats", type=int, default=2, help="Repeats per synthetic size")
    parser.add_argument("--top-n", type=int, default=10, help="Longest real trajectories")
    parser.add_argument("--min-points", type=int, default=5000, help="Min length for geolife mode")
    parser.add_argument(
        "--output",
        default="results/scalability_results.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    algorithm_params = {
        "greedy_policy": {"alpha": 0.5},
        "proposed": {},
    }

    print(f"Mode: {args.mode}  algorithms: {args.algorithms}  CR: {args.compression_ratio}x")

    if args.mode == "synthetic":
        df = run_synthetic(
            args.sizes,
            args.algorithms,
            args.compression_ratio,
            args.repeats,
            algorithm_params,
        )
    else:
        df = run_geolife(
            Path(args.pickle),
            args.top_n,
            args.min_points,
            args.algorithms,
            args.compression_ratio,
            algorithm_params,
        )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nSaved {len(df)} rows to {out}")

    summary = df.groupby(["algorithm", "trajectory_size"])[
        ["runtime_seconds", "memory_mb", "throughput_traj_per_sec"]
    ].mean()
    print("\nMean by algorithm and size:")
    print(summary.to_string())


if __name__ == "__main__":
    main()

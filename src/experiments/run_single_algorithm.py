"""
Run one algorithm on every trajectory in trajectories.pkl (sequentially).

Same measurements as ``run_experiments.py`` (size, budget, runtime, memory,
throughput, metrics) but only for a single algorithm — useful for full-dataset
runs without the full algorithm matrix.

Examples::

    # All trajectories, four compression ratios (like main experiments)
    python -m src.experiments.run_single_algorithm \\
        --algorithm proposed \\
        --compression-ratios 2.0 5.0 10.0 20.0

    # One ratio, first 50 trajectories
    python -m src.experiments.run_single_algorithm \\
        -a vw --compression-ratios 5.0 --max-trajectories 50
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import pandas as pd
from tqdm import tqdm

from src.experiments.run_experiments import ExperimentRunner
from src.utils.config import DATASET, EXPERIMENTS
from src.utils.trajectory_stream import normalize_user_id


def filter_trajectories(
    trajectories: List[pd.DataFrame],
    *,
    user_ids: Optional[Sequence[Union[str, int]]] = None,
    min_points: Optional[int] = None,
    max_trajectories: Optional[int] = None,
) -> List[pd.DataFrame]:
    """Subset trajectories by user, minimum length, and count cap."""
    out = trajectories

    if user_ids is not None:
        allowed = {normalize_user_id(u) for u in user_ids}
        out = [
            t
            for t in out
            if isinstance(t, pd.DataFrame)
            and "user_id" in t.columns
            and normalize_user_id(t["user_id"].iloc[0]) in allowed
        ]

    if min_points is not None:
        out = [t for t in out if len(t) >= min_points]

    if max_trajectories is not None:
        out = out[:max_trajectories]

    return out


def default_algorithm_params(algorithm: str) -> Dict[str, Any]:
    """Kwargs for ``run_single_experiment`` / ``proposed_simplification``."""
    params_map = EXPERIMENTS.get("algorithm_params", {})
    if algorithm == "greedy_policy":
        return dict(params_map.get("greedy_policy", {"alpha": 0.5}))
    if algorithm == "proposed":
        return dict(params_map.get("proposed", {}))
    return {}


def run_algorithm_on_all_trajectories(
    algorithm: str,
    trajectories: List[pd.DataFrame],
    compression_ratios: Sequence[float],
    *,
    algorithm_params: Optional[Dict[str, Any]] = None,
    output_dir: Union[str, Path] = "results",
    output_csv: Optional[Union[str, Path]] = None,
    show_progress: bool = True,
    write_summary: bool = True,
) -> pd.DataFrame:
    """
    Run one algorithm on each trajectory, for each compression ratio.

    Each row includes: ``trajectory_size`` / ``input_points``, ``output_points``,
    ``budget``, ``compression_ratio``, ``actual_compression_ratio``,
    ``runtime_seconds``, ``memory_mb``, ``throughput_traj_per_sec``, plus metrics.
    """
    if not trajectories:
        raise ValueError("No trajectories to process")

    algorithm_params = algorithm_params if algorithm_params is not None else default_algorithm_params(algorithm)
    runner = ExperimentRunner(trajectories, output_dir=str(output_dir))
    rows: List[Dict[str, Any]] = []

    traj_iter = enumerate(trajectories)
    if show_progress:
        traj_iter = enumerate(
            tqdm(trajectories, desc=algorithm, unit="traj"),
            start=0,
        )

    for traj_idx, trajectory in traj_iter:
        meta: Dict[str, Any] = {
            "trajectory_id": traj_idx,
            "trajectory_size": len(trajectory),
        }
        if "user_id" in trajectory.columns and len(trajectory):
            meta["user_id"] = str(trajectory["user_id"].iloc[0])
        if "file_id" in trajectory.columns and len(trajectory):
            meta["file_id"] = str(trajectory["file_id"].iloc[0])

        for comp_ratio in compression_ratios:
            result = runner.run_single_experiment(
                trajectory,
                algorithm,
                comp_ratio,
                algorithm_params,
            )
            result.update(meta)
            rows.append(result)

    results_df = pd.DataFrame(rows)

    out_path = Path(output_csv) if output_csv else Path(output_dir) / f"single_{algorithm}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_path, index=False)

    if write_summary and len(results_df) > 0 and "runtime_seconds" in results_df.columns:
        preferred = [
            "input_points", "output_points", "actual_compression_ratio",
            "runtime_seconds", "memory_mb", "throughput_traj_per_sec",
            "hausdorff_distance", "turn_preservation", "stop_preservation",
        ]
        # keep only preferred cols that exist AND are numeric (no tuples)
        avail = [
            c for c in preferred
            if c in results_df.columns
            and pd.api.types.is_numeric_dtype(results_df[c])
        ]
        if avail:
            summary = (
                results_df.groupby("compression_ratio")[avail]
                .agg(["mean", "std"])
                .reset_index()
            )
            summary.columns = ["_".join(c).strip("_") for c in summary.columns.values]
            summary_path = out_path.with_name(f"{out_path.stem}_summary.csv")
            summary.to_csv(summary_path, index=False)

    return results_df


def main() -> None:
    default_crs = EXPERIMENTS.get("compression_ratios", [2.0, 5.0, 10.0, 20.0])

    parser = argparse.ArgumentParser(
        description="Run one algorithm on all trajectories (like run_experiments, single algo).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--algorithm",
        "-a",
        required=True,
        help="dp, vw, squish, rw, greedy_policy, proposed, …",
    )
    parser.add_argument(
        "--data-file",
        default=DATASET["data_file"],
        help="Path to trajectories.pkl",
    )
    parser.add_argument(
        "--compression-ratios",
        "--compression-ratio",
        dest="compression_ratios",
        type=float,
        nargs="+",
        default=default_crs,
        help=f"Target compression ratios (default: {default_crs})",
    )
    parser.add_argument(
        "--max-trajectories",
        type=int,
        default=None,
        help="Cap number of trajectories (default: all in pickle)",
    )
    parser.add_argument(
        "--users",
        nargs="+",
        default=None,
        help="Only these user ids, e.g. 000 001",
    )
    parser.add_argument(
        "--min-points",
        type=int,
        default=None,
        help="Skip trajectories shorter than this",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="CSV path (default: results/single_<algorithm>.csv)",
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Skip writing *_summary.csv",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
    )
    args = parser.parse_args()

    data_path = Path(args.data_file)
    if not data_path.is_file():
        raise SystemExit(f"Missing {data_path}. Run: python src/utils/preprocess_geolife.py")

    print(f"Loading {data_path} ...")
    with data_path.open("rb") as f:
        all_trajectories = pickle.load(f)
    print(f"  Loaded {len(all_trajectories)} trajectories")

    trajectories = filter_trajectories(
        all_trajectories,
        user_ids=args.users,
        min_points=args.min_points,
        max_trajectories=args.max_trajectories,
    )
    n_traj = len(trajectories)
    n_exp = n_traj * len(args.compression_ratios)
    print(f"  Running {args.algorithm} on {n_traj} trajectories × {len(args.compression_ratios)} CRs = {n_exp} runs")

    if not trajectories:
        raise SystemExit("No trajectories match filters.")

    results_df = run_algorithm_on_all_trajectories(
        args.algorithm,
        trajectories,
        args.compression_ratios,
        algorithm_params=default_algorithm_params(args.algorithm),
        output_dir=args.output_dir,
        output_csv=args.output,
        show_progress=not args.quiet,
        write_summary=not args.no_summary,
    )

    out = Path(args.output) if args.output else Path(args.output_dir) / f"single_{args.algorithm}.csv"
    n_err = int(results_df["error"].notna().sum()) if "error" in results_df.columns else 0
    print(f"\nDone: {len(results_df)} rows, {n_err} errors")
    print(f"  CSV: {out}")
    if not args.no_summary:
        print(f"  Summary: {out.with_name(out.stem + '_summary.csv')}")

    if "runtime_seconds" in results_df.columns:
        ok = results_df[results_df["error"].isna()] if "error" in results_df.columns else results_df
        print(
            f"  Mean runtime: {ok['runtime_seconds'].mean():.3f}s  "
            f"Mean input size: {ok['input_points'].mean():.0f} pts  "
            f"Mean output size: {ok['output_points'].mean():.0f} pts"
        )


if __name__ == "__main__":
    main()

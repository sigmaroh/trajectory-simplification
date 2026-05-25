"""
Stream trajectories from ``data/processed/trajectories.pkl`` by user id.

Simulates an online feed by replaying stored GeoLife trips one point at a time.
Use ``TrajectoryPickleStreamer`` for raw GPS events; use ``stream_squish_for_users``
to pipe those events through streaming SQUISH per trajectory.

Example::

    from src.utils.trajectory_stream import TrajectoryPickleStreamer

    server = TrajectoryPickleStreamer(
        "data/processed/trajectories.pkl",
        user_ids=["000", "001"],
        max_trajectories_per_user=5,
    )
    for event in server.stream_events():
        if event.kind == "point":
            print(event.user_id, event.lat, event.lon)
"""

from __future__ import annotations

import argparse
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import pandas as pd

from src.utils.config import DATASET

DEFAULT_PICKLE = DATASET["data_file"]

STREAMING_MODE = "pickle_replay"


def normalize_user_id(user_id: Union[str, int]) -> str:
    """GeoLife-style ids: ``0`` → ``'000'``, ``'1'`` → ``'001'``."""
    s = str(user_id).strip()
    if s.isdigit():
        return s.zfill(3)
    return s


@dataclass(frozen=True)
class TrajectoryMeta:
    """Metadata for one trajectory in the pickle."""

    trajectory_index: int
    user_id: str
    file_id: str
    num_points: int


@dataclass(frozen=True)
class StreamEvent:
    """
    One item from :meth:`TrajectoryPickleStreamer.stream_events`.

    ``kind`` is ``trajectory_start``, ``point``, or ``trajectory_end``.
    """

    kind: str
    user_id: str
    file_id: str
    trajectory_index: int
    point_index: int = -1
    lat: Optional[float] = None
    lon: Optional[float] = None
    timestamp: Any = None
    alt: Optional[float] = None
    num_points: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_point(self) -> bool:
        return self.kind == "point"

    @property
    def is_trajectory_start(self) -> bool:
        return self.kind == "trajectory_start"

    @property
    def is_trajectory_end(self) -> bool:
        return self.kind == "trajectory_end"


class TrajectoryPickleStreamer:
    """
    Read ``trajectories.pkl`` and stream points for selected users.

    Parameters
    ----------
    pickle_path:
        Path to preprocessed pickle (default from project config).
    user_ids:
        GeoLife user ids, e.g. ``['000', '001']``. ``None`` → all users in file.
    max_trajectories_per_user:
        Cap trips per user (``None`` = no cap).
    max_trajectories_total:
        Global cap across all selected users.
    """

    mode = STREAMING_MODE

    def __init__(
        self,
        pickle_path: Union[str, Path] = DEFAULT_PICKLE,
        user_ids: Optional[Sequence[Union[str, int]]] = None,
        *,
        max_trajectories_per_user: Optional[int] = None,
        max_trajectories_total: Optional[int] = None,
    ):
        self.pickle_path = Path(pickle_path)
        if not self.pickle_path.is_file():
            raise FileNotFoundError(
                f"Missing {self.pickle_path}. Run: python src/utils/preprocess_geolife.py"
            )

        with self.pickle_path.open("rb") as f:
            self._trajectories: List[pd.DataFrame] = pickle.load(f)

        self._index_by_user: Dict[str, List[int]] = {}
        for i, traj in enumerate(self._trajectories):
            uid = self._trajectory_user_id(traj)
            self._index_by_user.setdefault(uid, []).append(i)

        all_users = sorted(self._index_by_user.keys())
        if user_ids is None:
            self.user_ids = all_users
        else:
            requested = [normalize_user_id(u) for u in user_ids]
            missing = [u for u in requested if u not in self._index_by_user]
            if missing:
                raise ValueError(
                    f"Unknown user_id(s): {missing}. "
                    f"Available sample: {all_users[:10]}{'...' if len(all_users) > 10 else ''}"
                )
            self.user_ids = requested

        self.max_trajectories_per_user = max_trajectories_per_user
        self.max_trajectories_total = max_trajectories_total

        self._trajectory_plan = self._build_trajectory_plan()

    @staticmethod
    def _trajectory_user_id(traj: pd.DataFrame) -> str:
        if "user_id" in traj.columns and len(traj):
            return normalize_user_id(traj["user_id"].iloc[0])
        return "unknown"

    @staticmethod
    def _trajectory_file_id(traj: pd.DataFrame) -> str:
        if "file_id" in traj.columns and len(traj):
            return str(traj["file_id"].iloc[0])
        return "unknown"

    def _build_trajectory_plan(self) -> List[int]:
        """Ordered list of trajectory indices to stream."""
        plan: List[int] = []
        total = 0
        for uid in self.user_ids:
            indices = self._index_by_user.get(uid, [])
            if self.max_trajectories_per_user is not None:
                indices = indices[: self.max_trajectories_per_user]
            for idx in indices:
                plan.append(idx)
                total += 1
                if self.max_trajectories_total is not None and total >= self.max_trajectories_total:
                    return plan
        return plan

    def list_trajectories(self) -> List[TrajectoryMeta]:
        """Metadata for trajectories that will be streamed."""
        out: List[TrajectoryMeta] = []
        for idx in self._trajectory_plan:
            traj = self._trajectories[idx]
            out.append(
                TrajectoryMeta(
                    trajectory_index=idx,
                    user_id=self._trajectory_user_id(traj),
                    file_id=self._trajectory_file_id(traj),
                    num_points=len(traj),
                )
            )
        return out

    def available_users(self) -> List[str]:
        return sorted(self._index_by_user.keys())

    def stream_events(self) -> Iterator[StreamEvent]:
        """
        Yield trajectory boundaries and points in temporal order.

        For each selected trip: ``trajectory_start`` → ``point`` × n → ``trajectory_end``.
        """
        for plan_pos, traj_idx in enumerate(self._trajectory_plan):
            traj = self._trajectories[traj_idx]
            uid = self._trajectory_user_id(traj)
            fid = self._trajectory_file_id(traj)
            n = len(traj)

            yield StreamEvent(
                kind="trajectory_start",
                user_id=uid,
                file_id=fid,
                trajectory_index=traj_idx,
                num_points=n,
                extra={"plan_position": plan_pos},
            )

            for point_index, row in traj.iterrows():
                yield StreamEvent(
                    kind="point",
                    user_id=uid,
                    file_id=fid,
                    trajectory_index=traj_idx,
                    point_index=int(point_index) if isinstance(point_index, (int, float)) else point_index,
                    lat=float(row["lat"]),
                    lon=float(row["lon"]),
                    timestamp=row["timestamp"] if "timestamp" in traj.columns else None,
                    alt=float(row["alt"]) if "alt" in traj.columns else None,
                    num_points=n,
                )

            yield StreamEvent(
                kind="trajectory_end",
                user_id=uid,
                file_id=fid,
                trajectory_index=traj_idx,
                num_points=n,
                extra={"plan_position": plan_pos},
            )

    def stream_points(self) -> Iterator[StreamEvent]:
        """Only GPS points (no start/end markers)."""
        for event in self.stream_events():
            if event.is_point:
                yield event

    def iter_trajectory_dataframes(self) -> Iterator[Tuple[TrajectoryMeta, pd.DataFrame]]:
        """Yield full trajectory DataFrames one at a time (bounded memory per trip)."""
        for meta in self.list_trajectories():
            yield meta, self._trajectories[meta.trajectory_index].copy()


def is_streaming_mode(obj: Any) -> bool:
    """True for :class:`TrajectoryPickleStreamer` (pickle replay, not batch random access)."""
    return getattr(obj, "mode", None) == STREAMING_MODE


def stream_squish_for_users(
    user_ids: Sequence[Union[str, int]],
    pickle_path: Union[str, Path] = DEFAULT_PICKLE,
    *,
    max_trajectories_per_user: Optional[int] = None,
    max_trajectories_total: Optional[int] = None,
    chunk_size: int = 250,
    chunk_compression_ratio: float = 2.0,
    overlap: int = 1,
) -> List[Tuple[TrajectoryMeta, pd.DataFrame]]:
    """
    Stream selected users from pickle → streaming SQUISH → one simplified DF per trip.

    Returns:
        List of ``(metadata, simplified DataFrame)`` in stream order.
    """
    from src.algorithms.squish_stream import SquishStreamProcessor

    server = TrajectoryPickleStreamer(
        pickle_path,
        user_ids=user_ids,
        max_trajectories_per_user=max_trajectories_per_user,
        max_trajectories_total=max_trajectories_total,
    )

    results: List[Tuple[TrajectoryMeta, pd.DataFrame]] = []
    proc: Optional[SquishStreamProcessor] = None
    current_meta: Optional[TrajectoryMeta] = None

    for event in server.stream_events():
        if event.is_trajectory_start:
            proc = SquishStreamProcessor(
                chunk_size=chunk_size,
                chunk_compression_ratio=chunk_compression_ratio,
                overlap=overlap,
            )
            current_meta = TrajectoryMeta(
                trajectory_index=event.trajectory_index,
                user_id=event.user_id,
                file_id=event.file_id,
                num_points=event.num_points,
            )
        elif event.is_point and proc is not None:
            proc.push(
                event.lat,
                event.lon,
                timestamp=event.timestamp,
                alt=event.alt,
                user_id=event.user_id,
                file_id=event.file_id,
            )
        elif event.is_trajectory_end and proc is not None and current_meta is not None:
            results.append((current_meta, proc.finalize()))
            proc = None
            current_meta = None

    return results


def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Stream trajectories from trajectories.pkl for selected users.",
    )
    parser.add_argument(
        "--pickle",
        default=DEFAULT_PICKLE,
        help="Path to trajectories.pkl",
    )
    parser.add_argument(
        "--users",
        nargs="+",
        required=True,
        help="User ids, e.g. 000 001",
    )
    parser.add_argument(
        "--max-per-user",
        type=int,
        default=None,
        help="Max trajectories per user",
    )
    parser.add_argument(
        "--max-total",
        type=int,
        default=3,
        help="Max trajectories overall (demo default: 3)",
    )
    parser.add_argument(
        "--squish",
        action="store_true",
        help="Run streaming SQUISH on each trajectory and print summary",
    )
    args = parser.parse_args()

    if args.squish:
        simplified = stream_squish_for_users(
            args.users,
            args.pickle,
            max_trajectories_per_user=args.max_per_user,
            max_trajectories_total=args.max_total,
        )
        print(f"mode: {STREAMING_MODE}  squish: streaming_replay")
        for meta, df in simplified:
            print(
                f"  user={meta.user_id} file={meta.file_id} "
                f"in={meta.num_points} out={len(df)} idx={meta.trajectory_index}"
            )
        return

    server = TrajectoryPickleStreamer(
        args.pickle,
        user_ids=args.users,
        max_trajectories_per_user=args.max_per_user,
        max_trajectories_total=args.max_total,
    )
    print(f"mode: {server.mode}  is_streaming: {is_streaming_mode(server)}")
    print(f"users: {server.user_ids}  trips planned: {len(server.list_trajectories())}")

    point_count = 0
    for event in server.stream_events():
        if event.is_trajectory_start:
            print(f"  START user={event.user_id} file={event.file_id} n={event.num_points}")
        elif event.is_point:
            point_count += 1
        elif event.is_trajectory_end:
            print(f"  END   user={event.user_id} file={event.file_id}")
    print(f"total points streamed: {point_count}")


if __name__ == "__main__":
    _cli()

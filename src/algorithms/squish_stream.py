"""
Streaming-style SQUISH API (simulated replay on stored trajectories).

This module does NOT read live GPS sockets. It replays existing trajectory data
one point at a time through a bounded buffer, runs local SQUISH simplification per
chunk, and merges results in ``finalize()``.

Use this to:
  - Demonstrate an online / incremental processing interface
  - Compare ``mode='streaming_replay'`` vs batch ``squish()`` in baseline_algorithms

Batch reference: ``squish(trajectory, num_points)`` needs the full series up front.
Streaming here uses fixed-size chunks so memory stays bounded while points arrive.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.algorithms.baseline_algorithms import squish

STREAMING_MODE = "streaming_replay"
BATCH_MODE = "batch"


def processing_mode(processor: Union["SquishStreamProcessor", str]) -> str:
    """Return ``streaming_replay`` or ``batch`` for a processor instance or mode string."""
    if isinstance(processor, SquishStreamProcessor):
        return processor.mode
    return processor if processor in (STREAMING_MODE, BATCH_MODE) else str(processor)


def is_streaming_mode(processor: Union["SquishStreamProcessor", str]) -> bool:
    return processing_mode(processor) == STREAMING_MODE


class SquishStreamProcessor:
    """
    Incremental SQUISH processor with a bounded buffer (simulated streaming).

  Each ``push()`` adds one GPS sample. When the buffer reaches ``chunk_size``,
  ``squish`` runs on that chunk; simplified points are appended to the output
  except ``overlap`` anchor point(s) kept for continuity with the next chunk.
  ``finalize()`` flushes the last buffer and returns the simplified trajectory.
    """

    mode = STREAMING_MODE

    def __init__(
        self,
        chunk_size: int = 250,
        chunk_compression_ratio: float = 2.0,
        overlap: int = 1,
    ):
        if chunk_size < 4:
            raise ValueError("chunk_size must be at least 4")
        if chunk_compression_ratio < 1.0:
            raise ValueError("chunk_compression_ratio must be >= 1.0")
        if overlap < 1 or overlap >= chunk_size:
            raise ValueError("overlap must be in [1, chunk_size)")

        self.chunk_size = chunk_size
        self.chunk_compression_ratio = chunk_compression_ratio
        self.overlap = overlap

        self._buffer: List[Dict[str, Any]] = []
        self._output_rows: List[Dict[str, Any]] = []
        self._points_received = 0
        self._chunks_processed = 0
        self._closed = False

    @property
    def points_received(self) -> int:
        return self._points_received

    @property
    def chunks_processed(self) -> int:
        return self._chunks_processed

    @property
    def buffer_size(self) -> int:
        return len(self._buffer)

    def push(
        self,
        lat: float,
        lon: float,
        timestamp: Any = None,
        **extra: Any,
    ) -> int:
        """
        Accept one incoming point. Returns number of output points committed this call.
        """
        if self._closed:
            raise RuntimeError("Processor is closed; create a new instance or call reset().")

        row = {"lat": float(lat), "lon": float(lon)}
        if timestamp is not None:
            row["timestamp"] = timestamp
        row.update(extra)

        self._buffer.append(row)
        self._points_received += 1

        if len(self._buffer) < self.chunk_size:
            return 0
        return self._flush_buffer(final=False)

    def reset(self) -> None:
        self._buffer.clear()
        self._output_rows.clear()
        self._points_received = 0
        self._chunks_processed = 0
        self._closed = False

    def finalize(self) -> pd.DataFrame:
        """Flush remaining buffer and return the simplified trajectory DataFrame."""
        if not self._closed:
            if self._buffer:
                self._flush_buffer(final=True)
            self._closed = True

        if not self._output_rows:
            return pd.DataFrame(columns=["lat", "lon"])

        out = pd.DataFrame(self._output_rows)
        return self._dedupe_consecutive(out)

    def _flush_buffer(self, final: bool) -> int:
        chunk_df = pd.DataFrame(self._buffer)
        n = len(chunk_df)
        budget = max(2, int(n / self.chunk_compression_ratio))

        if final:
            keep_overlap = 0
        else:
            budget = max(2, min(budget, n - self.overlap))
            keep_overlap = self.overlap

        indices = squish(chunk_df, num_points=budget, indices=True)
        simplified = chunk_df.iloc[indices].reset_index(drop=True)

        if keep_overlap > 0 and len(simplified) > keep_overlap:
            commit = simplified.iloc[:-keep_overlap]
            carry = simplified.iloc[-keep_overlap:]
        else:
            commit = simplified
            carry = pd.DataFrame(columns=chunk_df.columns)

        n_commit = len(commit)
        if n_commit > 0:
            self._output_rows.extend(commit.to_dict("records"))

        self._chunks_processed += 1
        self._buffer = carry.to_dict("records") if len(carry) else []
        return n_commit

    @staticmethod
    def _dedupe_consecutive(df: pd.DataFrame) -> pd.DataFrame:
        if len(df) <= 1:
            return df.reset_index(drop=True)
        pts = df[["lat", "lon"]].values
        keep = [0]
        for i in range(1, len(pts)):
            if not np.allclose(pts[i], pts[keep[-1]]):
                keep.append(i)
        return df.iloc[keep].reset_index(drop=True)


def iter_trajectory_points(
    trajectory: pd.DataFrame,
    *,
    lat_col: str = "lat",
    lon_col: str = "lon",
    timestamp_col: str = "timestamp",
) -> Iterator[Dict[str, Any]]:
    """Yield one dict per row from an in-memory trajectory (replay order)."""
    for _, row in trajectory.iterrows():
        item: Dict[str, Any] = {
            "lat": row[lat_col],
            "lon": row[lon_col],
        }
        if timestamp_col in trajectory.columns:
            item["timestamp"] = row[timestamp_col]
        yield item


def iter_plt_points(plt_path: Union[str, Path]) -> Iterator[Dict[str, Any]]:
    """
    Yield points from a GeoLife ``.plt`` file (skips 6-line header).

    Columns: lat, lon, _, altitude, date, time → combined timestamp string.
    """
    path = Path(plt_path)
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            if line_no < 6:
                continue
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            lat, lon = float(parts[0]), float(parts[1])
            alt = float(parts[3]) if parts[3] else 0.0
            ts = f"{parts[4].strip()} {parts[5].strip()}"
            yield {"lat": lat, "lon": lon, "altitude": alt, "timestamp": ts}


def stream_squish_points(
    point_source: Iterator[Dict[str, Any]],
    *,
    chunk_size: int = 250,
    chunk_compression_ratio: float = 2.0,
    overlap: int = 1,
) -> Tuple[pd.DataFrame, SquishStreamProcessor]:
    """
    Replay any point iterator through the streaming processor.

    Returns:
        (simplified DataFrame, processor with stats: points_received, chunks_processed)
    """
    proc = SquishStreamProcessor(
        chunk_size=chunk_size,
        chunk_compression_ratio=chunk_compression_ratio,
        overlap=overlap,
    )
    for pt in point_source:
        proc.push(pt["lat"], pt["lon"], timestamp=pt.get("timestamp"), **{
            k: v for k, v in pt.items() if k not in ("lat", "lon", "timestamp")
        })
    return proc.finalize(), proc


def stream_squish_dataframe(
    trajectory: pd.DataFrame,
    *,
    chunk_size: int = 250,
    chunk_compression_ratio: float = 2.0,
    overlap: int = 1,
) -> Tuple[pd.DataFrame, SquishStreamProcessor]:
    """Replay an existing trajectory DataFrame through streaming SQUISH."""
    return stream_squish_points(
        iter_trajectory_points(trajectory),
        chunk_size=chunk_size,
        chunk_compression_ratio=chunk_compression_ratio,
        overlap=overlap,
    )


def stream_squish_pickle(
    pickle_path: Union[str, Path],
    trajectory_index: int = 0,
    **kwargs: Any,
) -> Tuple[pd.DataFrame, SquishStreamProcessor]:
    """Load ``trajectories.pkl`` and stream-simplify one trajectory by index."""
    path = Path(pickle_path)
    with path.open("rb") as f:
        trajectories = pickle.load(f)
    if trajectory_index < 0 or trajectory_index >= len(trajectories):
        raise IndexError(f"trajectory_index {trajectory_index} out of range [0, {len(trajectories)})")
    return stream_squish_dataframe(trajectories[trajectory_index], **kwargs)


def stream_squish_plt(
    plt_path: Union[str, Path],
    **kwargs: Any,
) -> Tuple[pd.DataFrame, SquishStreamProcessor]:
    """Stream-simplify one GeoLife ``.plt`` file from disk."""
    return stream_squish_points(iter_plt_points(plt_path), **kwargs)


def compare_batch_vs_stream(
    trajectory: pd.DataFrame,
    compression_ratio: float = 5.0,
    **stream_kwargs: Any,
) -> Dict[str, Any]:
    """
    Run batch ``squish`` vs streaming replay on the same data for comparison.

    Returns counts, modes, and whether outputs are identical (usually they are not).
    """
    n = len(trajectory)
    budget = max(2, int(n / compression_ratio))

    batch_pts = squish(trajectory, num_points=budget, indices=False)
    stream_df, proc = stream_squish_dataframe(trajectory, **stream_kwargs)

    batch_df = pd.DataFrame(batch_pts, columns=["lat", "lon"])
    same_len = len(batch_df) == len(stream_df)
    same_points = False
    if same_len and len(batch_df) > 0:
        same_points = np.allclose(
            batch_df[["lat", "lon"]].values,
            stream_df[["lat", "lon"]].values,
            rtol=0,
            atol=1e-9,
        )

    return {
        "batch_mode": BATCH_MODE,
        "stream_mode": proc.mode,
        "original_points": n,
        "batch_budget": budget,
        "batch_output_points": len(batch_df),
        "stream_output_points": len(stream_df),
        "stream_chunks_processed": proc.chunks_processed,
        "outputs_identical": same_points,
        "is_streaming": is_streaming_mode(proc),
    }


if __name__ == "__main__":
    repo = Path(__file__).resolve().parents[2]
    pkl = repo / "data" / "processed" / "trajectories.pkl"

    if not pkl.is_file():
        raise SystemExit(f"Missing {pkl}. Run: python src/utils/preprocess_geolife.py")

    simplified, processor = stream_squish_pickle(pkl, trajectory_index=0)
    with pkl.open("rb") as f:
        trajectories = pickle.load(f)
    traj = trajectories[0]

    cmp = compare_batch_vs_stream(traj, compression_ratio=5.0)
    print("Streaming SQUISH demo")
    print(f"  mode:              {processor.mode}")
    print(f"  is_streaming_mode: {is_streaming_mode(processor)}")
    print(f"  points received:   {processor.points_received}")
    print(f"  chunks processed:  {processor.chunks_processed}")
    print(f"  output points:     {len(simplified)}")
    print(f"  batch vs stream:   {cmp}")

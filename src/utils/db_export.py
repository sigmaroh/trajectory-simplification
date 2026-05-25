"""
PostgreSQL export for trajectory data and experiment results.

Tables created / used:
  trajectories        — every GPS point from processed trajectories.pkl
  trajectory_index    — one row per trajectory (metadata)
  experiment_results  — all rows from experiment_results.csv / single_*.csv

Setup::

    # 1. Install deps (once, outside sandbox)
    pip install psycopg2-binary sqlalchemy

    # 2. Create the database (psql prompt)
    CREATE DATABASE trajectory_db;

    # 3. Set env vars (or pass url=... directly)
    export PGHOST=localhost
    export PGPORT=5432
    export PGUSER=postgres
    export PGPASSWORD=your_password
    export PGDATABASE=trajectory_db

    # 4. Run export
    python -m src.utils.db_export --what trajectories
    python -m src.utils.db_export --what results --results-file results/experiment_results.csv
"""

from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd

from src.utils.config import DATASET

# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------

def build_url(
    host: Optional[str] = None,
    port: Optional[int] = None,
    user: Optional[str] = None,
    password: Optional[str] = None,
    database: Optional[str] = None,
) -> str:
    """Build a SQLAlchemy PostgreSQL URL from args or environment variables."""
    h = host     or os.getenv("PGHOST",     "localhost")
    p = port     or int(os.getenv("PGPORT", "5432"))
    u = user     or os.getenv("PGUSER",     "postgres")
    pw = password or os.getenv("PGPASSWORD", "")
    db = database or os.getenv("PGDATABASE", "trajectory_db")
    return f"postgresql+psycopg2://{u}:{pw}@{h}:{p}/{db}"


def get_engine(url: str):
    """Return a SQLAlchemy engine; raises ImportError if deps missing."""
    try:
        from sqlalchemy import create_engine
    except ImportError:
        raise ImportError(
            "sqlalchemy not installed. Run:  pip install psycopg2-binary sqlalchemy"
        )
    return create_engine(url, echo=False)


# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------

CREATE_TRAJECTORIES_SQL = """
CREATE TABLE IF NOT EXISTS trajectories (
    id              BIGSERIAL PRIMARY KEY,
    trajectory_id   INTEGER      NOT NULL,
    user_id         VARCHAR(16),
    file_id         VARCHAR(64),
    point_index     INTEGER      NOT NULL,
    lat             DOUBLE PRECISION NOT NULL,
    lon             DOUBLE PRECISION NOT NULL,
    alt             DOUBLE PRECISION,
    timestamp       TIMESTAMP WITH TIME ZONE
);
CREATE INDEX IF NOT EXISTS idx_traj_user   ON trajectories (user_id);
CREATE INDEX IF NOT EXISTS idx_traj_tid    ON trajectories (trajectory_id);
"""

CREATE_TRAJ_INDEX_SQL = """
CREATE TABLE IF NOT EXISTS trajectory_index (
    trajectory_id   INTEGER PRIMARY KEY,
    user_id         VARCHAR(16),
    file_id         VARCHAR(64),
    num_points      INTEGER,
    start_time      TIMESTAMP WITH TIME ZONE,
    end_time        TIMESTAMP WITH TIME ZONE,
    mean_lat        DOUBLE PRECISION,
    mean_lon        DOUBLE PRECISION
);
"""

CREATE_RESULTS_SQL = """
CREATE TABLE IF NOT EXISTS experiment_results (
    id                      BIGSERIAL PRIMARY KEY,
    algorithm               VARCHAR(64),
    compression_ratio       DOUBLE PRECISION,
    trajectory_id           INTEGER,
    user_id                 VARCHAR(16),
    file_id                 VARCHAR(64),
    input_points            INTEGER,
    output_points           INTEGER,
    budget                  INTEGER,
    actual_compression_ratio DOUBLE PRECISION,
    runtime_seconds         DOUBLE PRECISION,
    memory_mb               DOUBLE PRECISION,
    throughput_traj_per_sec DOUBLE PRECISION,
    hausdorff_distance      DOUBLE PRECISION,
    average_pte             DOUBLE PRECISION,
    frechet_distance        DOUBLE PRECISION,
    turn_preservation       DOUBLE PRECISION,
    stop_preservation       DOUBLE PRECISION
);
CREATE INDEX IF NOT EXISTS idx_res_algo ON experiment_results (algorithm);
CREATE INDEX IF NOT EXISTS idx_res_cr   ON experiment_results (compression_ratio);
"""


def create_tables(engine) -> None:
    """Create all tables if they do not exist."""
    from sqlalchemy import text
    with engine.begin() as conn:
        conn.execute(text(CREATE_TRAJECTORIES_SQL))
        conn.execute(text(CREATE_TRAJ_INDEX_SQL))
        conn.execute(text(CREATE_RESULTS_SQL))
    print("Tables created (or already exist).")


# ---------------------------------------------------------------------------
# Trajectory export
# ---------------------------------------------------------------------------

def export_trajectories(
    engine,
    pickle_path: Union[str, Path] = DATASET["data_file"],
    max_trajectories: Optional[int] = None,
    batch_size: int = 5000,
    if_exists: str = "append",
) -> int:
    """
    Load trajectories.pkl and insert every GPS point into ``trajectories``
    and one summary row per trip into ``trajectory_index``.

    Returns total points written.
    """
    path = Path(pickle_path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. Run preprocess_geolife.py first.")

    print(f"Loading {path} ...")
    with path.open("rb") as f:
        trajectories: List[pd.DataFrame] = pickle.load(f)

    if max_trajectories:
        trajectories = trajectories[:max_trajectories]

    print(f"  Exporting {len(trajectories)} trajectories ...")

    point_rows: List[dict] = []
    index_rows: List[dict] = []
    total_points = 0

    def flush_points():
        nonlocal point_rows
        if not point_rows:
            return
        pd.DataFrame(point_rows).to_sql(
            "trajectories", engine, if_exists=if_exists, index=False, method="multi"
        )
        point_rows = []

    for traj_id, traj in enumerate(trajectories):
        uid = str(traj["user_id"].iloc[0]) if "user_id" in traj.columns else None
        fid = str(traj["file_id"].iloc[0]) if "file_id" in traj.columns else None

        for i, row in traj.iterrows():
            ts = None
            if "timestamp" in traj.columns:
                try:
                    ts = pd.to_datetime(row["timestamp"])
                    if ts.tzinfo is None:
                        ts = ts.tz_localize("UTC")
                except Exception:
                    ts = None

            point_rows.append({
                "trajectory_id": traj_id,
                "user_id": uid,
                "file_id": fid,
                "point_index": int(i) if isinstance(i, (int, float)) else 0,
                "lat": float(row["lat"]),
                "lon": float(row["lon"]),
                "alt": float(row["alt"]) if "alt" in traj.columns else None,
                "timestamp": ts,
            })
            total_points += 1

            if len(point_rows) >= batch_size:
                flush_points()

        # trajectory index row
        times = pd.to_datetime(traj["timestamp"]) if "timestamp" in traj.columns else None
        index_rows.append({
            "trajectory_id": traj_id,
            "user_id": uid,
            "file_id": fid,
            "num_points": len(traj),
            "start_time": (
                times.iloc[0].tz_localize("UTC") if times is not None and times.iloc[0].tzinfo is None
                else (times.iloc[0] if times is not None else None)
            ),
            "end_time": (
                times.iloc[-1].tz_localize("UTC") if times is not None and times.iloc[-1].tzinfo is None
                else (times.iloc[-1] if times is not None else None)
            ),
            "mean_lat": float(traj["lat"].mean()),
            "mean_lon": float(traj["lon"].mean()),
        })

        if (traj_id + 1) % 100 == 0:
            flush_points()
            print(f"  {traj_id + 1}/{len(trajectories)} trajectories ...")

    flush_points()

    # write trajectory index
    pd.DataFrame(index_rows).to_sql(
        "trajectory_index", engine, if_exists=if_exists, index=False, method="multi"
    )

    print(f"Done: {total_points} GPS points, {len(index_rows)} trajectory rows written.")
    return total_points


# ---------------------------------------------------------------------------
# Experiment results export
# ---------------------------------------------------------------------------

def export_results(
    engine,
    results_path: Union[str, Path],
    if_exists: str = "append",
) -> int:
    """
    Load a results CSV (experiment_results.csv or single_*.csv) and
    insert numeric metric columns into ``experiment_results``.

    Returns number of rows written.
    """
    path = Path(results_path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}")

    df = pd.read_csv(path)
    print(f"Loaded {len(df)} rows from {path}")

    # keep only columns that match the table schema
    schema_cols = [
        "algorithm", "compression_ratio", "trajectory_id",
        "user_id", "file_id", "input_points", "output_points",
        "budget", "actual_compression_ratio",
        "runtime_seconds", "memory_mb", "throughput_traj_per_sec",
        "hausdorff_distance", "average_pte", "frechet_distance",
        "turn_preservation", "stop_preservation",
    ]
    # only keep cols present AND with scalar (non-tuple) values
    keep = []
    for c in schema_cols:
        if c in df.columns:
            # drop if column contains tuples/objects masquerading as floats
            try:
                if pd.api.types.is_numeric_dtype(df[c]) or df[c].dtype == object:
                    keep.append(c)
            except Exception:
                pass

    export_df = df[keep].copy()

    # coerce numeric cols
    for c in keep:
        if c not in ("algorithm", "user_id", "file_id"):
            export_df[c] = pd.to_numeric(export_df[c], errors="coerce")

    export_df.to_sql(
        "experiment_results", engine, if_exists=if_exists, index=False, method="multi"
    )
    print(f"Written {len(export_df)} rows to experiment_results table.")
    return len(export_df)


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

def query(engine, sql: str) -> pd.DataFrame:
    """Run any SQL and return a DataFrame."""
    return pd.read_sql(sql, engine)


def summary_by_algorithm(engine) -> pd.DataFrame:
    return query(engine, """
        SELECT algorithm,
               compression_ratio,
               COUNT(*)                         AS runs,
               ROUND(AVG(input_points)::numeric, 0)       AS mean_input_pts,
               ROUND(AVG(runtime_seconds)::numeric, 4)    AS mean_runtime_s,
               ROUND(AVG(hausdorff_distance)::numeric, 2) AS mean_hausdorff_m,
               ROUND(AVG(turn_preservation)::numeric, 4)  AS mean_turn_pres,
               ROUND(AVG(stop_preservation)::numeric, 4)  AS mean_stop_pres
        FROM experiment_results
        GROUP BY algorithm, compression_ratio
        ORDER BY algorithm, compression_ratio
    """)


def trajectories_for_user(engine, user_id: str) -> pd.DataFrame:
    return query(engine, f"""
        SELECT trajectory_id, num_points, start_time, end_time, mean_lat, mean_lon
        FROM trajectory_index
        WHERE user_id = '{user_id}'
        ORDER BY trajectory_id
    """)


def gps_points_for_trajectory(engine, trajectory_id: int) -> pd.DataFrame:
    return query(engine, f"""
        SELECT point_index, lat, lon, alt, timestamp
        FROM trajectories
        WHERE trajectory_id = {trajectory_id}
        ORDER BY point_index
    """)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Export processed data to PostgreSQL")
    parser.add_argument(
        "--what",
        choices=["trajectories", "results", "both"],
        required=True,
        help="What to export",
    )
    parser.add_argument("--host",     default=None)
    parser.add_argument("--port",     type=int, default=None)
    parser.add_argument("--user",     default=None)
    parser.add_argument("--password", default=None)
    parser.add_argument("--database", default=None)
    parser.add_argument("--pickle",   default=DATASET["data_file"])
    parser.add_argument(
        "--results-file",
        default="results/experiment_results.csv",
        help="CSV to import (for --what results/both)",
    )
    parser.add_argument(
        "--max-trajectories",
        type=int,
        default=None,
        help="Limit number of trajectories exported (default: all)",
    )
    parser.add_argument(
        "--if-exists",
        choices=["append", "replace", "fail"],
        default="append",
        help="Pandas to_sql if_exists behaviour (default: append)",
    )
    args = parser.parse_args()

    url = build_url(args.host, args.port, args.user, args.password, args.database)
    print(f"Connecting to: {url.replace(args.password or '', '***') if args.password else url}")

    engine = get_engine(url)
    create_tables(engine)

    if args.what in ("trajectories", "both"):
        export_trajectories(
            engine,
            pickle_path=args.pickle,
            max_trajectories=args.max_trajectories,
            if_exists=args.if_exists,
        )

    if args.what in ("results", "both"):
        export_results(engine, args.results_file, if_exists=args.if_exists)

    print("\nQuery example:")
    try:
        df = summary_by_algorithm(engine)
        print(df.to_string())
    except Exception as e:
        print(f"  (could not query yet: {e})")


if __name__ == "__main__":
    main()

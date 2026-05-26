-- ============================================================
-- Migration 001 — Initial schema
-- GeoLife trajectory simplification project
-- ============================================================

-- ── 1. users ─────────────────────────────────────────────────
-- One row per GeoLife participant folder (000 – 181).
CREATE TABLE IF NOT EXISTS users (
    user_id    CHAR(3)     PRIMARY KEY,   -- zero-padded, e.g. '000', '010'
    created_at TIMESTAMPTZ DEFAULT now()
);

-- ── 2. trajectory_index ──────────────────────────────────────
-- One row per cleaned trajectory (after preprocessing).
-- PK is composite (user_id, trajectory_id) so the same
-- integer trajectory_id can safely exist for different users.
CREATE TABLE IF NOT EXISTS trajectory_index (
    user_id               CHAR(3)          NOT NULL REFERENCES users(user_id),
    trajectory_id         INTEGER          NOT NULL,
    file_id               VARCHAR(20),             -- raw .plt filename stem
    num_points            INTEGER,
    start_time            TIMESTAMPTZ,
    end_time              TIMESTAMPTZ,
    mean_lat              DOUBLE PRECISION,
    mean_lon              DOUBLE PRECISION,
    -- statistics from trajectory_properties.csv
    duration              DOUBLE PRECISION,        -- seconds
    total_distance        DOUBLE PRECISION,        -- metres
    mean_interval         DOUBLE PRECISION,        -- seconds
    std_interval          DOUBLE PRECISION,
    cv_interval           DOUBLE PRECISION,        -- sampling irregularity
    mean_speed            DOUBLE PRECISION,        -- m/s
    max_speed             DOUBLE PRECISION,        -- m/s
    mean_direction_change DOUBLE PRECISION,        -- degrees
    num_stops             INTEGER,
    stop_ratio            DOUBLE PRECISION,
    num_turns             INTEGER,
    turn_ratio            DOUBLE PRECISION,
    updated_at            TIMESTAMPTZ DEFAULT now(),

    PRIMARY KEY (user_id, trajectory_id)
);

CREATE INDEX IF NOT EXISTS idx_ti_user ON trajectory_index(user_id);

-- ── 3. trajectory_points ─────────────────────────────────────
-- One row per GPS fix.
-- FK cascades delete so removing a trajectory_index row also
-- removes all its points.
CREATE TABLE IF NOT EXISTS trajectory_points (
    user_id       CHAR(3)          NOT NULL,
    trajectory_id INTEGER          NOT NULL,
    point_index   INTEGER          NOT NULL,  -- 0-based position in trajectory
    lat           DOUBLE PRECISION NOT NULL,
    lon           DOUBLE PRECISION NOT NULL,
    alt           DOUBLE PRECISION,           -- metres, NULL if unavailable
    recorded_at   TIMESTAMPTZ,

    PRIMARY KEY (user_id, trajectory_id, point_index),
    FOREIGN KEY (user_id, trajectory_id)
        REFERENCES trajectory_index(user_id, trajectory_id)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_tp_traj ON trajectory_points(user_id, trajectory_id);
CREATE INDEX IF NOT EXISTS idx_tp_time ON trajectory_points(recorded_at);

-- ── 4. experiment_results ────────────────────────────────────
-- One row per (algorithm × trajectory × compression_ratio) run.
-- PK prevents duplicate rows when results are re-imported.
CREATE TABLE IF NOT EXISTS experiment_results (
    algorithm                VARCHAR(64)      NOT NULL,
    trajectory_id            INTEGER          NOT NULL,
    user_id                  CHAR(3)          NOT NULL,
    compression_ratio        DOUBLE PRECISION NOT NULL,
    file_id                  VARCHAR(20),
    input_points             INTEGER,
    output_points            INTEGER,
    budget                   INTEGER,
    actual_compression_ratio DOUBLE PRECISION,
    runtime_seconds          DOUBLE PRECISION,
    memory_mb                DOUBLE PRECISION,
    throughput_traj_per_sec  DOUBLE PRECISION,
    hausdorff_distance       DOUBLE PRECISION,
    average_pte              DOUBLE PRECISION,
    frechet_distance         DOUBLE PRECISION,
    turn_preservation        DOUBLE PRECISION,
    stop_preservation        DOUBLE PRECISION,

    PRIMARY KEY (algorithm, trajectory_id, user_id, compression_ratio)
);

CREATE INDEX IF NOT EXISTS idx_er_algo ON experiment_results(algorithm);
CREATE INDEX IF NOT EXISTS idx_er_cr   ON experiment_results(compression_ratio);
CREATE INDEX IF NOT EXISTS idx_er_user ON experiment_results(user_id);

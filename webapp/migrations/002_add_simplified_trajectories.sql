-- ============================================================
-- Migration 002 — Simplified trajectories table
-- Stores the output of each simplification algorithm so the
-- webapp can serve pre-computed simplified paths without
-- re-running Python on every request.
-- ============================================================

CREATE TABLE IF NOT EXISTS simplified_trajectories (
    algorithm         VARCHAR(64)      NOT NULL,
    user_id           CHAR(3)          NOT NULL,
    trajectory_id     INTEGER          NOT NULL,
    compression_ratio DOUBLE PRECISION NOT NULL,
    budget            INTEGER,
    -- serialised as a JSON array of [lat, lon] pairs
    points            JSONB            NOT NULL,
    created_at        TIMESTAMPTZ      DEFAULT now(),

    PRIMARY KEY (algorithm, user_id, trajectory_id, compression_ratio),
    FOREIGN KEY (user_id, trajectory_id)
        REFERENCES trajectory_index(user_id, trajectory_id)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_st_user_traj
    ON simplified_trajectories(user_id, trajectory_id);
CREATE INDEX IF NOT EXISTS idx_st_algo
    ON simplified_trajectories(algorithm);

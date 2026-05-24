"""
Central project configuration.

All project-wide constants are defined here as plain Python values.
Import what you need::

    from src.utils.config import EARTH_RADIUS_M, STOP_SPEED_THRESHOLD_MS
    from src.utils.config import EXPERIMENTS, DATASET, OUTPUT   # structured dicts
"""

# ---------------------------------------------------------------------------
# Physics
# ---------------------------------------------------------------------------
EARTH_RADIUS_M: float = 6_371_000           # metres — WGS-84 mean radius

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
STOP_SPEED_THRESHOLD_MS: float = 1.0        # m/s  — speed below which a point is a stop
TURN_THRESHOLD_DEG: float = 30.0            # deg  — minimum direction change for a turn
MAX_VALID_INTERVAL_S: float = 3_600.0       # s    — discard GPS gaps longer than 1 hour
MAX_VALID_SPEED_MS: float = 80.0            # m/s  — discard unrealistic speeds (~288 km/h)
MIN_STOP_DURATION_S: float = 30.0           # s    — minimum duration for a significant stop
MAX_PREPROCESS_SPEED_MS: float = 200.0      # m/s  — ceiling used when preprocessing raw trajectories

# ---------------------------------------------------------------------------
# Binary-search parameters (used by budget-constrained algorithm wrappers)
# ---------------------------------------------------------------------------
BINARY_SEARCH_ITERATIONS: int = 20          # number of bisection steps
BINARY_SEARCH_EPS_MIN: float = 0.0          # lower bound for epsilon (metres)
BINARY_SEARCH_EPS_MAX: float = 1_000.0      # upper bound for epsilon (metres)
BINARY_SEARCH_TOLERANCE: int = 1            # acceptable absolute difference from target budget

# ---------------------------------------------------------------------------
# Plot defaults
# ---------------------------------------------------------------------------
PLOT_DPI: int = 150
PLOT_FIGSIZE: tuple = (12, 8)
PLOT_FONT_SIZE: int = 10

# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------
EXPERIMENTS = {
    "max_trajectories": 50,
    "compression_ratios": [2.0, 5.0, 10.0, 20.0],
    "algorithms": ["original", "dp", "squish", "vw", "sw", "rw", "greedy_policy", "proposed"],
    "algorithm_params": {
        "greedy_policy": {
            "alpha": 0.5,
        },
        "proposed": {
            "weights": {
                "turn": 0.3,
                "stop": 0.3,
                "speed": 0.2,
                "irregular": 0.2,
            },
        },
    },
}

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
DATASET = {
    "data_file": "data/processed/trajectories.pkl",
    "min_points": 100,
    "max_points": 5_000,
}

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------
OUTPUT = {
    "results_dir": "results",
    "figures_dir": "results/figures",
    "tables_dir": "results/tables",
}

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED: int = 42

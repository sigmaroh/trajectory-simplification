"""
RL-based Trajectory Simplification — DQN Policy (NumPy implementation).

Implements the Markov Decision Process formulation of trajectory simplification
described in:

    Wang et al. (2021). Trajectory Simplification with Reinforcement Learning.
    ICDE 2021, pp. 684-695. IEEE.

The paper trains a neural-network agent that walks a trajectory left-to-right
and decides at each interior point whether to KEEP or DROP it, subject to a
compression budget.  We reproduce this formulation faithfully using a pure-NumPy
DQN (Deep Q-Network) so the implementation is self-contained without requiring
PyTorch or TensorFlow.

Architecture
------------
State  (6 features per point):
  1. geo_dev          — perpendicular distance from p_i to chord
                        last_kept → p_{i+1}, normalised by traj diagonal
  2. bearing_change   — |direction change| at p_i, normalised to [0,1]
  3. speed_change     — |speed change| at p_i, normalised to [0,1]
  4. time_gap_ratio   — Δt_i / median_Δt, clipped to [0,1]
  5. budget_used      — kept_so_far / total_budget
  6. progress         — i / (n-1)

Action space: {0: drop, 1: keep}

Q-network: Linear(6→hidden) → ReLU → Linear(hidden→2)

Training: DQN with
  - experience replay (circular buffer, capacity 50 000)
  - ε-greedy exploration, ε decayed from 1.0 → 0.05
  - target network updated every `target_update` steps
  - mean-squared Bellman loss, SGD with momentum

Usage
-----
    from src.algorithms.rl_policy import RLPolicySimplification

    model = RLPolicySimplification()
    model.train(trajectories, epochs=50)          # offline training
    model.save("models/rl_policy.npz")

    # Later — no retraining needed
    model2 = RLPolicySimplification()
    model2.load("models/rl_policy.npz")
    simplified = model2.simplify(trajectory, budget=20)
"""

from __future__ import annotations

import pickle
from collections import deque
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
STATE_DIM  = 6
ACTION_DIM = 2          # 0=drop, 1=keep
EARTH_R    = 6_371_000  # metres


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _haversine(p1: np.ndarray, p2: np.ndarray) -> float:
    """Great-circle distance in metres between two (lat, lon) points (degrees)."""
    la1, lo1 = np.radians(p1)
    la2, lo2 = np.radians(p2)
    dlat, dlon = la2 - la1, lo2 - lo1
    a = np.sin(dlat / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin(dlon / 2) ** 2
    return EARTH_R * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _perp_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Perpendicular distance from point p to line segment a→b (metres)."""
    ab = b - a
    ab_len = np.linalg.norm(ab)
    if ab_len < 1e-10:
        return _haversine(p, a)
    t = np.clip(np.dot(p - a, ab) / (ab_len ** 2), 0.0, 1.0)
    proj = a + t * ab
    return np.linalg.norm(p - proj) * EARTH_R * np.pi / 180


def _bearing(p1: np.ndarray, p2: np.ndarray) -> float:
    """Forward azimuth from p1 to p2 in degrees [0, 360)."""
    la1, lo1 = np.radians(p1)
    la2, lo2 = np.radians(p2)
    dlon = lo2 - lo1
    b = np.arctan2(
        np.sin(dlon) * np.cos(la2),
        np.cos(la1) * np.sin(la2) - np.sin(la1) * np.cos(la2) * np.cos(dlon),
    )
    return (np.degrees(b) + 360) % 360


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def _precompute(traj: pd.DataFrame) -> dict:
    """Pre-compute per-point arrays needed for state feature extraction."""
    pts = traj[['lat', 'lon']].values
    n   = len(pts)

    # Timestamps → time intervals
    if 'timestamp' in traj.columns:
        ts  = pd.to_datetime(traj['timestamp']).values
        dts = np.array(
            [(pd.Timestamp(ts[i]) - pd.Timestamp(ts[i - 1])).total_seconds()
             for i in range(1, n)],
            dtype=float,
        )
        dts = np.where(dts > 0, dts, 1.0)
        median_dt = float(np.median(dts)) or 1.0
    else:
        dts       = np.ones(n - 1)
        median_dt = 1.0

    # Step distances (metres)
    dists = np.array([_haversine(pts[i - 1], pts[i]) for i in range(1, n)])

    # Speeds (m/s)
    speeds = dists / dts

    # Bearings
    bearings = np.array([_bearing(pts[i - 1], pts[i]) for i in range(1, n)])

    # Bearing changes (0 at first interior point)
    bear_ch = np.zeros(n)
    for i in range(1, n - 1):
        d = abs(bearings[i] - bearings[i - 1])
        bear_ch[i] = min(d, 360 - d)

    # Speed changes
    speed_ch = np.zeros(n)
    for i in range(1, n - 1):
        speed_ch[i] = abs(speeds[i] - speeds[i - 1])

    # Trajectory diagonal for geo_dev normalisation
    diag = max(_haversine(pts[0], pts[-1]), 1.0)

    return dict(
        pts=pts, n=n,
        dts=dts, median_dt=median_dt,
        speeds=speeds, bearings=bearings,
        bear_ch=bear_ch, speed_ch=speed_ch,
        diag=diag,
    )


def _state(pc: dict, i: int, last_kept: int, budget_used: float,
           max_bear_ch: float, max_speed_ch: float) -> np.ndarray:
    """Return the 6-dim state vector for point i."""
    pts = pc['pts']

    # 1. Geometric deviation from chord last_kept → i+1
    next_i = min(i + 1, pc['n'] - 1)
    geo = _perp_distance(pts[i], pts[last_kept], pts[next_i]) / pc['diag']

    # 2. Bearing change (normalised)
    bc  = pc['bear_ch'][i] / (max_bear_ch + 1e-8)

    # 3. Speed change (normalised)
    sc  = pc['speed_ch'][i] / (max_speed_ch + 1e-8)

    # 4. Time-gap ratio (clipped)
    tgr = 0.0
    if i > 0:
        tgr = min(pc['dts'][i - 1] / pc['median_dt'], 3.0) / 3.0

    # 5. Budget used so far
    bu  = float(np.clip(budget_used, 0.0, 1.0))

    # 6. Progress through trajectory
    prog = i / max(pc['n'] - 1, 1)

    return np.array([geo, bc, sc, tgr, bu, prog], dtype=np.float32)


# ---------------------------------------------------------------------------
# Q-Network (2-layer MLP, NumPy)
# ---------------------------------------------------------------------------

class _QNetwork:
    """
    Lightweight 2-layer MLP: Linear(in→h) → ReLU → Linear(h→2).

    Parameters stored as plain numpy arrays; trained with mini-batch
    gradient descent (momentum SGD).
    """

    def __init__(self, in_dim: int = STATE_DIM, hidden: int = 64,
                 lr: float = 1e-3, momentum: float = 0.9,
                 seed: int = 42) -> None:
        rng = np.random.default_rng(seed)
        scale1 = np.sqrt(2.0 / in_dim)
        scale2 = np.sqrt(2.0 / hidden)
        self.W1 = rng.normal(0, scale1, (hidden, in_dim)).astype(np.float32)
        self.b1 = np.zeros(hidden, dtype=np.float32)
        self.W2 = rng.normal(0, scale2, (ACTION_DIM, hidden)).astype(np.float32)
        self.b2 = np.zeros(ACTION_DIM, dtype=np.float32)
        self.lr, self.mom = lr, momentum
        # Momentum buffers
        self.vW1 = np.zeros_like(self.W1)
        self.vb1 = np.zeros_like(self.b1)
        self.vW2 = np.zeros_like(self.W2)
        self.vb2 = np.zeros_like(self.b2)

    # ---- forward ----
    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """x: (batch, in_dim) → q: (batch, 2), cache for backward."""
        h  = x @ self.W1.T + self.b1          # (batch, hidden)
        h_relu = np.maximum(h, 0)
        q  = h_relu @ self.W2.T + self.b2     # (batch, 2)
        return q, (x, h, h_relu)

    def predict(self, x: np.ndarray) -> np.ndarray:
        q, _ = self.forward(x)
        return q

    # ---- backward + SGD update ----
    def update(self, x: np.ndarray, actions: np.ndarray,
               targets: np.ndarray) -> float:
        """One mini-batch gradient step; returns mean squared loss."""
        q, (x_, h, h_relu) = self.forward(x)

        # MSE loss on selected actions only
        batch = len(x)
        dq = np.zeros_like(q)
        dq[np.arange(batch), actions] = 2.0 * (q[np.arange(batch), actions] - targets) / batch

        # Layer 2 gradients
        dW2 = dq.T @ h_relu
        db2 = dq.sum(axis=0)
        dh_relu = dq @ self.W2

        # ReLU backward
        dh = dh_relu * (h > 0)

        # Layer 1 gradients
        dW1 = dh.T @ x_
        db1 = dh.sum(axis=0)

        # Momentum SGD
        self.vW1 = self.mom * self.vW1 - self.lr * dW1
        self.vb1 = self.mom * self.vb1 - self.lr * db1
        self.vW2 = self.mom * self.vW2 - self.lr * dW2
        self.vb2 = self.mom * self.vb2 - self.lr * db2

        self.W1 += self.vW1
        self.b1 += self.vb1
        self.W2 += self.vW2
        self.b2 += self.vb2

        loss = float(np.mean((q[np.arange(batch), actions] - targets) ** 2))
        return loss

    def copy_weights_from(self, other: '_QNetwork') -> None:
        self.W1[:] = other.W1
        self.b1[:] = other.b1
        self.W2[:] = other.W2
        self.b2[:] = other.b2

    def get_weights(self) -> dict:
        return dict(W1=self.W1.copy(), b1=self.b1.copy(),
                    W2=self.W2.copy(), b2=self.b2.copy())

    def set_weights(self, d: dict) -> None:
        self.W1[:] = d['W1']
        self.b1[:] = d['b1']
        self.W2[:] = d['W2']
        self.b2[:] = d['b2']


# ---------------------------------------------------------------------------
# Experience replay buffer
# ---------------------------------------------------------------------------

class _ReplayBuffer:
    def __init__(self, capacity: int = 50_000, seed: int = 0) -> None:
        self.buf: deque = deque(maxlen=capacity)
        self.rng = np.random.default_rng(seed)

    def push(self, s, a, r, s_next, done) -> None:
        self.buf.append((s, a, r, s_next, done))

    def sample(self, batch: int):
        idx  = self.rng.integers(0, len(self.buf), size=batch)
        batch_data = [self.buf[i] for i in idx]
        s      = np.stack([x[0] for x in batch_data]).astype(np.float32)
        a      = np.array([x[1] for x in batch_data], dtype=np.int32)
        r      = np.array([x[2] for x in batch_data], dtype=np.float32)
        s_next = np.stack([x[3] for x in batch_data]).astype(np.float32)
        done   = np.array([x[4] for x in batch_data], dtype=np.float32)
        return s, a, r, s_next, done

    def __len__(self) -> int:
        return len(self.buf)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class RLPolicySimplification:
    """
    DQN-based trajectory simplification agent (Wang et al., ICDE 2021).

    Parameters
    ----------
    state_dim : int
        Dimension of the state feature vector (default 6).
    hidden : int
        Hidden layer width of the Q-network (default 64).
    gamma : float
        Discount factor for future rewards.
    lr : float
        Learning rate for momentum SGD.
    batch_size : int
        Mini-batch size for replay updates.
    target_update : int
        Number of gradient steps between target-network syncs.
    buffer_capacity : int
        Maximum number of transitions in the replay buffer.
    eps_start / eps_end / eps_decay : float
        Epsilon-greedy schedule parameters.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        state_dim:       int   = STATE_DIM,
        hidden:          int   = 64,
        gamma:           float = 0.99,
        lr:              float = 1e-3,
        batch_size:      int   = 64,
        target_update:   int   = 200,
        buffer_capacity: int   = 50_000,
        eps_start:       float = 1.0,
        eps_end:         float = 0.05,
        eps_decay:       float = 0.995,
        seed:            int   = 42,
    ) -> None:
        self.gamma        = gamma
        self.batch_size   = batch_size
        self.target_update = target_update
        self.eps          = eps_start
        self.eps_end      = eps_end
        self.eps_decay    = eps_decay
        self.rng          = np.random.default_rng(seed)

        self.online = _QNetwork(state_dim, hidden, lr=lr, seed=seed)
        self.target = _QNetwork(state_dim, hidden, lr=lr, seed=seed)
        self.target.copy_weights_from(self.online)

        self.buffer  = _ReplayBuffer(buffer_capacity, seed=seed)
        self._steps  = 0
        self._trained = False

    # ------------------------------------------------------------------
    # MDP helpers
    # ------------------------------------------------------------------

    def _reward(self, action: int, geo_dev: float,
                budget_exceeded: bool) -> float:
        """
        Reward function (Wang et al. formulation):
          keep  → small budget-usage penalty
          drop  → penalty proportional to geometric deviation
          budget exceeded → large negative reward
        """
        if budget_exceeded:
            return -10.0
        if action == 1:  # keep
            return -0.05
        else:             # drop
            return -float(geo_dev)

    def _run_episode(self, traj: pd.DataFrame, budget: int,
                     train: bool = True) -> List[int]:
        """
        Run one episode on `traj`.  If train=True, push transitions to
        buffer and perform gradient updates.  Returns selected indices.
        """
        pc     = _precompute(traj)
        n      = pc['n']
        budget = max(2, min(budget, n))

        max_bc = pc['bear_ch'].max() or 1.0
        max_sc = pc['speed_ch'].max() or 1.0

        kept       = [0]           # always keep first point
        budget_rem = budget - 2    # reserve 1 for last point

        for i in range(1, n - 1):
            budget_used = (len(kept) - 1) / max(budget - 2, 1)
            s = _state(pc, i, kept[-1], budget_used, max_bc, max_sc)

            # Force keep if we must fill remaining budget
            points_left = (n - 1) - i   # interior points left + last
            must_keep   = (budget_rem >= points_left)
            must_drop   = (budget_rem <= 0)

            if must_keep:
                action = 1
            elif must_drop:
                action = 0
            elif train and self.rng.random() < self.eps:
                action = int(self.rng.integers(0, 2))
            else:
                q = self.online.predict(s[np.newaxis])[0]
                action = int(np.argmax(q))

            geo_dev = float(s[0])
            done    = (i == n - 2)

            if action == 1:
                budget_rem -= 1
                kept.append(i)

            # Next state
            budget_used_next = (len(kept) - 1) / max(budget - 2, 1)
            s_next = _state(pc, min(i + 1, n - 2), kept[-1],
                            budget_used_next, max_bc, max_sc)

            r = self._reward(action, geo_dev, budget_rem < 0)

            if train:
                self.buffer.push(s, action, r, s_next, float(done))
                self._steps += 1

                if len(self.buffer) >= self.batch_size:
                    self._update()

                if self._steps % self.target_update == 0:
                    self.target.copy_weights_from(self.online)

        kept.append(n - 1)   # always keep last point
        return sorted(set(kept))

    def _update(self) -> None:
        """One DQN gradient step from a replay-buffer mini-batch."""
        s, a, r, s_next, done = self.buffer.sample(self.batch_size)
        q_next  = self.target.predict(s_next)                 # target network
        targets = r + self.gamma * q_next.max(axis=1) * (1 - done)
        self.online.update(s, a, targets)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(
        self,
        trajectories: List[pd.DataFrame],
        epochs:        int   = 50,
        compression:   float = 5.0,
        verbose:       bool  = True,
    ) -> None:
        """
        Offline training on a list of trajectory DataFrames.

        Parameters
        ----------
        trajectories : list of pd.DataFrame
            Training trajectories (each with 'lat', 'lon', 'timestamp').
        epochs : int
            Number of passes over the training set.
        compression : float
            Target compression ratio used to compute budget per trajectory.
        verbose : bool
            Print epoch-level loss/epsilon.
        """
        rng = np.random.default_rng(0)
        for epoch in range(1, epochs + 1):
            order = rng.permutation(len(trajectories))
            ep_lens = []

            for idx in order:
                traj   = trajectories[idx]
                budget = max(2, int(len(traj) / compression))
                sel    = self._run_episode(traj, budget, train=True)
                ep_lens.append(len(sel))

                # Decay epsilon after each trajectory
                self.eps = max(self.eps_end, self.eps * self.eps_decay)

            if verbose:
                print(f"Epoch {epoch:3d}/{epochs}  "
                      f"ε={self.eps:.4f}  "
                      f"mean_kept={np.mean(ep_lens):.1f}  "
                      f"buffer={len(self.buffer)}")

        self._trained = True

    def simplify(
        self,
        trajectory: pd.DataFrame,
        budget:     int,
        indices:    bool = False,
    ):
        """
        Simplify one trajectory using the trained policy (greedy, ε=0).

        Parameters
        ----------
        trajectory : pd.DataFrame
        budget : int
            Target number of points to keep.
        indices : bool
            If True return list of selected indices; otherwise return
            (N, 2) numpy array of [lat, lon].

        Returns
        -------
        np.ndarray of shape (k, 2) or list of int
        """
        pts  = trajectory[['lat', 'lon']].values
        sel  = self._run_episode(trajectory, budget, train=False)
        if indices:
            return sel
        return pts[sel]

    def save(self, path: str) -> None:
        """Save network weights to a .npz file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        w = self.online.get_weights()
        np.savez(str(p), **w, _trained=np.array([self._trained]))
        print(f"Saved RL policy weights → {p}")

    def load(self, path: str) -> None:
        """Load network weights from a .npz file."""
        data = np.load(str(path))
        self.online.set_weights({k: data[k] for k in ('W1', 'b1', 'W2', 'b2')})
        self.target.copy_weights_from(self.online)
        self._trained = bool(data.get('_trained', np.array([False]))[0])
        print(f"Loaded RL policy weights ← {path}")


# ---------------------------------------------------------------------------
# Convenience wrapper for simplify_with_budget integration
# ---------------------------------------------------------------------------

_GLOBAL_MODEL: Optional[RLPolicySimplification] = None
_DEFAULT_WEIGHTS = Path(__file__).parent.parent.parent / "models" / "rl_policy.npz"


def get_or_load_model(weights_path: Optional[str] = None) -> RLPolicySimplification:
    """Return a cached trained model, loading weights if available."""
    global _GLOBAL_MODEL
    if _GLOBAL_MODEL is not None:
        return _GLOBAL_MODEL
    _GLOBAL_MODEL = RLPolicySimplification()
    p = Path(weights_path) if weights_path else _DEFAULT_WEIGHTS
    if p.exists():
        _GLOBAL_MODEL.load(str(p))
    else:
        print(f"[RL] No weights found at {p}. Run train_rl_policy.py first.")
    return _GLOBAL_MODEL


# ---------------------------------------------------------------------------
# CLI — train and save weights
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse, pickle

    parser = argparse.ArgumentParser(description='Train RL trajectory simplification policy')
    parser.add_argument('--data-file',   default='data/processed/trajectories.pkl')
    parser.add_argument('--weights-out', default='models/rl_policy.npz')
    parser.add_argument('--epochs',      type=int,   default=50)
    parser.add_argument('--compression', type=float, default=5.0)
    parser.add_argument('--max-traj',    type=int,   default=200,
                        help='Max trajectories to train on')
    args = parser.parse_args()

    print(f'Loading trajectories from {args.data_file}...')
    with open(args.data_file, 'rb') as f:
        trajs = pickle.load(f)
    trajs = trajs[:args.max_traj]
    print(f'Training on {len(trajs)} trajectories for {args.epochs} epochs...')

    model = RLPolicySimplification()
    model.train(trajs, epochs=args.epochs, compression=args.compression)
    model.save(args.weights_out)

    # Quick sanity check
    sample = trajs[0]
    budget = max(2, int(len(sample) / args.compression))
    result = model.simplify(sample, budget)
    print(f'\nSanity check — original: {len(sample)}, simplified: {len(result)}, '
          f'ratio: {len(sample)/len(result):.2f}×')

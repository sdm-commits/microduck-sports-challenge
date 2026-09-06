"""Procedural, deterministic celebration reference (all 14 actuated joints) played after a goal.

Offsets from HOME_POSE as a function of time since the goal event, for `duration` seconds, then home.
Knees bounce, head yaw sweeps side to side, head pitch nods, neck lifts. Generated from CelebrationConfig only.
"""
from __future__ import annotations

import numpy as np

from .config import CelebrationConfig, HOME_POSE, JOINT_NAMES

J = {name: i for i, name in enumerate(JOINT_NAMES)}


def celebration_offsets(cfg: CelebrationConfig, t_since_goal: float) -> np.ndarray:
    """14-dim joint offset from home, t_since_goal in seconds (<0 or > duration -> zeros)."""
    q = np.zeros(14)
    t = t_since_goal
    if t < 0.0 or t >= cfg.duration:
        return q
    env = np.sin(np.pi * min(t / cfg.ramp, 1.0) / 2.0) ** 2 if t < cfg.ramp else 1.0          # ease in
    if t > cfg.duration - cfg.ramp:
        env *= np.sin(np.pi * (cfg.duration - t) / cfg.ramp / 2.0) ** 2                       # ease out
    w = 2.0 * np.pi * cfg.bounce_hz
    bounce = 0.5 * (1.0 - np.cos(w * t))            # 0..1 knee flex pulse
    for side in ("left", "right"):
        q[J[f"{side}_knee"]] = cfg.bounce_knee * bounce * env
        q[J[f"{side}_hip_pitch"]] = (-1 if side == "left" else 1) * cfg.bounce_hip_pitch * bounce * env
        q[J[f"{side}_ankle"]] = -cfg.bounce_ankle * bounce * env
    q[J["head_yaw"]] = cfg.head_yaw_amp * np.sin(2.0 * np.pi * cfg.head_yaw_hz * t) * env
    q[J["head_pitch"]] = cfg.head_pitch_amp * np.sin(2.0 * np.pi * cfg.head_pitch_hz * t) * env
    q[J["neck_pitch"]] = cfg.neck_lift * env
    q[J["head_roll"]] = cfg.head_roll_amp * np.sin(2.0 * np.pi * cfg.head_yaw_hz * t + np.pi / 2) * env
    return q

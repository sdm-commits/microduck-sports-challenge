"""Procedural, deterministic joint-space kick reference for the right leg.

The reference is a piecewise smooth (cosine-interpolated) trajectory of the 10 leg joints, expressed
as offsets from HOME_POSE, sampled at the control rate. It is generated from ReferenceConfig only,
so it is fully reproducible and committed by construction (no external mocap files).
"""
from __future__ import annotations

import numpy as np

from .config import ReferenceConfig, HOME_POSE, LEG_JOINT_IDX, LEG_JOINT_NAMES

# Index of each leg joint inside the 10-dim leg vector.
L = {name: i for i, name in enumerate(LEG_JOINT_NAMES)}


def _smooth(a: float, b: float, s: float) -> float:
    """Cosine ease between a and b for s in [0, 1]."""
    s = float(np.clip(s, 0.0, 1.0))
    return a + (b - a) * (0.5 - 0.5 * np.cos(np.pi * s))


def keyframes(cfg: ReferenceConfig):
    """(time, 10-dim offset) keyframes; the trajectory eases between consecutive keyframes."""
    home = np.zeros(10)
    shift = np.zeros(10)
    shift[L["left_hip_roll"]] = cfg.weight_shift_roll
    shift[L["right_hip_roll"]] = cfg.weight_shift_roll + cfg.swing_abduction_roll
    shift[L["left_hip_pitch"]] = cfg.left_hip_pitch_compensation

    back = shift.copy()
    back[L["right_hip_pitch"]] = cfg.backswing_hip_pitch
    back[L["right_knee"]] = cfg.backswing_knee

    strike = shift.copy()
    strike[L["right_hip_pitch"]] = cfg.strike_hip_pitch
    strike[L["right_knee"]] = cfg.strike_knee
    strike[L["right_ankle"]] = cfg.strike_ankle

    return [
        (0.0, home),
        (cfg.t_shift_end, shift),
        (cfg.t_back_end, back),
        (cfg.t_strike_end, strike),
        (cfg.t_recover_end, home),
    ]


def reference_offsets(cfg: ReferenceConfig, t: float) -> np.ndarray:
    """10-dim leg joint offset from home at time t (seconds). Home for t >= t_recover_end."""
    kfs = keyframes(cfg)
    if t <= 0.0:
        return kfs[0][1].copy()
    for (t0, q0), (t1, q1) in zip(kfs[:-1], kfs[1:]):
        if t <= t1:
            s = (t - t0) / max(t1 - t0, 1e-9)
            return np.array([_smooth(a, b, s) for a, b in zip(q0, q1)])
    return kfs[-1][1].copy()


def build_reference(cfg: ReferenceConfig, ctrl_dt: float, horizon_steps: int) -> np.ndarray:
    """Array [horizon_steps, 10] of absolute leg joint targets (home + offset) at each control step."""
    home_legs = np.asarray(HOME_POSE)[LEG_JOINT_IDX]
    out = np.zeros((horizon_steps, 10))
    for k in range(horizon_steps):
        out[k] = home_legs + reference_offsets(cfg, k * ctrl_dt)
    return out


if __name__ == "__main__":  # print a compact table for inspection
    cfg = ReferenceConfig()
    ref = build_reference(cfg, 0.02, 90)
    np.set_printoptions(precision=3, suppress=True, linewidth=160)
    print("step  t     " + "  ".join(f"{n[:9]:>9s}" for n in LEG_JOINT_NAMES))
    for k in range(0, 90, 5):
        print(f"{k:4d} {k*0.02:5.2f} ", ref[k])

"""Warm-start helper: load an older (49-obs / 10-action) shootout checkpoint into the 63-obs / 14-action model.

Old observation layout: [gyro3, grav3, legq10, legqd10, prev_a10, ball3, ballv3, phase2, onehot3, tgt2]
New observation layout: [gyro3, grav3, legq10, headq4, legqd10, headqd4, prev_a14, ball3, ballv3, phase2, onehot3, tgt2, celeb2]
New inputs get zero weights (and mean 0 / var 1 normalization); new head-joint outputs start at zero mean with the
reset log-std. Same-shape checkpoints load unchanged.
"""
from __future__ import annotations

import numpy as np
import torch

from .config import LEG_JOINT_IDX

OLD_OBS, NEW_OBS = 49, 63


def old_to_new_obs_index():
    m = {}
    def block(old0, new0, n):
        for i in range(n):
            m[old0 + i] = new0 + i
    block(0, 0, 6)            # gyro, grav
    block(6, 6, 10)           # leg q
    block(16, 20, 10)         # leg qd  (new: head q at 16..19)
    for k, leg in enumerate(LEG_JOINT_IDX):      # prev action 10 -> 14 (joint order)
        m[26 + k] = 34 + leg
    block(36, 48, 13)         # ball3, ballv3, phase2, onehot3, tgt2
    return m


def load_padded(model, obs_rms, ck):
    sd = ck["model_state"]
    tgt = model.state_dict()
    if sd["actor.0.weight"].shape == tgt["actor.0.weight"].shape and sd["log_std"].shape == tgt["log_std"].shape:
        model.load_state_dict(sd); obs_rms.load(ck["obs_rms"]); return "exact"
    assert sd["actor.0.weight"].shape[1] == OLD_OBS and tgt["actor.0.weight"].shape[1] == NEW_OBS
    idx = old_to_new_obs_index()
    new = {k: v.clone() for k, v in tgt.items()}
    for net in ("actor", "critic"):
        W = torch.zeros_like(tgt[f"{net}.0.weight"])
        for o, n in idx.items():
            W[:, n] = sd[f"{net}.0.weight"][:, o]
        new[f"{net}.0.weight"] = W; new[f"{net}.0.bias"] = sd[f"{net}.0.bias"].clone()
        for k in sd:
            if k.startswith(f"{net}.") and not k.startswith(f"{net}.0.") and sd[k].shape == tgt[k].shape:
                new[k] = sd[k].clone()
    # actor output: 10 leg rows -> 14 joint rows
    last = max(int(k.split(".")[1]) for k in sd if k.startswith("actor."))
    Wl = torch.zeros_like(tgt[f"actor.{last}.weight"]); bl = torch.zeros_like(tgt[f"actor.{last}.bias"])
    ls = torch.full_like(tgt["log_std"], float(tgt["log_std"].mean()))
    for k, leg in enumerate(LEG_JOINT_IDX):
        Wl[leg] = sd[f"actor.{last}.weight"][k]; bl[leg] = sd[f"actor.{last}.bias"][k]; ls[leg] = sd["log_std"][k]
    new[f"actor.{last}.weight"] = Wl; new[f"actor.{last}.bias"] = bl; new["log_std"] = ls
    model.load_state_dict(new)
    mean = np.zeros(NEW_OBS); var = np.ones(NEW_OBS)
    om, ov = np.asarray(ck["obs_rms"]["mean"]), np.asarray(ck["obs_rms"]["var"])
    for o, n in idx.items():
        mean[n] = om[o]; var[n] = ov[o]
    obs_rms.load({"mean": mean, "var": var, "count": ck["obs_rms"]["count"]})
    return "padded"

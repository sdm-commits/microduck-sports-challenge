"""Collect post-walk kick-start states with the current approach policy, for fine-tuning the kick policy.

    python tools/collect_state_bank.py --approach-checkpoint runs/approach/final.pt --episodes 300 --out evidence/state_bank_postwalk.npz
Each stored state is the full (qpos, qvel) at the moment the kick policy would take over (after the settle), plus the
commanded target zone. Only clean, in-reach triggers are stored (pocket error < far_trigger_skip, ball undisturbed).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from duck_kick.approach_env import ApproachEnv
from duck_kick.config import EnvConfig
from duck_kick.policy import PPOPolicy


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--approach-checkpoint", type=Path, required=True)
    p.add_argument("--episodes", type=int, default=300)
    p.add_argument("--first-seed", type=int, default=5000)
    p.add_argument("--out", type=Path, default=Path("evidence/state_bank_postwalk.npz"))
    a = p.parse_args()
    approach = PPOPolicy(a.approach_checkpoint)
    cfg = EnvConfig()
    env = ApproachEnv(cfg, seed=0)
    qpos, qvel, target, errs = [], [], [], []
    for i in range(a.episodes):
        seed = a.first_seed + i
        obs = env.reset(seed=seed, target=i % 3); approach.reset(); done = False
        while not done:
            act = approach.act(obs)
            if act[2] > cfg.approach.trigger_threshold:
                err = float(np.linalg.norm(env.pocket_error()))
                env._check_disturbed()
                if err < cfg.approach.far_trigger_skip and not env.disturbed:
                    for _ in range(int(round(cfg.approach.settle_seconds / cfg.control.ctrl_dt))):
                        env._walk_step(np.zeros(3))
                    env._check_disturbed()
                    if not env.disturbed and not env.inner._fallen():
                        qpos.append(env.inner.data.qpos.copy()); qvel.append(env.inner.data.qvel.copy()); target.append(i % 3); errs.append(err)
                break
            obs, _, done, _ = env.step(np.array([act[0], act[1], -1.0]))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, qpos=np.array(qpos), qvel=np.array(qvel), target=np.array(target))
    print(f"stored {len(qpos)} states from {a.episodes} episodes -> {a.out}; pocket error mean {np.mean(errs):.3f} m")


if __name__ == "__main__":
    main()

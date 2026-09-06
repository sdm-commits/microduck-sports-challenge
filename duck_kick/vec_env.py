"""Minimal multiprocess vectorized wrapper around KickEnv (no gymnasium dependency).

Each worker process owns `envs_per_worker` KickEnv instances and steps them serially; workers run in
parallel. Episodes auto-reset; the observation returned after a terminal step is the first observation
of the next episode and the terminal episode summary is passed back in `infos`.
"""
from __future__ import annotations

import multiprocessing as mp
import os
from typing import List, Optional

# Every worker runs single-threaded physics and tiny matrix products; BLAS/OpenMP thread pools in dozens of processes
# oversubscribe the machine and slow everything down. Set before any worker imports numpy/torch.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

from .config import EnvConfig
from .env import KickEnv, OBS_DIM, ACT_DIM


def _make(cfg, seed, randomize, env_kind, kick_checkpoint):
    if env_kind == "celebration":
        from .celebration_env import CelebrationEnv
        return CelebrationEnv(cfg, seed=seed, randomize=randomize, kick_checkpoint=kick_checkpoint)
    if env_kind == "approach":
        from .approach_env import ApproachEnv
        return ApproachEnv(cfg, seed=seed, randomize=randomize, kick_checkpoint=kick_checkpoint)
    return KickEnv(cfg, seed=seed, randomize=randomize)


def _worker(remote, cfg: EnvConfig, seeds: List[int], randomize: Optional[bool], env_kind: str = "kick",
            kick_checkpoint=None):
    import torch
    torch.set_num_threads(1)                      # frozen-policy inference inside workers must not oversubscribe cores
    envs = [_make(cfg, s, randomize, env_kind, kick_checkpoint) for s in seeds]
    obs = np.stack([e.reset() for e in envs])
    try:
        while True:
            cmd, arg = remote.recv()
            if cmd == "step":
                rewards = np.zeros(len(envs), dtype=np.float32)
                dones = np.zeros(len(envs), dtype=bool)
                infos = [None] * len(envs)
                for i, e in enumerate(envs):
                    o, r, d, info = e.step(arg[i])
                    rewards[i], dones[i] = r, d
                    if d:
                        infos[i] = {"episode": info["episode"], "fell": info["fell"], "timeout": info["timeout"]}
                        o = e.reset()
                    obs[i] = o
                remote.send((obs.copy(), rewards, dones, infos))
            elif cmd == "reset":
                obs = np.stack([e.reset() for e in envs])
                remote.send(obs.copy())
            elif cmd == "close":
                remote.close()
                break
    except (EOFError, KeyboardInterrupt):
        pass


class VecEnv:
    def __init__(self, cfg: EnvConfig, num_envs: int, num_workers: int, seed: int = 0,
                 randomize: Optional[bool] = None, env_kind: str = "kick", kick_checkpoint=None):
        assert num_envs % num_workers == 0, "num_envs must be divisible by num_workers"
        self.num_envs, self.num_workers = num_envs, num_workers
        per = num_envs // num_workers
        ctx = mp.get_context("spawn")
        self.remotes, self.procs = [], []
        for w in range(num_workers):
            parent, child = ctx.Pipe()
            seeds = [seed * 100_000 + w * per + i for i in range(per)]
            p = ctx.Process(target=_worker, args=(child, cfg, seeds, randomize, env_kind, kick_checkpoint), daemon=True)
            p.start(); child.close()
            self.remotes.append(parent); self.procs.append(p)
        self.per = per
        if env_kind == "approach":
            from .approach_env import APPROACH_OBS_DIM, APPROACH_ACT_DIM
            self.obs_dim, self.act_dim = APPROACH_OBS_DIM, APPROACH_ACT_DIM
        else:
            self.obs_dim, self.act_dim = OBS_DIM, ACT_DIM

    def reset(self) -> np.ndarray:
        for r in self.remotes:
            r.send(("reset", None))
        return np.concatenate([r.recv() for r in self.remotes])

    def step(self, actions: np.ndarray):
        for w, r in enumerate(self.remotes):
            r.send(("step", actions[w * self.per:(w + 1) * self.per]))
        obs, rew, done, infos = [], [], [], []
        for r in self.remotes:
            o, rw, d, inf = r.recv()
            obs.append(o); rew.append(rw); done.append(d); infos.extend(inf)
        return np.concatenate(obs), np.concatenate(rew), np.concatenate(done), infos

    def close(self):
        for r in self.remotes:
            try:
                r.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
        for p in self.procs:
            p.join(timeout=5)

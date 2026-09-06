"""Post-goal celebration environment.

Each episode: the FROZEN kick policy (49-obs / 10-action shootout checkpoint) plays the shot in KickEnv. If the ball
crosses the goal line (any zone), control switches to the learner: it commands all 14 joints for
`celebration.duration + settle` seconds and is rewarded by KickEnv's post-goal terms (celebration imitation,
liveliness, tilt, height, action rate, torque, fall). Shots that do not score are re-rolled (the learner never sees
them). The kick policy's weights are never updated here.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from .config import EnvConfig
from .env import KickEnv
from .policy import PPOPolicy, act_any


class CelebrationEnv:
    def __init__(self, cfg: Optional[EnvConfig] = None, seed: int = 0, randomize: Optional[bool] = None,
                 kick_checkpoint: Optional[Path] = None, settle: Optional[float] = None):
        """settle: seconds the learner keeps control after the celebration window; None = until the episode ends."""
        self.cfg = cfg or EnvConfig()
        self.inner = KickEnv(self.cfg, seed=seed, randomize=randomize)
        self.kick = PPOPolicy(kick_checkpoint) if kick_checkpoint else PPOPolicy()
        self.settle = settle
        self.celebration_steps = int(round(self.cfg.celebration.duration / self.cfg.control.ctrl_dt))
        self.reset()

    def reset(self, seed: Optional[int] = None, target: Optional[int] = None) -> np.ndarray:
        for _ in range(20):                       # re-roll shots that miss
            obs = self.inner.reset(seed=seed, target=target)
            seed = None
            self.kick.reset()
            done = False
            while not done and self.inner.goal_step is None:
                a = act_any(self.kick, obs)
                obs, _, done, _ = self.inner.step(a)
            if self.inner.goal_step is not None and not done:
                self.learner_t = 0
                remaining = self.inner.max_steps - self.inner.t
                self.max_learner_steps = (remaining if self.settle is None
                                          else min(remaining, self.celebration_steps + int(round(self.settle / self.cfg.control.ctrl_dt))))
                return obs
        raise RuntimeError("kick policy failed to score in 20 attempts")

    def step(self, action: np.ndarray):
        obs, r, done, info = self.inner.step(action)
        self.learner_t += 1
        if self.learner_t >= self.max_learner_steps:
            done = True
        if done and "episode" not in info:
            info["episode"] = self.inner.episode_summary()
        info["timeout"] = self.learner_t >= self.max_learner_steps
        return obs, r, done, info

    @property
    def data(self):
        return self.inner.data

    def render(self, *a, **k):
        return self.inner.render(*a, **k)

    def close(self):
        self.inner.close()

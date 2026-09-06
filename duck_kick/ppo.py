"""PPO (clipped objective, GAE) with a Gaussian MLP actor-critic and running observation normalization."""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn


class RunningMeanStd:
    def __init__(self, shape):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4

    def update(self, x: np.ndarray):
        bm, bv, bc = x.mean(0), x.var(0), x.shape[0]
        delta = bm - self.mean
        tot = self.count + bc
        self.mean = self.mean + delta * bc / tot
        m2 = self.var * self.count + bv * bc + delta ** 2 * self.count * bc / tot
        self.var = m2 / tot
        self.count = tot

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return np.clip((x - self.mean) / np.sqrt(self.var + 1e-8), -10.0, 10.0)

    def state(self) -> Dict:
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load(self, s: Dict):
        self.mean, self.var, self.count = np.asarray(s["mean"]), np.asarray(s["var"]), float(s["count"])


def mlp(inp: int, hidden: List[int], out: int) -> nn.Sequential:
    layers, d = [], inp
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ELU()]
        d = h
    layers.append(nn.Linear(d, out))
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: List[int], init_log_std: float, min_log_std: float = -20.0):
        super().__init__()
        self.min_log_std = float(min_log_std)
        self.actor = mlp(obs_dim, hidden, act_dim)
        self.critic = mlp(obs_dim, hidden, 1)
        self.log_std = nn.Parameter(torch.full((act_dim,), float(init_log_std)))
        nn.init.zeros_(self.actor[-1].bias)
        self.actor[-1].weight.data.mul_(0.01)

    def dist(self, obs: torch.Tensor) -> torch.distributions.Normal:
        return torch.distributions.Normal(self.actor(obs), self.log_std.clamp(min=self.min_log_std).exp())

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)

    @torch.no_grad()
    def act(self, obs: torch.Tensor, deterministic: bool = False):
        d = self.dist(obs)
        a = d.mean if deterministic else d.sample()
        return a, d.log_prob(a).sum(-1), self.value(obs)


def compute_gae(rewards, values, dones, last_value, gamma, lam):
    """rewards/values/dones: [T, N]; last_value: [N]. Returns advantages, returns."""
    T = rewards.shape[0]
    adv = np.zeros_like(rewards)
    gae = np.zeros(rewards.shape[1], dtype=np.float32)
    for t in reversed(range(T)):
        next_v = last_value if t == T - 1 else values[t + 1]
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_v * nonterminal - values[t]
        gae = delta + gamma * lam * nonterminal * gae
        adv[t] = gae
    return adv, adv + values


def ppo_update(model: ActorCritic, opt: torch.optim.Optimizer, batch: Dict[str, torch.Tensor], cfg) -> Dict[str, float]:
    n = batch["obs"].shape[0]
    stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0, "clip_frac": 0.0}
    count = 0
    for _ in range(cfg.epochs):
        perm = torch.randperm(n)
        for i in range(0, n, cfg.minibatch_size):
            idx = perm[i:i + cfg.minibatch_size]
            obs, act = batch["obs"][idx], batch["act"][idx]
            old_lp, adv, ret = batch["logp"][idx], batch["adv"][idx], batch["ret"][idx]
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            d = model.dist(obs)
            lp = d.log_prob(act).sum(-1)
            ratio = (lp - old_lp).exp()
            pl = -torch.min(ratio * adv, ratio.clamp(1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv).mean()
            vl = 0.5 * (model.value(obs) - ret).pow(2).mean()
            ent = d.entropy().sum(-1).mean()
            loss = pl + cfg.value_coef * vl - cfg.entropy_coef * ent
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            opt.step()
            with torch.no_grad():
                stats["policy_loss"] += pl.item(); stats["value_loss"] += vl.item(); stats["entropy"] += ent.item()
                stats["approx_kl"] += ((ratio - 1) - ratio.log()).mean().item()
                stats["clip_frac"] += ((ratio - 1).abs() > cfg.clip_eps).float().mean().item()
            count += 1
    return {k: v / max(count, 1) for k, v in stats.items()}

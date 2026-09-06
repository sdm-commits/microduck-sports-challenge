"""Inference-time policies: the trained PPO checkpoint and a zero-action (neutral) baseline."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import torch

from .config import EnvConfig, PPOConfig
from .env import OBS_DIM, ACT_DIM
from .ppo import ActorCritic, RunningMeanStd

DEFAULT_CHECKPOINT = Path(__file__).resolve().parent / "checkpoints" / "final.pt"                 # kick (49 obs / 10 act)
DEFAULT_CELEBRATION = Path(__file__).resolve().parent / "checkpoints" / "celebration.pt"         # celebration (63 obs / 14 act)
_KICK_IDX = None


def kick_obs_from_full(obs63: np.ndarray) -> np.ndarray:
    """Project the 63-dim observation onto the 49-dim layout the kick policy was trained on."""
    global _KICK_IDX
    if _KICK_IDX is None:
        from .warmstart import old_to_new_obs_index
        m = old_to_new_obs_index(); _KICK_IDX = np.array([m[i] for i in range(49)])
    return np.asarray(obs63, dtype=np.float32)[_KICK_IDX]


def pad_kick_action(a10: np.ndarray) -> np.ndarray:
    """10 leg actions -> 14 joint actions with head joints at zero (home)."""
    from .config import LEG_JOINT_IDX
    a = np.zeros(14, dtype=np.float32); a[np.asarray(LEG_JOINT_IDX)] = a10
    return a


def save_checkpoint(path: Path, model: ActorCritic, obs_rms: RunningMeanStd, env_cfg: EnvConfig,
                    ppo_cfg: PPOConfig, meta: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    obs_dim = int(model.actor[0].in_features); act_dim = int(model.log_std.shape[0])
    torch.save({
        "model_state": {k: v.cpu() for k, v in model.state_dict().items()},
        "obs_rms": obs_rms.state(),
        "env_cfg": env_cfg.to_dict(), "ppo_cfg": ppo_cfg.to_dict(),
        "obs_dim": obs_dim, "act_dim": act_dim, "meta": meta,
    }, path)


class PPOPolicy:
    """Closed-loop observation -> action controller loaded from a PPO checkpoint (deterministic mean)."""
    label = "trained PPO policy"

    def __init__(self, checkpoint: Path = DEFAULT_CHECKPOINT):
        ck = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
        torch.set_num_threads(1)
        hidden = ck["ppo_cfg"]["hidden"]
        self.obs_dim, self.act_dim = int(ck["obs_dim"]), int(ck["act_dim"])
        self.model = ActorCritic(ck["obs_dim"], ck["act_dim"], hidden, ck["ppo_cfg"]["init_log_std"],
                                 ck["ppo_cfg"].get("min_log_std", -20.0))
        self.model.load_state_dict(ck["model_state"]); self.model.eval()
        self.obs_rms = RunningMeanStd((ck["obs_dim"],)); self.obs_rms.load(ck["obs_rms"])
        self.meta = ck.get("meta", {})
        self.checkpoint = Path(checkpoint)
        self.call_count = 0
        # numpy copy of the actor for fast inference inside simulation workers (identical outputs, no torch overhead)
        self._layers = []
        for mod in self.model.actor:
            if isinstance(mod, torch.nn.Linear):
                self._layers.append((mod.weight.detach().cpu().numpy().astype(np.float64), mod.bias.detach().cpu().numpy().astype(np.float64)))
        self._mean = self.obs_rms.mean.copy(); self._std = np.sqrt(self.obs_rms.var + 1e-8)

    def reset(self):
        self.call_count = 0

    def act(self, observation: np.ndarray) -> np.ndarray:
        obs = np.asarray(observation, dtype=np.float32)
        if obs.shape != (self.obs_dim,):
            raise ValueError(f"expected observation shape {(self.obs_dim,)}, got {obs.shape}")
        x = np.clip((obs.astype(np.float64) - self._mean) / self._std, -10.0, 10.0)
        n = len(self._layers)
        for i, (W, b) in enumerate(self._layers):
            x = W @ x + b
            if i < n - 1:
                x = np.where(x > 0, x, np.expm1(x))          # ELU
        self.call_count += 1
        return np.clip(x, -1.0, 1.0).astype(np.float32)

    def act_torch(self, observation: np.ndarray) -> np.ndarray:
        """Reference implementation through torch (used by the tests to check the numpy path)."""
        obs = np.asarray(observation, dtype=np.float32)
        x = torch.as_tensor(self.obs_rms.normalize(obs), dtype=torch.float32)
        with torch.no_grad():
            a = self.model.actor(x).numpy()
        return np.clip(a, -1.0, 1.0).astype(np.float32)


def act_any(policy: "PPOPolicy", obs63: np.ndarray) -> np.ndarray:
    """Run a kick policy of either layout (49 obs/10 act or 63 obs/14 act) on the current 63-dim observation."""
    o = kick_obs_from_full(obs63) if policy.obs_dim == 49 else np.asarray(obs63, dtype=np.float32)
    a = policy.act(o)
    return pad_kick_action(a) if policy.act_dim == 10 else a


class CombinedPolicy:
    """Event-switched controller: the kick policy (49 obs / 10 leg actions, head held at home) drives the shot; once the
    observation's scored flag is set (ball crossed the goal line) the celebration policy (63 obs / 14 actions) takes over.
    Both are closed-loop PPO networks; the switch is the single boolean already in the observation."""
    label = "trained PPO kick policy + trained PPO celebration policy (event switch on goal)"

    def __init__(self, kick_checkpoint: Path = DEFAULT_CHECKPOINT, celebration_checkpoint: Path = DEFAULT_CELEBRATION):
        self.kick = PPOPolicy(kick_checkpoint); self.celebration = PPOPolicy(celebration_checkpoint)
        assert self.kick.obs_dim in (49, OBS_DIM) and self.celebration.obs_dim == OBS_DIM
        self.meta = {"kick": self.kick.meta, "celebration": self.celebration.meta}
        self.checkpoint = kick_checkpoint
        self.call_count = 0

    def reset(self):
        self.kick.reset(); self.celebration.reset(); self.call_count = 0

    def act(self, observation: np.ndarray) -> np.ndarray:
        obs = np.asarray(observation, dtype=np.float32)
        if obs.shape != (OBS_DIM,):
            raise ValueError(f"expected observation shape {(OBS_DIM,)}, got {obs.shape}")
        self.call_count += 1
        if obs[-2] > 0.5:                       # scored flag set: celebration policy controls the rest of the episode
            return self.celebration.act(obs)
        return act_any(self.kick, obs)


class ZeroActionPolicy:
    """Neutral baseline: every leg joint held at its home target. Expected NOT to kick the ball."""
    label = "zero-action baseline"

    def __init__(self):
        self.call_count = 0

    def reset(self):
        self.call_count = 0

    def act(self, observation: np.ndarray) -> np.ndarray:
        if np.asarray(observation).shape != (OBS_DIM,):
            raise ValueError("bad observation shape")
        self.call_count += 1
        return np.zeros(ACT_DIM, dtype=np.float32)

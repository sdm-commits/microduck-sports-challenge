"""PPO training entry point.

Example (smoke):   python -m duck_kick.train --total-steps 200000 --num-envs 16 --num-workers 4 --out runs/smoke
Example (full):    python -m duck_kick.train --total-steps 20000000 --num-envs 64 --num-workers 16 --out runs/full
Writes: <out>/metrics.jsonl (one line per iteration), <out>/ckpt_<iter>.pt, <out>/final.pt, <out>/config.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .config import EnvConfig, PPOConfig
from .policy import save_checkpoint
from .ppo import ActorCritic, RunningMeanStd, compute_gae, ppo_update
from .vec_env import VecEnv


def parse(argv=None) -> PPOConfig:
    p = argparse.ArgumentParser()
    d = PPOConfig()
    p.add_argument("--total-steps", type=int, default=d.total_steps)
    p.add_argument("--num-envs", type=int, default=d.num_envs)
    p.add_argument("--num-workers", type=int, default=d.num_workers)
    p.add_argument("--rollout-steps", type=int, default=d.rollout_steps)
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--lr", type=float, default=d.lr)
    p.add_argument("--checkpoint-every", type=int, default=d.checkpoint_every_iters)
    p.add_argument("--out", type=Path, default=Path("runs/default"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--init-checkpoint", type=Path, default=None, help="warm start weights + obs stats from a checkpoint")
    p.add_argument("--reset-log-std", type=float, default=None, help="with --init-checkpoint: reset exploration log-std")
    p.add_argument("--env", choices=["kick", "celebration", "approach"], default="kick",
                   help="celebration: learner controls only the post-goal phase; the frozen --kick-checkpoint plays the shot")
    p.add_argument("--kick-checkpoint", type=Path, default=None)
    p.add_argument("--env-overrides", type=str, default=None,
                   help='JSON of dotted EnvConfig fields, e.g. {"randomization.ball_x": [0.06, 0.13], "celebration.enabled": false}')
    a = p.parse_args(argv)
    cfg = PPOConfig(total_steps=a.total_steps, num_envs=a.num_envs, num_workers=a.num_workers,
                    rollout_steps=a.rollout_steps, seed=a.seed, lr=a.lr, checkpoint_every_iters=a.checkpoint_every)
    return cfg, a.out, a.device, a.init_checkpoint, a.reset_log_std, a.env, a.kick_checkpoint, a.env_overrides


def apply_overrides(cfg, overrides: dict):
    for key, val in (overrides or {}).items():
        obj = cfg; parts = key.split(".")
        for part in parts[:-1]:
            obj = getattr(obj, part)
        assert hasattr(obj, parts[-1]), f"unknown EnvConfig field {key}"
        setattr(obj, parts[-1], val)
    return cfg


def train(cfg: PPOConfig, out: Path, device: str = "cpu", init_checkpoint: Path = None, reset_log_std: float = None,
          env_kind: str = "kick", kick_checkpoint: Path = None, env_overrides: str = None):
    out.mkdir(parents=True, exist_ok=True)
    env_cfg = apply_overrides(EnvConfig(), json.loads(env_overrides) if env_overrides else {})
    (out / "config.json").write_text(json.dumps({"env": env_cfg.to_dict(), "ppo": cfg.to_dict(), "device": device,
                                                "init_checkpoint": str(init_checkpoint) if init_checkpoint else None,
                                                "reset_log_std": reset_log_std, "env_kind": env_kind, "env_overrides": env_overrides,
                                                "kick_checkpoint": str(kick_checkpoint) if kick_checkpoint else None}, indent=2))
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    dev = torch.device(device)
    venv = VecEnv(env_cfg, cfg.num_envs, cfg.num_workers, seed=cfg.seed, env_kind=env_kind, kick_checkpoint=kick_checkpoint)
    OBS_D, ACT_D = venv.obs_dim, venv.act_dim
    model = ActorCritic(OBS_D, ACT_D, cfg.hidden, cfg.init_log_std, cfg.min_log_std).to(dev)
    obs_rms = RunningMeanStd((OBS_D,))
    if init_checkpoint is not None:
        ck = torch.load(init_checkpoint, map_location="cpu", weights_only=False)
        from .warmstart import load_padded
        load_padded(model, obs_rms, ck)
        if reset_log_std is not None:
            with torch.no_grad():
                model.log_std.fill_(float(reset_log_std))
        print(f"warm start from {init_checkpoint} (step {ck.get('meta', {}).get('step')}), log_std reset to {reset_log_std}")
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, eps=1e-5)

    T, N = cfg.rollout_steps, cfg.num_envs
    n_iters = max(1, cfg.total_steps // (T * N))
    obs = venv.reset()
    obs_rms.update(obs)
    log = open(out / "metrics.jsonl", "a")
    ep_stats, t0, global_step = [], time.time(), 0
    print(f"PPO: {n_iters} iterations x {T*N} steps, device={device}, envs={N}, workers={cfg.num_workers}")
    for it in range(1, n_iters + 1):
        frac = 1.0 - (it - 1) / n_iters
        for g in opt.param_groups:
            g["lr"] = cfg.lr * (cfg.lr_final_frac + (1 - cfg.lr_final_frac) * frac)
        buf_obs = np.zeros((T, N, OBS_D), np.float32); buf_act = np.zeros((T, N, ACT_D), np.float32)
        buf_logp = np.zeros((T, N), np.float32); buf_val = np.zeros((T, N), np.float32)
        buf_rew = np.zeros((T, N), np.float32); buf_done = np.zeros((T, N), np.float32)
        raw_obs = []
        for t in range(T):
            nobs = obs_rms.normalize(obs).astype(np.float32)
            with torch.no_grad():
                a, lp, v = model.act(torch.as_tensor(nobs, device=dev))
            a_np = a.cpu().numpy()
            buf_obs[t], buf_act[t], buf_logp[t], buf_val[t] = nobs, a_np, lp.cpu().numpy(), v.cpu().numpy()
            obs, rew, done, infos = venv.step(np.clip(a_np, -1, 1))
            buf_rew[t], buf_done[t] = rew, done
            raw_obs.append(obs)
            for inf in infos:
                if inf is not None:
                    ep_stats.append(inf["episode"])
        global_step += T * N
        obs_rms.update(np.concatenate(raw_obs))
        with torch.no_grad():
            last_v = model.value(torch.as_tensor(obs_rms.normalize(obs).astype(np.float32), device=dev)).cpu().numpy()
        adv, ret = compute_gae(buf_rew, buf_val, buf_done, last_v, cfg.gamma, cfg.gae_lambda)
        batch = {"obs": torch.as_tensor(buf_obs.reshape(T * N, -1), device=dev),
                 "act": torch.as_tensor(buf_act.reshape(T * N, -1), device=dev),
                 "logp": torch.as_tensor(buf_logp.reshape(-1), device=dev),
                 "adv": torch.as_tensor(adv.reshape(-1), device=dev),
                 "ret": torch.as_tensor(ret.reshape(-1), device=dev)}
        stats = ppo_update(model, opt, batch, cfg)

        if ep_stats:
            dx = np.array([e["ball_dx"] for e in ep_stats]); fell = np.array([e["fell"] for e in ep_stats])
            contact = np.array([e["contact"] for e in ep_stats])
            kicked = (dx >= 0.15) & contact & ~fell
            on_target = np.array([e.get("on_target", False) for e in ep_stats]); goal = np.array([e.get("goal", False) for e in ep_stats])
            celeb = [e["celebration_rmse_rad"] for e in ep_stats if e.get("celebration_rmse_rad") is not None]
            trig = [e for e in ep_stats if e.get("triggered")]
            summary = {"episodes": len(ep_stats), "ball_dx_mean": float(dx.mean()), "contact_rate": float(contact.mean()),
                       "fall_rate": float(fell.mean()), "kick_rate": float(kicked.mean()),
                       "on_target_rate": float(on_target.mean()), "goal_rate": float(goal.mean()),
                       "celebration_rmse_mean": float(np.mean(celeb)) if celeb else None,
                       "trigger_rate": float(np.mean([bool(e.get("triggered")) for e in ep_stats])),
                       "pocket_err_at_trigger": float(np.mean([e["pocket_error_at_trigger"] for e in trig])) if trig else None,
                       "celebration_head_yaw_deg_mean": float(np.mean([e["celebration_max_head_yaw_deg"] for e in ep_stats if e.get("celebrated")] or [0.0])),
                       "ep_len_mean": float(np.mean([e["steps"] for e in ep_stats]))}
        else:
            summary = {"episodes": 0}
        rec = {"iter": it, "step": global_step, "reward_per_step": float(buf_rew.mean()),
               "elapsed_s": time.time() - t0, "sps": global_step / (time.time() - t0),
               "log_std_mean": float(model.log_std.mean().item()), **stats, **summary}
        log.write(json.dumps(rec) + "\n"); log.flush()
        if it % 5 == 0 or it == 1:
            print(f"it {it}/{n_iters} step {global_step} r/step {rec['reward_per_step']:.3f} "
                  f"kick {summary.get('kick_rate', 0):.2f} goal {summary.get('goal_rate', 0):.2f} on_tgt {summary.get('on_target_rate', 0):.2f} "
                  f"celebRMSE {summary.get('celebration_rmse_mean') or 0:.3f} trig {summary.get('trigger_rate', 0):.2f} pocket {summary.get('pocket_err_at_trigger') or 0:.3f} "
                  f"fall {summary.get('fall_rate', 0):.2f} kl {stats['approx_kl']:.4f} sps {rec['sps']:.0f}")
        ep_stats = []
        if it % cfg.checkpoint_every_iters == 0:
            save_checkpoint(out / f"ckpt_{it:05d}.pt", model, obs_rms, env_cfg, cfg, {"iter": it, "step": global_step})
    save_checkpoint(out / "final.pt", model, obs_rms, env_cfg, cfg, {"iter": n_iters, "step": global_step,
                                                                     "train_seconds": time.time() - t0})
    log.close(); venv.close()
    print(f"done: {global_step} steps in {time.time()-t0:.0f}s -> {out/'final.pt'}")


if __name__ == "__main__":
    cfg, out, device, init_ck, reset_ls, env_kind, kick_ck, ovr = parse()
    train(cfg, out, device, init_ck, reset_ls, env_kind, kick_ck, ovr)

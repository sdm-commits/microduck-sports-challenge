"""Deterministic shootout evaluation in the contact-enabled simulator, with optional video capture.

A shootout is N shots; shot i uses seed = first_seed + i and commanded target zone = i mod 3 (left, centre, right),
so every zone gets the same number of attempts and (checkpoint, seed) reproduces exactly. The policy acts
deterministically (mean action). Each seed also fixes the episode's domain randomization and observation noise.

  python -m duck_kick.evaluate --checkpoint duck_kick/checkpoints/final.pt --shots 30 --json eval.json
  python -m duck_kick.evaluate --policy zero --shots 30                      # neutral baseline, must not score
  python -m duck_kick.evaluate --checkpoint ... --video result.mp4 --video-shots 9   # scoreboard + two cameras
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .config import EnvConfig, KICK_MIN_TRAVEL_M
from .env import KickEnv, ACT_DIM
from .policy import PPOPolicy, ZeroActionPolicy, CombinedPolicy, DEFAULT_CHECKPOINT, DEFAULT_CELEBRATION


def is_kick(s: Dict[str, Any]) -> bool:
    return bool(s["contact"] and s["ball_dx"] >= KICK_MIN_TRAVEL_M and abs(s["ball_dy"]) < s["ball_dx"] and not s["fell"])


def shot_result(s: Dict[str, Any]) -> str:
    if s["fell"]:
        return "FELL"
    if s["on_target"]:
        return "ON TARGET"
    if s["goal"]:
        return "GOAL (other zone)"
    if s["miss"]:
        return "MISS"
    return "NO GOAL"


class Overlay:
    """Draws the scoreboard onto rendered frames (Pillow)."""

    def __init__(self, width: int, label: str = ""):
        from PIL import ImageFont
        self.font = ImageFont.load_default(size=26)
        self.small = ImageFont.load_default(size=18)
        self.label_font = ImageFont.load_default(size=22)
        self.width = width
        self.label = label

    def draw(self, frame: np.ndarray, shot: int, n_shots: int, target: str, score: int, status: str, t: float, pip: Optional[np.ndarray]):
        from PIL import Image, ImageDraw
        img = Image.fromarray(frame)
        if pip is not None:
            p = Image.fromarray(pip).resize((frame.shape[1] // 3, frame.shape[0] // 3))
            img.paste(p, (frame.shape[1] - p.width - 12, 12))
            ImageDraw.Draw(img).rectangle([frame.shape[1] - p.width - 13, 11, frame.shape[1] - 12, 12 + p.height], outline=(255, 255, 255), width=2)
        d = ImageDraw.Draw(img, "RGBA")
        d.rectangle([12, 12, 420, 108], fill=(0, 0, 0, 150))
        d.text((24, 18), f"Shot {shot}/{n_shots}   Target: {target.upper()}", font=self.font, fill=(255, 255, 255))
        d.text((24, 52), f"Score {score}  |  t = {t:4.2f} s", font=self.font, fill=(255, 230, 80))
        d.text((24, 84), status, font=self.small, fill=(255, 255, 255))
        if self.label:
            font = self.label_font
            if int(d.textlength(self.label, font=font)) > frame.shape[1] - 48:
                font = self.small          # shrink rather than run off the frame
            w = int(d.textlength(self.label, font=font))
            d.rectangle([12, frame.shape[0] - 66, 12 + w + 24, frame.shape[0] - 32], fill=(0, 0, 0, 150))
            d.text((24, frame.shape[0] - 62), self.label, font=font, fill=(120, 230, 255))
        d.text((12, frame.shape[0] - 26), "Trained PPO policy, MuJoCo simulation, Open Duck Mini v2. Not hardware.", font=self.small, fill=(255, 255, 255))
        return np.asarray(img)


def run_shot(policy, seed: int, target: int, randomize: bool = True, writer=None, overlay: Optional[Overlay] = None,
             shot_idx: int = 0, n_shots: int = 0, score: int = 0) -> Dict[str, Any]:
    cfg = EnvConfig()
    env = KickEnv(cfg, seed=seed, randomize=randomize)
    try:
        obs = env.reset(seed=seed, target=target)
        policy.reset()
        acts, status = [], "..."
        def frame():
            main = env.render("side"); pip = env.render("behind")
            return overlay.draw(main, shot_idx, n_shots, cfg.goal.zone_names[target], score, status, env.t * cfg.control.ctrl_dt, pip)
        if writer is not None:
            for _ in range(15):          # 0.3 s hold on the starting pose so the commanded zone is readable
                writer.append_data(frame())
        done = False
        while not done:
            a = np.asarray(policy.act(obs), dtype=np.float64)
            assert a.shape == (ACT_DIM,) and np.all(np.isfinite(a)) and np.abs(a).max() <= 1.0 + 1e-6
            acts.append(a)
            obs, _, done, info = env.step(a)
            if writer is not None:
                if env.crossed and status == "...":
                    status = shot_result({**env.episode_summary()})
                    if env.goal:
                        status += "   duck celebrates!"
                    if env.on_target:
                        score += 1
                writer.append_data(frame())
        s = dict(info["episode"])
        s["on_target_at_crossing"] = s["on_target"]
        s["on_target"] = bool(s["on_target"] and not s["fell"])     # a fall anywhere in the episode voids the shot
        s["goal"] = bool(s["goal"] and not s["fell"])
        acts = np.stack(acts)
        s.update({"seed": seed, "randomized": randomize, "policy": policy.label, "policy_calls": policy.call_count,
                  "action_abs_max": float(np.abs(acts).max()), "action_abs_mean": float(np.abs(acts).mean()),
                  "kick": is_kick(s), "result": shot_result(s)})
        if writer is not None:
            status = s["result"]
            for _ in range(25):          # 0.5 s hold showing the result
                writer.append_data(frame())
        return s
    finally:
        env.close()


def aggregate(eps: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(eps)
    out = {"shots": n,
           "on_target": int(sum(e["on_target"] for e in eps)), "goals_any_zone": int(sum(e["goal"] for e in eps)),
           "misses": int(sum(e["miss"] for e in eps)), "no_goal": int(sum((not e["crossed_goal_line"]) for e in eps)),
           "falls": int(sum(e["fell"] for e in eps)), "contacts": int(sum(e["contact"] for e in eps)),
           "on_target_rate": float(np.mean([e["on_target"] for e in eps])),
           "kick_rate_legacy": float(np.mean([e["kick"] for e in eps])),
           "crossing_time_s_mean": float(np.mean([e["crossing_time_s"] for e in eps if e["crossing_time_s"] is not None] or [float("nan")])),
           "max_ball_speed_mean": float(np.mean([e["max_ball_speed"] for e in eps])),
           "celebrated": int(sum(e.get("celebrated", False) for e in eps)),
           "celebration_rmse_rad_mean": float(np.mean([e["celebration_rmse_rad"] for e in eps if e.get("celebration_rmse_rad") is not None] or [float("nan")])),
           "celebration_max_head_yaw_deg_mean": float(np.mean([e["celebration_max_head_yaw_deg"] for e in eps if e.get("celebrated")] or [0.0])),
           "per_zone": {}}
    for z in sorted({e["target_name"] for e in eps}):
        zs = [e for e in eps if e["target_name"] == z]
        err = [abs(e["crossing_y"] - e["target_y"]) for e in zs if e["crossing_y"] is not None]
        out["per_zone"][z] = {"shots": len(zs), "on_target": int(sum(e["on_target"] for e in zs)),
                              "goals_any_zone": int(sum(e["goal"] for e in zs)),
                              "crossing_y_abs_error_mean": float(np.mean(err)) if err else None}
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--policy", choices=["ppo", "combined", "zero"], default="combined",
                   help="combined: kick checkpoint + celebration checkpoint with the goal-event switch; ppo: a single 63-obs checkpoint")
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--celebration-checkpoint", type=Path, default=DEFAULT_CELEBRATION)
    p.add_argument("--shots", type=int, default=30)
    p.add_argument("--first-seed", type=int, default=0)
    p.add_argument("--no-randomization", action="store_true", help="nominal physics, no obs noise")
    p.add_argument("--json", type=Path, default=None)
    p.add_argument("--video", type=Path, default=None)
    p.add_argument("--video-shots", type=int, default=9)
    p.add_argument("--video-first-seed", type=int, default=0)
    p.add_argument("--video-label", default="", help="section caption drawn on every video frame")
    a = p.parse_args(argv)

    if a.policy == "zero":
        policy = ZeroActionPolicy()
    elif a.policy == "combined":
        policy = CombinedPolicy(a.checkpoint, a.celebration_checkpoint)
    else:
        policy = PPOPolicy(a.checkpoint)
    randomize = not a.no_randomization
    eps = [run_shot(policy, a.first_seed + i, i % 3, randomize) for i in range(a.shots)]
    summary = aggregate(eps)
    result = {"policy": policy.label, "randomized": randomize, "first_seed": a.first_seed, "shots": a.shots,
              "target_rule": "shot i uses seed first_seed+i and zone i mod 3 (left, centre, right)",
              "on_target_rule": "foot touched ball; ball crossed x=goal_x inside the commanded zone below the crossbar; robot did not fall",
              "summary": summary, "episodes": eps}
    if a.policy != "zero":
        result["checkpoint"] = str(a.checkpoint)
        result["checkpoint_sha256"] = hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()
        if a.policy == "combined":
            result["celebration_checkpoint"] = str(a.celebration_checkpoint)
            result["celebration_checkpoint_sha256"] = hashlib.sha256(a.celebration_checkpoint.read_bytes()).hexdigest()
        result["checkpoint_meta"] = policy.meta
    if a.video is not None:
        import imageio.v2 as imageio
        a.video.parent.mkdir(parents=True, exist_ok=True)
        w = imageio.get_writer(a.video, fps=50, codec="libx264", quality=8, macro_block_size=None)
        ov = Overlay(960, a.video_label); score = 0; vid = []
        for i in range(a.video_shots):
            s = run_shot(policy, a.video_first_seed + i, i % 3, randomize, writer=w, overlay=ov,
                         shot_idx=i + 1, n_shots=a.video_shots, score=score)
            score += int(s["on_target"]); vid.append(s)
        w.close()
        result["video"] = {"path": str(a.video), "shots": a.video_shots, "first_seed": a.video_first_seed,
                           "score_on_target": score, "summary": aggregate(vid), "episodes": vid}
    print(json.dumps({"policy": policy.label, "randomized": randomize, "summary": summary}, indent=2))
    if a.json is not None:
        a.json.write_text(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

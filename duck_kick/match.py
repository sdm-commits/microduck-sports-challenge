"""Full 'walk up and score' match: approach policy -> kick policy -> celebration policy, all learned, joined by events.

    python -m duck_kick.match --shots 30 --json eval_match.json
    python -m duck_kick.match --shots 1 --video result_match.mp4 --video-shots 6
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from .approach_env import ApproachEnv
from .config import EnvConfig
from .evaluate import Overlay, shot_result, aggregate
from .policy import PPOPolicy, DEFAULT_CELEBRATION

DEFAULT_APPROACH = Path(__file__).resolve().parent / "checkpoints" / "approach.pt"
# The match uses a kick policy fine-tuned on post-walk states (handed over mid-stride, off-centre pocket); the
# standing shootout uses duck_kick/checkpoints/final.pt. Both are in the archive; see README.
DEFAULT_MATCH_KICK = Path(__file__).resolve().parent / "checkpoints" / "match_kick.pt"


class ZeroApproachPolicy:
    """Neutral baseline: no walking command and never triggers a kick. Expected to reach no goals at all."""
    label = "zero-action baseline"
    obs_dim = None

    def __init__(self):
        self.call_count = 0

    def reset(self):
        self.call_count = 0

    def act(self, observation):
        self.call_count += 1
        return np.array([0.0, 0.0, -1.0])


def run_match_shot(approach: PPOPolicy, kick_ckpt: Path, celeb_ckpt: Path, seed: int, target: int, randomize: bool = True,
                   writer=None, overlay: Optional[Overlay] = None, shot_idx: int = 0, n_shots: int = 0, score: int = 0) -> Dict[str, Any]:
    cfg = EnvConfig()
    cfg.control.episode_seconds = cfg.approach.max_seconds + cfg.approach.settle_seconds + 4.5   # approach + settle + kick/celebration
    env = ApproachEnv(cfg, seed=seed, randomize=randomize, kick_checkpoint=kick_ckpt)
    celebration = PPOPolicy(celeb_ckpt)
    status = {"txt": "walking up..."}
    def frame():
        main = env.render("wide"); pip = env.render("behind")
        return overlay.draw(main, shot_idx, n_shots, cfg.goal.zone_names[target], score, status["txt"], env.inner.t * cfg.control.ctrl_dt, pip)
    try:
        obs = env.reset(seed=seed, target=target); approach.reset()
        if writer is not None:
            for _ in range(15): writer.append_data(frame())
        done = False; k = 0; outcome = None
        while not done:
            a = approach.act(obs)
            trigger = a[2] > cfg.approach.trigger_threshold
            if trigger:
                # hand over inside the env, but with celebration and video frames
                err = float(np.linalg.norm(env.pocket_error()))
                env.triggered = True; env.learner_t += 1
                status["txt"] = "kick!"
                outcome = env.run_kick_phase(celebration=celebration, writer=writer, frame_fn=frame if writer is not None else None)
                done = True
                summ = env.inner.episode_summary(); summ.update({"triggered": True, "approach_steps": env.learner_t, "pocket_error_at_trigger": err, "kick_outcome": outcome, "disturbed": env.disturbed})
                info = {"episode": summ, "fell": bool(env.inner.fell), "timeout": False, "triggered": True}
            else:
                obs, r, done, info = env.step(np.array([a[0], a[1], -1.0]))
                if writer is not None:
                    writer.append_data(frame())    # one frame per learner step (10 Hz) during the walk
            k += 1
        s = dict(info["episode"])
        oc = s.get("kick_outcome") or {}
        s["on_target"] = bool(oc.get("on_target", 0.0) and not s["fell"] and not s.get("disturbed"))
        s["goal"] = bool(oc.get("goal", 0.0) and not s["fell"] and not s.get("disturbed"))
        s["contact"] = bool(oc.get("contact", 0.0))
        s.update({"seed": seed, "randomized": randomize, "policy": "approach + kick + celebration (learned, event-switched)",
                  "policy_calls": approach.call_count, "kick": bool(s["contact"] and s["ball_dx"] >= 0.15 and not s["fell"]),
                  "result": shot_result(s) if s["triggered"] else "NO SHOT (timeout)"})
        if writer is not None:
            status["txt"] = s["result"] + ("   duck celebrates!" if s["goal"] else "")
            for _ in range(8): writer.append_data(frame())
        return s
    finally:
        env.close()


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--approach-checkpoint", type=Path, default=DEFAULT_APPROACH)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_MATCH_KICK)
    p.add_argument("--policy", choices=["learned", "zero"], default="learned",
                   help="zero: neutral baseline that never walks and never triggers a kick")
    p.add_argument("--celebration-checkpoint", type=Path, default=DEFAULT_CELEBRATION)
    p.add_argument("--shots", type=int, default=30)
    p.add_argument("--first-seed", type=int, default=0)
    p.add_argument("--no-randomization", action="store_true")
    p.add_argument("--json", type=Path, default=None)
    p.add_argument("--video", type=Path, default=None)
    p.add_argument("--video-shots", type=int, default=6)
    p.add_argument("--video-first-seed", type=int, default=0)
    p.add_argument("--video-label", default="", help="section caption drawn on every video frame")
    a = p.parse_args(argv)
    approach = ZeroApproachPolicy() if a.policy == "zero" else PPOPolicy(a.approach_checkpoint)
    randomize = not a.no_randomization
    eps = [run_match_shot(approach, a.checkpoint, a.celebration_checkpoint, a.first_seed + i, i % 3, randomize) for i in range(a.shots)]
    summary = aggregate(eps)
    summary.update({"triggered": int(sum(e["triggered"] for e in eps)),
                    "pocket_error_at_trigger_mean": float(np.mean([e["pocket_error_at_trigger"] for e in eps if e["triggered"]] or [float("nan")])),
                    "approach_seconds_mean": float(np.mean([e["approach_steps"] * 0.1 for e in eps]))})
    result = {"mode": "match", "policy": approach.label if a.policy == "zero" else "learned approach + kick + celebration",
              "shots": a.shots, "first_seed": a.first_seed, "summary": summary, "episodes": eps,
              "checkpoints": {k: {"path": str(v), "sha256": hashlib.sha256(v.read_bytes()).hexdigest()}
                              for k, v in {"approach": a.approach_checkpoint, "kick": a.checkpoint, "celebration": a.celebration_checkpoint}.items()},
              "walk_onnx_sha256": hashlib.sha256((Path(__file__).resolve().parent / "checkpoints" / "walk.onnx").read_bytes()).hexdigest()} if a.policy != "zero" else {
              "mode": "match", "policy": approach.label, "shots": a.shots, "first_seed": a.first_seed,
              "summary": summary, "episodes": eps}
    if a.video is not None:
        import imageio.v2 as imageio
        w = imageio.get_writer(a.video, fps=25, codec="libx264", quality=8, macro_block_size=None)
        ov = Overlay(960, a.video_label); score = 0; vid = []
        for i in range(a.video_shots):
            s = run_match_shot(approach, a.checkpoint, a.celebration_checkpoint, a.video_first_seed + i, i % 3, randomize,
                               writer=w, overlay=ov, shot_idx=i + 1, n_shots=a.video_shots, score=score)
            score += int(s["on_target"]); vid.append(s)
        w.close()
        result["video"] = {"path": str(a.video), "shots": a.video_shots, "score_on_target": score, "summary": aggregate(vid), "episodes": vid}
    print(json.dumps({"mode": "match", "summary": summary}, indent=2))
    if a.json is not None:
        a.json.write_text(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

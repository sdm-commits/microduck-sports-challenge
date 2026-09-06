"""Walk-up-and-score task for the learned approach policy.

The learner outputs walking commands (forward speed, turn rate) and a "kick now" trigger at 10 Hz. Three frozen
learned controllers do the rest inside the same physics episode: the Open Duck walking policy (ONNX) turns commands
into 50 Hz joint targets, the kick policy plays the shot after the trigger, and (at evaluation) the celebration
policy runs after a goal. Nothing is teleported: the robot's start pose is set only at reset.

Learner observation (25): ball rel. base in heading frame (2), ball rel. velocity (2), pocket error (2),
heading (cos, sin) (2), gyro z (1), projected gravity (3), base velocity in heading frame (2), previous command (3),
walker gait phase (2), time fraction (1), commanded zone one-hot (3), target point rel. base in heading frame (2).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import mujoco
import numpy as np

from .config import EnvConfig
from .env import KickEnv, yaw_from_quat
from .policy import PPOPolicy, act_any, DEFAULT_CHECKPOINT
from .walk_policy import WalkPolicy, DEFAULT_WALK_ONNX

APPROACH_OBS_DIM = 25
APPROACH_ACT_DIM = 3


class ApproachEnv:
    def __init__(self, cfg: Optional[EnvConfig] = None, seed: int = 0, randomize: Optional[bool] = None,
                 kick_checkpoint: Optional[Path] = None, walk_onnx: Optional[Path] = None):
        self.cfg = cfg or EnvConfig()
        self.ap = self.cfg.approach
        self.inner = KickEnv(self.cfg, seed=seed, randomize=randomize)
        self.rng = self.inner.rng
        self.walk = WalkPolicy(walk_onnx or DEFAULT_WALK_ONNX, steps_in_period=27)
        self.kick = PPOPolicy(kick_checkpoint or DEFAULT_CHECKPOINT)
        self.max_learner_steps = int(round(self.ap.max_seconds / (self.cfg.control.ctrl_dt * self.ap.cmd_hold_steps)))
        self.reset()

    # ------------------------------------------------------------------ helpers
    def _heading_frame(self):
        d = self.inner.data
        yaw = yaw_from_quat(d.qpos[3:7])
        c, s = np.cos(-yaw), np.sin(-yaw)
        return yaw, np.array([[c, -s], [s, c]])

    def _ball_rel(self):
        d, e = self.inner.data, self.inner
        yaw, R = self._heading_frame()
        rel = R @ (d.qpos[e.ball_qpos:e.ball_qpos + 2] - d.qpos[0:2])
        relv = R @ (d.qvel[e.ball_qvel:e.ball_qvel + 2] - d.qvel[0:2])
        return yaw, R, rel, relv

    def pocket_error(self):
        _, _, rel, _ = self._ball_rel()
        return rel - np.array([self.ap.pocket_x, self.ap.pocket_y])

    def _observe(self) -> np.ndarray:
        d, e = self.inner.data, self.inner
        yaw, R, rel, relv = self._ball_rel()
        err = rel - np.array([self.ap.pocket_x, self.ap.pocket_y])
        base_v = R @ d.qvel[0:2]
        tgt = R @ (e.target_point - d.qpos[0:2])
        onehot = np.zeros(3); onehot[e.target] = 1.0
        phase = np.array([np.cos(self.walk.phase_i / self.walk.steps_in_period * 2 * np.pi),
                          np.sin(self.walk.phase_i / self.walk.steps_in_period * 2 * np.pi)])
        obs = np.concatenate([rel, relv, err, [np.cos(yaw), np.sin(yaw)], [d.qvel[5]], e._projected_gravity(),
                              base_v, self.prev_cmd, phase, [self.learner_t / self.max_learner_steps], onehot, tgt]).astype(np.float32)
        assert obs.shape == (APPROACH_OBS_DIM,)
        return obs

    def _walk_step(self, cmd3):
        """One 50 Hz control step of the frozen walker under the given (vx, vy, yaw_rate) command."""
        e = self.inner
        command = np.zeros(7); command[:3] = cmd3
        obs_w = self.walk.observe(e.model, e.data, command, e.joint_qpos, e.joint_qvel, e.feet_contacts())
        targets = self.walk.act(obs_w)
        e.data.ctrl[:] = targets
        for _ in range(e.n_substeps):
            mujoco.mj_step(e.model, e.data)
        e.t += 1
        return e._fallen()

    # ------------------------------------------------------------------ API
    def reset(self, seed: Optional[int] = None, target: Optional[int] = None) -> np.ndarray:
        e = self.inner
        e.reset(seed=seed, target=target)
        self.rng = e.rng
        ap = self.ap
        bx = self.rng.uniform(*ap.ball_x); by = self.rng.uniform(*ap.ball_y)
        e.data.qpos[e.ball_qpos:e.ball_qpos + 2] = [bx, by]
        e.data.qvel[e.ball_qvel:e.ball_qvel + 6] = 0.0
        x = bx - self.rng.uniform(*ap.start_dist)
        y = by + self.rng.uniform(-ap.start_lateral, ap.start_lateral)
        yaw = self.rng.uniform(-ap.start_yaw, ap.start_yaw)
        e.set_base_pose(x, y, yaw)
        e.ball_start = e.data.qpos[e.ball_qpos:e.ball_qpos + 3].copy(); e.prev_ball = e.ball_start.copy()
        self.walk.reset()
        self.prev_cmd = np.zeros(3)
        self.learner_t = 0
        self.prev_err = float(np.linalg.norm(self.pocket_error()))
        self.triggered = False
        self.outcome = None
        self.ball_ref = e.data.qpos[e.ball_qpos:e.ball_qpos + 2].copy()
        self.disturbed = False
        return self._observe()

    def _check_disturbed(self):
        e = self.inner
        if np.linalg.norm(e.data.qpos[e.ball_qpos:e.ball_qpos + 2] - self.ball_ref) > self.ap.disturb_dist:
            self.disturbed = True
        return self.disturbed

    def step(self, action: np.ndarray):
        ap, e = self.ap, self.inner
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        cmd = np.array([ap.vx_max * a[0], 0.0, ap.yaw_rate_max * a[1]])
        trigger = a[2] > ap.trigger_threshold
        fell = False
        for _ in range(ap.cmd_hold_steps):
            if self._walk_step(cmd):
                fell = True; break
        self.learner_t += 1
        self.prev_cmd = a.copy()
        self._check_disturbed()
        err = float(np.linalg.norm(self.pocket_error()))
        yaw = yaw_from_quat(e.data.qpos[3:7])
        reward = (ap.w_progress * float(np.clip(self.prev_err - err, -0.1, 0.1)) + ap.w_heading * (np.cos(yaw) - 1.0) + ap.w_time)
        self.prev_err = err
        info = {"fell": fell, "timeout": False, "triggered": False}
        done = False
        if fell:
            reward += ap.w_fall; done = True
        elif trigger:
            self.triggered = True
            reward += ap.w_trigger_err * err + ap.w_trigger_yaw * float(yaw ** 2)
            if err > ap.far_trigger_skip:
                outcome = {"contact": 0.0, "goal": 0.0, "on_target": 0.0, "fell": 0.0, "crossed": 0.0, "skipped": True}
                self.outcome = outcome
            else:
                outcome = self.run_kick_phase()
            if self.disturbed:                      # ball was moved before the kick policy took over: shot voided
                reward += ap.w_disturb
                outcome = {**outcome, "goal": 0.0, "on_target": 0.0, "voided_disturbed": True}
                self.outcome = outcome
            reward += (ap.w_contact * outcome["contact"] + ap.w_goal * outcome["goal"] + ap.w_on_target * outcome["on_target"]
                       + ap.w_kick_fall * outcome["fell"])
            info["triggered"] = True; info["kick_outcome"] = outcome; done = True
        elif self.learner_t >= self.max_learner_steps:
            reward += ap.w_no_shot; info["timeout"] = True; done = True
        if done:
            summ = e.episode_summary()
            summ.update({"triggered": self.triggered, "approach_steps": self.learner_t, "disturbed": self.disturbed,
                         "pocket_error_at_trigger": err if self.triggered else None,
                         "kick_outcome": info.get("kick_outcome")})
            info["episode"] = summ
        return self._observe(), float(reward), done, info

    def run_kick_phase(self, celebration: Optional[PPOPolicy] = None, writer=None, frame_fn=None):
        """Settle with zero command, then let the frozen kick policy play until the ball crosses the line, the robot
        falls, or kick_seconds elapse. Returns the outcome dict. With a celebration policy, continues to the episode end."""
        ap, e = self.ap, self.inner
        for _ in range(int(round(ap.settle_seconds / self.cfg.control.ctrl_dt))):
            self._walk_step(np.zeros(3))
            if writer is not None: writer.append_data(frame_fn())
        self._check_disturbed()
        self.kick_start_time = float(e.data.time)
        e.begin_kick_phase()
        self.kick.reset()
        obs = e._observe()
        contact = False
        for k in range(int(round(ap.kick_seconds / self.cfg.control.ctrl_dt))):
            a = act_any(self.kick, obs)
            obs, _, done, info = e.step(a)
            contact = contact or info["contact"]
            if writer is not None: writer.append_data(frame_fn())
            ball_speed = float(np.linalg.norm(e.data.qvel[e.ball_qvel:e.ball_qvel + 2]))
            ball_stopped = contact and k > 25 and ball_speed < 0.05
            if e.fell or done or (celebration is None and (e.crossed or ball_stopped)):
                break
        kick_contact = bool(contact)
        crossed_after_kick = bool(e.crossed and e.crossing_time is not None and e.crossing_time >= self.kick_start_time)
        outcome = {"contact": float(kick_contact), "goal": float(e.goal and kick_contact and crossed_after_kick),
                   "on_target": float(e.on_target and kick_contact and crossed_after_kick),
                   "fell": float(e.fell), "crossed": float(crossed_after_kick), "kick_start_time_s": self.kick_start_time}
        if celebration is not None and not e.fell:
            celebration.reset()
            # the celebration policy was trained to control from the goal to the end of a 4.5 s shootout episode
            # (about 3.5 s); the match episode ends the same 3.5 s after the goal (or after the kick window if no goal)
            post_goal_steps = int(round(3.5 / self.cfg.control.ctrl_dt))
            end_step = (e.goal_step + post_goal_steps) if e.goal_step is not None else e.t
            while e.t < min(e.max_steps, end_step) and not e.fell:
                a = celebration.act(obs) if obs[-2] > 0.5 else act_any(self.kick, obs)
                obs, _, done, info = e.step(a)
                if writer is not None: writer.append_data(frame_fn())
                if done: break
            outcome["fell"] = float(e.fell)
        self.outcome = outcome
        return outcome

    @property
    def data(self):
        return self.inner.data

    def render(self, *a, **k):
        return self.inner.render(*a, **k)

    def close(self):
        self.inner.close()

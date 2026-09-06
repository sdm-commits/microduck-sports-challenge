"""Closed-loop MuJoCo environment for the Microduck soccer kick.

Observation -> policy -> 10 leg joint position targets -> MuJoCo position servos -> rigid-body dynamics
with contacts (feet <-> floor, feet <-> ball, ball <-> floor/goal posts/net). Nothing is teleported: after reset
the ball moves only because a foot hits it. The policy is commanded a target zone (left/centre/right) of a goal
1 m ahead and is rewarded for putting the ball there.

Action latency: exactly one control step. The target computed from the action given to step() at
control step t is sent to the servos during control step t+1 (the previous target is applied at t).
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import mujoco
import numpy as np

from .config import (EnvConfig, SCENE_XML, HOME_POSE, LEG_JOINT_IDX, JOINT_NAMES, FOOT_GEOMS,
                     FOOT_SITES)
from .reference import build_reference
from .celebration import celebration_offsets

OBS_DIM = 3 + 3 + 10 + 4 + 10 + 4 + 14 + 3 + 3 + 2 + 3 + 2 + 2   # 63
ACT_DIM = 14


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, q)
    return m.reshape(3, 3)


def yaw_from_quat(q: np.ndarray) -> float:
    w, x, y, z = q
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


class KickEnv:
    def __init__(self, cfg: Optional[EnvConfig] = None, seed: int = 0, randomize: Optional[bool] = None):
        self.cfg = cfg or EnvConfig()
        self.randomize = self.cfg.randomization.enabled if randomize is None else randomize
        for attempt in range(5):        # many workers load the same files at once; retry transient failures
            try:
                self.model = mujoco.MjModel.from_xml_path(SCENE_XML)
                break
            except Exception:
                if attempt == 4:
                    raise
                import time; time.sleep(0.5 + attempt)
        self.model.opt.timestep = self.cfg.control.sim_dt
        self.data = mujoco.MjData(self.model)
        self.rng = np.random.default_rng(seed)
        c = self.cfg.control
        self.n_substeps = int(round(c.ctrl_dt / c.sim_dt))
        assert abs(self.n_substeps * c.sim_dt - c.ctrl_dt) < 1e-9
        assert c.action_latency_steps == 1, "this environment implements exactly one step of latency"
        self.max_steps = int(round(c.episode_seconds / c.ctrl_dt))
        self.reference = build_reference(self.cfg.reference, c.ctrl_dt, self.max_steps + 1)

        self.home = np.asarray(HOME_POSE, dtype=np.float64)
        self.home_legs = self.home[LEG_JOINT_IDX]
        self.leg_idx = np.asarray(LEG_JOINT_IDX)
        self.ctrl_low = self.model.actuator_ctrlrange[:, 0].copy()
        self.ctrl_high = self.model.actuator_ctrlrange[:, 1].copy()

        self.ball_jnt = self.model.joint("ball_free")
        self.ball_qpos = int(self.ball_jnt.qposadr[0])
        self.ball_qvel = int(self.ball_jnt.dofadr[0])
        self.joint_qpos = np.array([self.model.joint(n).qposadr[0] for n in JOINT_NAMES])
        self.joint_qvel = np.array([self.model.joint(n).dofadr[0] for n in JOINT_NAMES])
        self.leg_qpos = self.joint_qpos[self.leg_idx]
        self.leg_qvel = self.joint_qvel[self.leg_idx]
        self.head_idx = np.array([i for i in range(14) if i not in LEG_JOINT_IDX])
        self.head_qpos = self.joint_qpos[self.head_idx]
        self.head_qvel = self.joint_qvel[self.head_idx]
        self.home_head = self.home[self.head_idx]
        self.ball_geom = self.model.geom("ball").id
        self.floor_geom = self.model.geom("floor").id
        self.foot_geom = {k: self.model.geom(v).id for k, v in FOOT_GEOMS.items()}
        self.foot_geom_set = set(self.foot_geom.values())
        self.foot_site = {k: self.model.site(v).id for k, v in FOOT_SITES.items()}
        self.ball_body = self.model.body("ball").id
        self.robot_bodies = [b for b in range(1, self.model.nbody) if b != self.ball_body]
        self.zone_geoms = [self.model.geom(f"zone_{n}").id for n in self.cfg.goal.zone_names]
        self.zone_rgba_off = self.model.geom_rgba[self.zone_geoms[0]].copy()
        self.zone_rgba_on = np.array([1.0, 0.9, 0.1, 0.9])
        self.forced_target = None

        self._nom = dict(
            body_mass=self.model.body_mass.copy(),
            body_inertia=self.model.body_inertia.copy(),
            gainprm=self.model.actuator_gainprm.copy(),
            biasprm=self.model.actuator_biasprm.copy(),
            dof_frictionloss=self.model.dof_frictionloss.copy(),
            floor_friction=self.model.geom_friction[self.floor_geom].copy(),
        )
        self._renderer = None
        self.state_bank = None
        if self.cfg.randomization.state_bank_path:
            b = np.load(self.cfg.randomization.state_bank_path)
            self.state_bank = {"qpos": b["qpos"], "qvel": b["qvel"], "target": b["target"]}
        self.reset()

    # ------------------------------------------------------------------ randomization
    def _apply_randomization(self):
        m, n = self.model, self._nom
        r = self.cfg.randomization
        m.body_mass[:] = n["body_mass"]; m.body_inertia[:] = n["body_inertia"]
        m.actuator_gainprm[:] = n["gainprm"]; m.actuator_biasprm[:] = n["biasprm"]
        m.dof_frictionloss[:] = n["dof_frictionloss"]
        m.geom_friction[self.floor_geom] = n["floor_friction"]
        self.dr_sample = {}
        if not self.randomize:
            return
        u = self.rng.uniform
        m.geom_friction[self.floor_geom, 0] = u(*r.floor_friction)
        for b in self.robot_bodies:
            s = u(*r.body_mass_scale)
            m.body_mass[b] = n["body_mass"][b] * s
            m.body_inertia[b] = n["body_inertia"][b] * s
        sb = u(*r.ball_mass_scale)
        m.body_mass[self.ball_body] = n["body_mass"][self.ball_body] * sb
        m.body_inertia[self.ball_body] = n["body_inertia"][self.ball_body] * sb
        kps = u(*r.kp_scale, size=m.nu)
        m.actuator_gainprm[:, 0] = n["gainprm"][:, 0] * kps
        m.actuator_biasprm[:, 1] = n["biasprm"][:, 1] * kps
        dofs = self.joint_qvel
        m.dof_frictionloss[dofs] = n["dof_frictionloss"][dofs] * u(*r.joint_frictionloss_scale, size=len(dofs))
        self.dr_sample = {"floor_friction": float(m.geom_friction[self.floor_geom, 0]),
                          "ball_mass_scale": float(sb), "kp_scale_mean": float(kps.mean())}

    # ------------------------------------------------------------------ reset / step
    def reset(self, seed: Optional[int] = None, target: Optional[int] = None) -> np.ndarray:
        """target: commanded zone index (0 left, 1 centre, 2 right); sampled uniformly when None."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._apply_randomization()
        r = self.cfg.randomization
        g = self.cfg.goal
        self.target = (int(self.rng.choice(len(g.zone_centers_y), p=np.asarray(r.target_probs) / np.sum(r.target_probs)))
                       if target is None else int(target))
        self.target_point = np.array([g.goal_x, g.zone_centers_y[self.target]])
        for i, gid in enumerate(self.zone_geoms):          # visual highlight only
            self.model.geom_rgba[gid] = self.zone_rgba_on if i == self.target else self.zone_rgba_off
        m, d, r = self.model, self.data, self.cfg.randomization
        mujoco.mj_resetData(m, d)
        d.qpos[0:3] = [0.0, 0.0, 0.165]
        yaw0 = self.rng.uniform(-r.init_yaw, r.init_yaw) if (self.randomize and r.init_yaw > 0) else 0.0
        d.qpos[3:7] = [np.cos(yaw0 / 2), 0.0, 0.0, np.sin(yaw0 / 2)]
        if self.randomize and r.init_base_vel > 0:
            d.qvel[0:2] = self.rng.uniform(-r.init_base_vel, r.init_base_vel, size=2)
        q = self.home.copy()
        if self.randomize:
            q += self.rng.uniform(-r.init_joint_noise, r.init_joint_noise, size=q.shape)
        d.qpos[self.joint_qpos] = q
        bx = self.rng.uniform(*r.ball_x) if self.randomize else float(np.mean(r.ball_x))
        by = self.rng.uniform(*r.ball_y) if self.randomize else float(np.mean(r.ball_y))
        d.qpos[self.ball_qpos:self.ball_qpos + 3] = [bx, by, self.cfg.ball_radius]
        d.qpos[self.ball_qpos + 3:self.ball_qpos + 7] = [1.0, 0.0, 0.0, 0.0]
        d.ctrl[:] = self.home
        self.from_bank = False
        if self.state_bank is not None and self.randomize and self.rng.uniform() < r.state_bank_prob:
            k = int(self.rng.integers(len(self.state_bank["qpos"])))
            d.qpos[:] = self.state_bank["qpos"][k]; d.qvel[:] = self.state_bank["qvel"][k]
            d.ctrl[:] = np.clip(d.qpos[self.joint_qpos], self.ctrl_low, self.ctrl_high)
            if target is None:
                self.target = int(self.state_bank["target"][k]); self.target_point = np.array([g.goal_x, g.zone_centers_y[self.target]])
                for i, gid in enumerate(self.zone_geoms):
                    self.model.geom_rgba[gid] = self.zone_rgba_on if i == self.target else self.zone_rgba_off
            # bank states are stored in the same world frame as the shootout (goal at x = 1.0), no transform needed
            self.from_bank = True
        mujoco.mj_forward(m, d)

        self.t = 0
        self.phase_origin = 0                     # control step at which the kick phase started (0 unless an approach phase preceded it)
        self.prev_action = np.zeros(ACT_DIM)
        self.delayed_ctrl = (np.clip(d.ctrl.copy(), self.ctrl_low, self.ctrl_high) if self.from_bank else self.home.copy())
        self.first_contact_done = False
        self.first_contact_time = None
        self.contact_steps = 0
        self.contact_substeps = 0
        self.ball_start = d.qpos[self.ball_qpos:self.ball_qpos + 3].copy()
        self.prev_ball = self.ball_start.copy()
        self.max_ball_speed = 0.0
        self.fell = False
        self.prev_dist = float(np.linalg.norm(self.ball_start[:2] - self.target_point))
        self.crossed = False
        self.goal = False
        self.on_target = False
        self.miss = False
        self.crossing_time = None
        self.crossing_y = None
        self.goal_step = None
        self.celeb_err_sq = []
        self.max_head_yaw = 0.0
        self.bounce_amp = 0.0
        return self._observe()

    def _targets_from_action(self, action: np.ndarray) -> np.ndarray:
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        ctrl = self.home + self.cfg.control.action_scale * a          # all 14 joints
        return np.clip(ctrl, self.ctrl_low, self.ctrl_high)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        m, d = self.model, self.data
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        # one control step of latency: apply the target computed from the PREVIOUS action
        d.ctrl[:] = self.delayed_ctrl
        self.delayed_ctrl = self._targets_from_action(a)
        contact = False
        for _ in range(self.n_substeps):
            mujoco.mj_step(m, d)
            if self._foot_ball_contact():          # sampled at every physics substep
                contact = True
                self.contact_substeps += 1
        self.t += 1
        if contact:
            self.contact_steps += 1
            if self.first_contact_time is None:
                self.first_contact_time = float(d.time)

        reward, terms = self._reward(a, contact)
        self.prev_action = a
        ball = d.qpos[self.ball_qpos:self.ball_qpos + 3]
        self.max_ball_speed = max(self.max_ball_speed,
                                  float(np.linalg.norm(d.qvel[self.ball_qvel:self.ball_qvel + 3])))
        self.prev_ball = ball.copy()
        if contact:
            self.first_contact_done = True

        fell = self._fallen()
        self.fell = self.fell or fell
        timeout = self.t >= self.max_steps
        if fell:
            reward += self.cfg.reward.termination
            terms["termination"] = self.cfg.reward.termination
        done = fell or timeout
        info = {"terms": terms, "fell": fell, "timeout": timeout, "contact": contact}
        if done:
            info["episode"] = self.episode_summary()
        return self._observe(), float(reward), done, info

    def begin_kick_phase(self):
        """Called when a preceding approach phase hands over to the kick policy: restart the kick-local clock and the
        per-kick bookkeeping so the frozen kick policy sees the same observation statistics as in its own training."""
        d = self.data
        self.phase_origin = self.t
        self.prev_action = np.zeros(ACT_DIM)
        self.delayed_ctrl = np.clip(d.ctrl.copy(), self.ctrl_low, self.ctrl_high)
        self.first_contact_done = False
        self.ball_start = d.qpos[self.ball_qpos:self.ball_qpos + 3].copy()
        self.prev_ball = self.ball_start.copy()
        self.prev_dist = float(np.linalg.norm(self.ball_start[:2] - self.target_point))

    def feet_contacts(self):
        """(left, right) foot-to-floor contact flags (used by the walking policy)."""
        d, m = self.data, self.model
        out = [False, False]
        for i in range(d.ncon):
            c = d.contact[i]
            for k, g in enumerate((self.foot_geom["left"], self.foot_geom["right"])):
                if (c.geom1 == g and c.geom2 == self.floor_geom) or (c.geom2 == g and c.geom1 == self.floor_geom):
                    out[k] = True
        return out

    def set_base_pose(self, x: float, y: float, yaw: float):
        """Reset-time placement of the robot (used by the approach task). Not used during an episode."""
        d = self.data
        d.qpos[0:2] = [x, y]
        d.qpos[3:7] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
        d.qvel[:6] = 0.0
        mujoco.mj_forward(self.model, d)

    # ------------------------------------------------------------------ helpers
    def _foot_ball_contact(self) -> bool:
        d = self.data
        for i in range(d.ncon):
            c = d.contact[i]
            if ((c.geom1 == self.ball_geom and c.geom2 in self.foot_geom_set)
                    or (c.geom2 == self.ball_geom and c.geom1 in self.foot_geom_set)):
                return True
        return False

    def _projected_gravity(self) -> np.ndarray:
        R = quat_to_mat(self.data.qpos[3:7])
        return R.T @ np.array([0.0, 0.0, -1.0])

    def _fallen(self) -> bool:
        tc = self.cfg.termination
        return bool(self.data.qpos[2] < tc.min_base_height
                    or self._projected_gravity()[2] > tc.max_tilt_gravity_z)

    def _reward(self, a: np.ndarray, contact: bool):
        d, w = self.data, self.cfg.reward
        q_all = d.qpos[self.joint_qpos]
        q_ref = self.home.copy(); q_ref[self.leg_idx] = self.reference[min(max(self.t - self.phase_origin, 0), len(self.reference) - 1)]
        celebrating = False
        if self.goal_step is not None and self.cfg.celebration.enabled:
            ts = (self.t - self.goal_step) * self.cfg.control.ctrl_dt
            if 0.0 <= ts < self.cfg.celebration.duration:
                q_ref = self.home + celebration_offsets(self.cfg.celebration, ts); celebrating = True
                self.celeb_err_sq.append(float(np.mean((q_all - q_ref) ** 2)))
                self.max_head_yaw = max(self.max_head_yaw, abs(float(d.qpos[self.joint_qpos[7]])))
                self.bounce_amp = max(self.bounce_amp, abs(float(d.qvel[2])))
        ball = d.qpos[self.ball_qpos:self.ball_qpos + 3]
        bvel = d.qvel[self.ball_qvel:self.ball_qvel + 2]
        g = self._projected_gravity()
        foot = d.site_xpos[self.foot_site["right"]]
        dist = float(np.linalg.norm(ball[:2] - self.target_point))
        progress = float(np.clip(self.prev_dist - dist, -0.05, 0.05))
        self.prev_dist = dist
        speed = float(np.linalg.norm(bvel))
        aim = 0.0
        if speed > 0.2 and dist > 0.05:
            to_t = (self.target_point - ball[:2]) / dist
            aim = float(np.dot(bvel / speed, to_t)) - 1.0          # 0 when heading straight at the target
        event = 0.0
        if not self.crossed and ball[0] >= self.cfg.goal.goal_x:
            self.crossed = True
            self.crossing_time = float(d.time); self.crossing_y = float(ball[1])
            gc = self.cfg.goal
            in_mouth = abs(ball[1]) <= gc.half_width and ball[2] <= gc.crossbar_z
            if in_mouth:
                self.goal_step = self.t
            if in_mouth and abs(ball[1] - self.target_point[1]) <= gc.zone_half_width:
                self.goal = self.on_target = True; event = w.goal_on_target
            elif in_mouth:
                self.goal = True; event = w.goal_other_zone
            else:
                self.miss = True; event = w.miss
            if w.aim_precision:
                event += w.aim_precision * float(np.exp(-((ball[1] - self.target_point[1]) / w.aim_precision_width) ** 2))
        terms = {
            "imitation": w.imitation * float(np.exp(-w.imitation_k * np.mean((q_all - q_ref) ** 2))),
            "liveliness": (w.liveliness * (min(abs(float(d.qvel[self.joint_qvel[7]])), 3.0) / 3.0 + min(abs(float(d.qvel[2])), 0.3) / 0.3)
                           if celebrating else 0.0),
            "target_progress": w.target_progress * progress,
            "aim": w.aim * aim,
            "goal_event": event,
            "foot_contact_bonus": w.foot_contact_bonus if (contact and not self.first_contact_done) else 0.0,
            "approach": (w.approach * float(np.exp(-np.linalg.norm(foot - ball) / 0.05))
                         if not self.first_contact_done else 0.0),
            "tilt": -w.tilt * float(g[0] ** 2 + g[1] ** 2),
            "height": -w.height * float(abs(d.qpos[2] - 0.16)),
            "heading": -w.heading * float((yaw_from_quat(d.qpos[3:7])
                                            - np.arctan2(self.target_point[1] - d.qpos[1], self.target_point[0] - d.qpos[0])) ** 2),
            "action_rate": -w.action_rate * float(np.mean((a - self.prev_action) ** 2)),
            "torque": -w.torque * float(np.mean(d.actuator_force ** 2)),
            "joint_vel": -w.joint_vel * float(np.mean(d.qvel[self.leg_qvel] ** 2)),
            "alive": w.alive,
        }
        return float(sum(terms.values())), terms

    def _observe(self) -> np.ndarray:
        d, o = self.data, self.cfg.obs
        noise = self.randomize
        u = (lambda s, n: self.rng.uniform(-s, s, size=n)) if noise else (lambda s, n: np.zeros(n))
        gyro = d.qvel[3:6].copy() + u(o.noise_gyro, 3)             # base angular velocity, body frame
        grav = self._projected_gravity() + u(o.noise_gravity, 3)
        qpos = d.qpos[self.leg_qpos] - self.home_legs + u(o.noise_joint_pos, 10)
        hpos = d.qpos[self.head_qpos] - self.home_head + u(o.noise_joint_pos, 4)
        qvel = d.qvel[self.leg_qvel] + u(o.noise_joint_vel, 10)
        hvel = d.qvel[self.head_qvel] + u(o.noise_joint_vel, 4)
        if self.goal_step is not None:
            ts = (self.t - self.goal_step) * self.cfg.control.ctrl_dt
            celeb = np.array([1.0, min(ts / self.cfg.celebration.duration, 1.0)])
        else:
            celeb = np.zeros(2)
        yaw = yaw_from_quat(d.qpos[3:7])
        cy, sy = np.cos(-yaw), np.sin(-yaw)
        Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
        ball_rel = Rz @ (d.qpos[self.ball_qpos:self.ball_qpos + 3] - d.qpos[0:3]) + u(o.noise_ball_pos, 3)
        ball_vel = Rz @ (d.qvel[self.ball_qvel:self.ball_qvel + 3] - d.qvel[0:3]) + u(o.noise_ball_vel, 3)
        t = (self.t - self.phase_origin) * self.cfg.control.ctrl_dt
        phase = np.array([min(t / self.cfg.reference.t_recover_end, 1.0), t / self.cfg.control.phase_horizon_seconds])
        onehot = np.zeros(len(self.cfg.goal.zone_centers_y)); onehot[self.target] = 1.0
        tgt = Rz @ np.array([self.target_point[0] - d.qpos[0], self.target_point[1] - d.qpos[1], 0.0])
        obs = np.concatenate([gyro * o.gyro_scale, grav, qpos, hpos, qvel * o.joint_vel_scale, hvel * o.joint_vel_scale,
                              self.prev_action, ball_rel, ball_vel * o.ball_vel_scale, phase, onehot, tgt[:2], celeb]).astype(np.float32)
        assert obs.shape == (OBS_DIM,)
        return obs

    def episode_summary(self) -> Dict:
        ball = self.data.qpos[self.ball_qpos:self.ball_qpos + 3]
        disp = ball - self.ball_start
        return {
            "ball_dx": float(disp[0]), "ball_dy": float(disp[1]),
            "ball_start": [float(v) for v in self.ball_start],
            "max_ball_speed": self.max_ball_speed,
            "contact": bool(self.contact_steps > 0), "contact_steps": int(self.contact_steps),
            "contact_substeps": int(self.contact_substeps),
            "first_contact_time_s": self.first_contact_time,
            "fell": bool(self.fell), "steps": int(self.t),
            "final_base_height": float(self.data.qpos[2]),
            "final_yaw_deg": float(np.degrees(yaw_from_quat(self.data.qpos[3:7]))),
            "target_zone": int(self.target), "target_name": self.cfg.goal.zone_names[self.target],
            "target_y": float(self.target_point[1]),
            "goal": bool(self.goal), "on_target": bool(self.on_target), "miss": bool(self.miss),
            "crossed_goal_line": bool(self.crossed), "crossing_time_s": self.crossing_time, "crossing_y": self.crossing_y,
            "final_ball_xy": [float(v) for v in self.data.qpos[self.ball_qpos:self.ball_qpos + 2]],
            "celebrated": bool(self.celeb_err_sq), "celebration_steps": len(self.celeb_err_sq),
            "celebration_rmse_rad": float(np.sqrt(np.mean(self.celeb_err_sq))) if self.celeb_err_sq else None,
            "celebration_max_head_yaw_deg": float(np.degrees(self.max_head_yaw)), "celebration_max_base_vz": self.bounce_amp,
            "dr_sample": self.dr_sample,
        }

    # ------------------------------------------------------------------ rendering
    def render(self, camera: str = "side", width: int = 960, height: int = 540) -> np.ndarray:
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=height, width=width)
        self._renderer.update_scene(self.data, camera=camera)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

"""Wrapper for the Open Duck Mini v2 joystick walking policy (ONNX exported by the authors' MJX/Brax trainer).

Observation contract (mirrors Open_Duck_Playground/playground/open_duck_mini_v2/mujoco_infer.py):
  [gyro(3), accelerometer(3) with +1.3 on x, command(c), joint_angles - home(14), joint_vel * 0.05 (14),
   last_action(14), last_last_action(14), last_last_last_action(14), motor_targets(14), feet contacts(2),
   gait phase (cos, sin)(2)]
Action: 14 normalized joint offsets; motor target = home + 0.25 * action, rate-limited to 5.24 rad/s.
The command dimension c is inferred from the ONNX input size (3 = vx, vy, yaw rate; 7 adds 4 head targets).
"""
from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from .config import HOME_POSE

DEFAULT_WALK_ONNX = Path(__file__).resolve().parent / "checkpoints" / "walk.onnx"
ACTION_SCALE = 0.25
DOF_VEL_SCALE = 0.05
MAX_MOTOR_VELOCITY = 5.24          # rad/s
CTRL_DT = 0.02


class WalkPolicy:
    label = "Open Duck joystick walking policy (ONNX)"

    def __init__(self, onnx_path: Path = DEFAULT_WALK_ONNX, steps_in_period: int = 50):
        import onnxruntime
        so = onnxruntime.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
        self.session = onnxruntime.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.obs_size = int(inp.shape[-1])
        self.cmd_dim = self.obs_size - (3 + 3 + 14 * 6 + 2 + 2)
        assert self.cmd_dim in (3, 7), f"unexpected walking-policy obs size {self.obs_size}"
        self.steps_in_period = steps_in_period          # gait phase steps per period (from the reference motion data)
        self.home = np.asarray(HOME_POSE, dtype=np.float64)
        self.reset()

    def reset(self):
        self.last = [np.zeros(14) for _ in range(3)]
        self.motor_targets = self.home.copy()
        self.prev_motor_targets = self.home.copy()
        self.phase_i = 0.0
        self.call_count = 0

    def observe(self, model: mujoco.MjModel, data: mujoco.MjData, command: np.ndarray,
                joint_qpos_adr: np.ndarray, joint_qvel_adr: np.ndarray, feet_contacts) -> np.ndarray:
        gyro = data.sensor("gyro").data.copy()
        acc = data.sensor("accelerometer").data.copy(); acc[0] += 1.3
        cmd = np.zeros(self.cmd_dim); cmd[:min(len(command), self.cmd_dim)] = command[:self.cmd_dim]
        q = data.qpos[joint_qpos_adr] - self.home
        qd = data.qvel[joint_qvel_adr] * DOF_VEL_SCALE
        phase = np.array([np.cos(self.phase_i / self.steps_in_period * 2 * np.pi),
                          np.sin(self.phase_i / self.steps_in_period * 2 * np.pi)])
        return np.concatenate([gyro, acc, cmd, q, qd, self.last[0], self.last[1], self.last[2],
                               self.motor_targets, np.asarray(feet_contacts, dtype=np.float64), phase]).astype(np.float32)

    def act(self, obs: np.ndarray) -> np.ndarray:
        """Returns the 14 motor targets (absolute joint positions) for the next control step."""
        self.phase_i = (self.phase_i + 1.0) % self.steps_in_period
        a = self.session.run(None, {self.input_name: obs[None, :]})[0][0].astype(np.float64)
        self.last = [a.copy(), self.last[0], self.last[1]]
        targets = self.home + ACTION_SCALE * a
        lim = MAX_MOTOR_VELOCITY * CTRL_DT
        targets = np.clip(targets, self.prev_motor_targets - lim, self.prev_motor_targets + lim)
        self.prev_motor_targets = targets.copy(); self.motor_targets = targets.copy()
        self.call_count += 1
        return targets

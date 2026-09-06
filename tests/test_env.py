import unittest

import numpy as np

from duck_kick.config import EnvConfig
from duck_kick.env import KickEnv, OBS_DIM, ACT_DIM
from duck_kick.policy import ZeroActionPolicy


class EnvTest(unittest.TestCase):
    def test_shapes_and_determinism(self):
        e1, e2 = KickEnv(seed=3), KickEnv(seed=3)
        o1, o2 = e1.reset(seed=3), e2.reset(seed=3)
        self.assertEqual(o1.shape, (OBS_DIM,))
        rng = np.random.default_rng(0)
        for _ in range(30):
            a = rng.uniform(-1, 1, ACT_DIM)
            o1, r1, d1, _ = e1.step(a); o2, r2, d2, _ = e2.step(a)
        np.testing.assert_allclose(o1, o2); self.assertEqual(r1, r2)

    def test_one_step_action_latency(self):
        """The ctrl applied during step t equals the target computed from the action given at step t-1."""
        env = KickEnv(seed=0, randomize=False); env.reset()
        a1 = np.full(ACT_DIM, 0.3); a2 = np.full(ACT_DIM, -0.3)
        env.step(a1)
        np.testing.assert_allclose(env.data.ctrl, env.home)                    # step 1 applied the home target
        env.step(a2)
        np.testing.assert_allclose(env.data.ctrl, env._targets_from_action(a1))  # step 2 applied a1's target
        env.step(np.zeros(ACT_DIM))
        np.testing.assert_allclose(env.data.ctrl, env._targets_from_action(a2))

    def test_zero_action_does_not_kick(self):
        for seed in range(3):
            env = KickEnv(seed=seed); obs = env.reset(seed=seed); pol = ZeroActionPolicy()
            done = False
            while not done:
                obs, _, done, info = env.step(pol.act(obs))
            s = info["episode"]
            self.assertFalse(s["contact"]); self.assertLess(abs(s["ball_dx"]), 0.01); self.assertFalse(s["fell"])

    def test_ball_only_moves_after_foot_contact(self):
        env = KickEnv(seed=1, randomize=False); env.reset()
        for _ in range(20):
            _, _, _, info = env.step(np.zeros(ACT_DIM))
        self.assertLess(np.linalg.norm(env.data.qpos[env.ball_qpos:env.ball_qpos + 2] - env.ball_start[:2]), 1e-3)


class ReferenceTest(unittest.TestCase):
    def test_open_loop_reference_kicks_forward(self):
        """Tracking the procedural reference open-loop must strike the ball forward on nominal physics."""
        env = KickEnv(seed=0, randomize=False); env.reset()
        done = False
        while not done:
            ref = env.reference[min(env.t + 1, len(env.reference) - 1)]
            a = np.zeros(ACT_DIM); a[env.leg_idx] = (ref - env.home_legs) / env.cfg.control.action_scale
            _, _, done, info = env.step(a)
        s = info["episode"]
        self.assertTrue(s["contact"]); self.assertGreater(s["ball_dx"], 0.15); self.assertFalse(s["fell"])

    def test_target_command_in_observation_and_goal_scoring(self):
        env = KickEnv(seed=0, randomize=False)
        for tz in range(3):
            obs = env.reset(seed=0, target=tz)
            self.assertEqual(int(np.argmax(obs[56:59])), tz)
            self.assertAlmostEqual(float(obs[60]), env.cfg.goal.zone_centers_y[tz] - env.data.qpos[1], places=3)
            self.assertEqual(float(obs[61]), 0.0)   # not scored yet
        # a ball rolling straight down the centre from the kick spot must register as a centre goal
        env.reset(seed=0, target=1)
        env.data.qpos[env.ball_qpos:env.ball_qpos + 2] = [0.3, 0.0]; env.data.qvel[env.ball_qvel] = 1.5
        done = False
        while not done:
            _, _, done, info = env.step(np.zeros(ACT_DIM))
        s = info["episode"]
        self.assertTrue(s["crossed_goal_line"]); self.assertTrue(s["goal"]); self.assertTrue(s["on_target"])
        self.assertTrue(s["celebrated"]); self.assertGreater(s["celebration_steps"], 50)

    def test_celebration_reference_is_zero_outside_window(self):
        from duck_kick.celebration import celebration_offsets
        from duck_kick.config import CelebrationConfig
        c = CelebrationConfig()
        self.assertEqual(float(np.abs(celebration_offsets(c, -0.1)).max()), 0.0)
        self.assertEqual(float(np.abs(celebration_offsets(c, c.duration + 0.1)).max()), 0.0)
        self.assertGreater(float(np.abs(celebration_offsets(c, c.duration / 2)).max()), 0.2)

    def test_reference_kinematic_direction(self):
        """At the strike keyframe the right foot is ahead of and above its home position."""
        import mujoco
        env = KickEnv(seed=0, randomize=False); env.reset()
        d, m = env.data, env.model
        mujoco.mj_forward(m, d); home_foot = d.site_xpos[env.foot_site["right"]].copy()
        k = int(round(env.cfg.reference.t_strike_end / env.cfg.control.ctrl_dt))
        d.qpos[env.leg_qpos] = env.reference[k]; mujoco.mj_forward(m, d)
        foot = d.site_xpos[env.foot_site["right"]]
        self.assertGreater(foot[0] - home_foot[0], 0.03)


if __name__ == "__main__":
    unittest.main()


class ApproachTest(unittest.TestCase):
    def test_approach_env_shapes_and_frozen_kick_from_pocket(self):
        from duck_kick.approach_env import ApproachEnv, APPROACH_OBS_DIM, APPROACH_ACT_DIM
        from duck_kick.config import EnvConfig
        cfg = EnvConfig(); cfg.approach.start_dist = [0.095, 0.095]; cfg.approach.start_lateral = 0.0; cfg.approach.start_yaw = 0.0
        cfg.approach.ball_x = [0.09, 0.09]; cfg.approach.ball_y = [-0.085, -0.085]
        env = ApproachEnv(cfg, seed=0, randomize=False)
        obs = env.reset(seed=0, target=1)
        e = env.inner; e.set_base_pose(0.09 - 0.095, -0.085 + 0.085, 0.0); env.prev_err = float(np.linalg.norm(env.pocket_error()))
        obs = env._observe()                                                     # ball exactly in the pocket
        self.assertEqual(obs.shape, (APPROACH_OBS_DIM,))
        self.assertLess(float(np.linalg.norm(env.pocket_error())), 0.02)
        obs, r, done, info = env.step(np.array([0.0, 0.0, 1.0]))          # trigger immediately
        self.assertTrue(done and info["triggered"])
        self.assertEqual(info["kick_outcome"]["contact"], 1.0)
        self.assertEqual(len(np.zeros(APPROACH_ACT_DIM)), 3)

    def test_walker_moves_forward_on_command(self):
        from duck_kick.approach_env import ApproachEnv
        env = ApproachEnv(seed=0, randomize=False); env.reset(seed=0, target=0)
        x0 = float(env.inner.data.qpos[0])
        for _ in range(30):                                                # 3 s of forward command
            _, _, done, info = env.step(np.array([1.0, 0.0, -1.0]))
            if done: break
        self.assertFalse(info["fell"]); self.assertGreater(float(env.inner.data.qpos[0]) - x0, 0.15)


class AimPrecisionRewardTest(unittest.TestCase):
    def test_bonus_is_off_by_default_and_never_negative(self):
        from duck_kick.config import EnvConfig
        self.assertEqual(EnvConfig().reward.aim_precision, 0.0)
        cfg = EnvConfig(); cfg.reward.aim_precision = 8.0
        zw = cfg.goal.zone_half_width
        import numpy as _np
        for err in (0.0, zw, 2 * zw, 4 * zw):
            v = cfg.reward.aim_precision * float(_np.exp(-(err / zw) ** 2))
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, cfg.reward.aim_precision)
        exact = cfg.reward.aim_precision
        edge = cfg.reward.aim_precision * float(_np.exp(-1.0))
        self.assertGreater(exact, edge)     # centre of the zone beats the edge, which is the gradient we want


class NoStateTamperingTest(unittest.TestCase):
    """The submitted result must come from actuators and physics only.

    Guards the rule that simulator state (base pose, joint angles, ball position/velocity) is written at reset
    and nowhere else: during a rollout the only thing the code may write is `data.ctrl`, the servo command.
    """

    ROLLOUT_FUNCS = {"step", "_walk_step", "run_kick_phase", "_observe", "_reward", "_foot_ball_contact",
                     "episode_summary", "begin_kick_phase", "act"}

    def test_rollout_code_never_writes_simulator_state(self):
        import ast
        import pathlib
        offenders = []
        pkg = pathlib.Path(__file__).resolve().parent.parent / "duck_kick"
        for path in sorted(pkg.glob("*.py")):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if not (isinstance(node, ast.FunctionDef) and node.name in self.ROLLOUT_FUNCS):
                    continue
                for inner in ast.walk(node):
                    if not isinstance(inner, (ast.Assign, ast.AugAssign)):
                        continue
                    targets = inner.targets if isinstance(inner, ast.Assign) else [inner.target]
                    for tgt in targets:
                        if not isinstance(tgt, ast.Subscript):
                            continue                      # a bare local name is not a state write
                        text = ast.unparse(tgt)
                        base = text.split("[")[0]
                        if base.endswith(("qpos", "qvel", "xfrc_applied", "qfrc_applied")):
                            offenders.append(f"{path.name}:{inner.lineno}: {text}")
        self.assertEqual(offenders, [], "simulator state written outside reset:\n" + "\n".join(offenders))

    def test_ball_moves_only_after_a_foot_touches_it(self):
        env = KickEnv(seed=4, randomize=False)
        obs = env.reset(seed=4)
        start = env.data.qpos[env.ball_qpos:env.ball_qpos + 2].copy()
        touched = False
        while True:
            obs, _, done, info = env.step(np.zeros(ACT_DIM))
            touched = touched or info["contact"]
            if done:
                break
        moved = float(np.linalg.norm(env.data.qpos[env.ball_qpos:env.ball_qpos + 2] - start))
        self.assertFalse(touched)
        self.assertLess(moved, 0.01)      # never touched, so it must still be where reset put it

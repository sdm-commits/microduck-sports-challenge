import tempfile
import unittest
from pathlib import Path

import numpy as np

from duck_kick.config import PPOConfig
from duck_kick.env import KickEnv, OBS_DIM, ACT_DIM
from duck_kick.policy import PPOPolicy, save_checkpoint
from duck_kick.ppo import ActorCritic, RunningMeanStd, compute_gae
from duck_kick.config import EnvConfig


class PPOTest(unittest.TestCase):
    def test_gae_matches_reference(self):
        r = np.array([[1.0], [1.0], [1.0]], np.float32); v = np.zeros((3, 1), np.float32); d = np.zeros((3, 1), np.float32)
        adv, ret = compute_gae(r, v, d, np.zeros(1, np.float32), gamma=0.5, lam=1.0)
        np.testing.assert_allclose(ret[:, 0], [1.75, 1.5, 1.0])

    def test_checkpoint_roundtrip_and_closed_loop(self):
        cfg = PPOConfig()
        model = ActorCritic(OBS_DIM, ACT_DIM, cfg.hidden, cfg.init_log_std)
        rms = RunningMeanStd((OBS_DIM,)); rms.update(np.random.default_rng(0).normal(size=(64, OBS_DIM)))
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "ck.pt"
            save_checkpoint(p, model, rms, EnvConfig(), cfg, {"iter": 0})
            pol = PPOPolicy(p)
            env = KickEnv(seed=0); obs = env.reset(seed=0)
            a = pol.act(obs); self.assertEqual(a.shape, (ACT_DIM,)); self.assertLessEqual(np.abs(a).max(), 1.0)
            # the policy output depends on the observation (closed loop), not only on time
            obs2 = obs.copy(); obs2[:3] += 1.0
            self.assertFalse(np.allclose(pol.act(obs), pol.act(obs2)))


if __name__ == "__main__":
    unittest.main()


class CombinedPolicyTest(unittest.TestCase):
    def test_kick_policy_projection_matches_49_layout(self):
        from duck_kick.policy import kick_obs_from_full, pad_kick_action, PPOPolicy, DEFAULT_CHECKPOINT
        env = KickEnv(seed=0, randomize=False); obs = env.reset(seed=0, target=2)
        k = kick_obs_from_full(obs)
        self.assertEqual(k.shape, (49,))
        self.assertEqual(int(np.argmax(k[44:47])), 2)                     # target one-hot lands where the kick policy expects it
        np.testing.assert_allclose(k[:6], obs[:6]); np.testing.assert_allclose(k[6:16], obs[6:16])
        a = pad_kick_action(np.ones(10)); self.assertEqual(a.shape, (14,)); self.assertEqual(float(a[5:9].sum()), 0.0)
        pol = PPOPolicy(DEFAULT_CHECKPOINT); self.assertIn(pol.obs_dim, (49, 63))

    def test_frozen_kick_still_scores_through_the_63_dim_env(self):
        from duck_kick.celebration_env import CelebrationEnv
        env = CelebrationEnv(seed=0, randomize=True)
        obs = env.reset(seed=0, target=1)
        self.assertGreater(float(obs[-2]), 0.5)                          # learner starts right after a goal
        self.assertTrue(env.inner.goal)


class NumpyInferenceTest(unittest.TestCase):
    def test_numpy_actor_matches_torch(self):
        from duck_kick.policy import PPOPolicy, DEFAULT_CHECKPOINT
        pol = PPOPolicy(DEFAULT_CHECKPOINT)
        rng = np.random.default_rng(0)
        for _ in range(20):
            obs = rng.normal(size=pol.obs_dim).astype(np.float32) * 2
            np.testing.assert_allclose(pol.act(obs), pol.act_torch(obs), atol=2e-5)

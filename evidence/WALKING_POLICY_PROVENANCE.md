# Walking policy provenance

`duck_kick/checkpoints/walk.onnx` is the Open Duck Mini v2 joystick walking policy trained by us with the authors'
trainer (Open_Duck_Playground commit b9be205, `playground/open_duck_mini_v2/runner.py --task flat_terrain
--num_timesteps 150000000`, MJX + Brax PPO with the bundled polynomial reference-motion data) on a HIM Compute
standard workspace (48 vCPU, NVIDIA L4) on 2026-09-04. Wall clock about 45 minutes for 151 M environment steps.

Environment pins used for that run (the repository's pyproject resolves to newer, API-incompatible releases):
`playground==0.0.4`, `jax[cuda12]==0.6.2`, `brax==0.12.4`, `mujoco==3.3.3`, `mujoco-mjx==3.3.3`, plus a vendored
`mujoco_playground/_src/collision.py` (the 10-line `geoms_colliding` helper removed upstream). Train with
`.venv/bin/python`, not `uv run`, which re-syncs the lockfile and undoes the pins.

The exported checkpoint at 97,320,960 steps (evaluation reward 297, the best of the run; `train_walk_reward_curve.txt`)
was selected after testing all exports in our MuJoCo scene: 0.57 m forward in 5 s on a 0.15 m/s command with 2 deg
of heading drift and no falls (`train_walk_stdout.txt` is the trainer's stdout). Its SHA-256 is in
`walk_onnx_sha256.txt`. The policy is used frozen (inference only) by the approach task; its observation contract is
reproduced in `duck_kick/walk_policy.py`.

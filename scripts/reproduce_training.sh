#!/usr/bin/env bash
# Reproduces the training pipeline behind the shipped checkpoints, in order.
#
# Honesty note: this runs the pipeline with the reward weights that ship in duck_kick/config.py today.
# The shipped checkpoint came from the same sequence historically, except that its very first run also had an
# aim penalty and a miss penalty that were later removed because they taught the robot to tap the ball gently
# (that run scored 9/60; see the Results table in README.md and training_evidence.json). Re-running this script
# therefore reproduces an equivalent policy, not a bit-identical checkpoint: PPO is stochastic and the first
# run's objective is deliberately no longer the default.
#
# Runtimes: the two large runs were done on a HIM Compute standard workspace (NVIDIA L4, ~10k steps/s);
# everything else on an 8-core Apple Silicon Mac (~8k steps/s). End to end is roughly 6 hours.
set -euo pipefail
cd "$(dirname "$0")/.."
DEVICE="${1:-cpu}"          # pass "cuda" on a GPU box
BIG_ENVS=64; BIG_WORKERS=8
if [ "$DEVICE" = "cuda" ]; then BIG_ENVS=256; BIG_WORKERS=64; fi

python -m unittest discover -s tests
mkdir -p runs

# 1. kick policy, from scratch
python -u -m duck_kick.train --total-steps 40000000 --num-envs $BIG_ENVS --num-workers $BIG_WORKERS \
  --rollout-steps 64 --seed 0 --out runs/repro_kick1 --device "$DEVICE"
# 2. warm start
python -u -m duck_kick.train --total-steps 20000000 --num-envs $BIG_ENVS --num-workers $BIG_WORKERS \
  --rollout-steps 64 --seed 1 --out runs/repro_kick2 --device "$DEVICE" \
  --init-checkpoint runs/repro_kick1/final.pt --reset-log-std -1.2
# 3. robustness to walk-up hand-over states (needs an approach policy; skipped unless one exists)
if [ -f duck_kick/checkpoints/approach.pt ]; then
  python tools/collect_state_bank.py --approach-checkpoint duck_kick/checkpoints/approach.pt \
    --episodes 400 --first-seed 9000 --out runs/repro_bank.npz
  python -u -m duck_kick.train --total-steps 16000000 --num-envs 64 --num-workers 8 --rollout-steps 64 --seed 41 \
    --out runs/repro_kick3 --device "$DEVICE" --init-checkpoint runs/repro_kick2/final.pt --reset-log-std -1.6 \
    --env-overrides '{"randomization.state_bank_path": "runs/repro_bank.npz", "randomization.state_bank_prob": 0.6, "celebration.enabled": false}'
  PREV=runs/repro_kick3
else
  PREV=runs/repro_kick2
fi
# 4-5. aim-precision refinement, two rounds
python -u -m duck_kick.train --total-steps 20000000 --num-envs 64 --num-workers 8 --rollout-steps 64 --seed 81 \
  --out runs/repro_kick4 --device "$DEVICE" --init-checkpoint $PREV/final.pt --reset-log-std -1.6 \
  --env-overrides '{"reward.aim_precision": 8.0, "celebration.enabled": false}'
python -u -m duck_kick.train --total-steps 20000000 --num-envs 64 --num-workers 8 --rollout-steps 64 --seed 86 \
  --out runs/repro_kick5 --device "$DEVICE" --init-checkpoint runs/repro_kick4/final.pt --reset-log-std -1.7 \
  --env-overrides '{"reward.aim_precision": 8.0, "celebration.enabled": false}'

# 6. celebration policy, kick frozen
python -u -m duck_kick.train --env celebration --kick-checkpoint runs/repro_kick5/final.pt \
  --total-steps 12000000 --num-envs 64 --num-workers 8 --seed 5 --out runs/repro_celeb1 --device "$DEVICE"
python -u -m duck_kick.train --env celebration --kick-checkpoint runs/repro_kick5/final.pt \
  --init-checkpoint runs/repro_celeb1/final.pt --reset-log-std -1.4 \
  --total-steps 14000000 --num-envs 64 --num-workers 8 --seed 6 --out runs/repro_celeb2 --device "$DEVICE"

python -m duck_kick.evaluate --checkpoint runs/repro_kick5/final.pt \
  --celebration-checkpoint runs/repro_celeb2/final.pt --shots 60 --json runs/repro_eval.json
echo "reproduction finished; see runs/repro_eval.json"

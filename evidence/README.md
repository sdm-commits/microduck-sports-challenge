# Evidence index

Every number quoted in `README.md` comes from a file here. Nothing is summarised by hand: the JSON files are the raw
output of `duck_kick.evaluate` and `duck_kick.match`, and the `.jsonl` files are one record per PPO iteration written
during training. Runs that were rejected are kept and labelled, because the reasoning behind the shipped choices is
only checkable if the failures are visible too.


## Shipped result: standing penalty shootout

- `eval_shootout60_combined_final.json` - the headline. Kick + celebration, 60 shots, seeds 0-59. **58/60 on target, 59 goals, 0 falls**
- `eval_shootout300_combined_final.json` - same controller, 300 independent shots, seeds 3000-3299. **284/300 on target, 297 goals, 1 falls**
- `eval_shootout60_zero_action.json` - neutral baseline on the same task; never touches the ball. **0/60 on target, 0 goals, 0 falls**
- `eval_video_9shots_final.json` - the 9 shots in result.mp4, seeds 0-8. **8/9 on target, 8 goals, 0 falls**

## Choosing the shipped kick: A/B tests on unseen seeds

- `ab_kick_refined_300.json` - refined kick, 300 shots, seeds 1000-1299. **285/300 on target, 300 goals, 0 falls**
- `ab_kick_previous_300.json` - previous kick, same 300 seeds. **274/300 on target, 297 goals, 1 falls**
- `ab_kick_refined_600.json` - refined kick, 600 more shots, seeds 2000-2599. **571/600 on target, 596 goals, 1 falls**
- `ab_kick_previous_600.json` - previous kick, same 600 seeds. **538/600 on target, 596 goals, 1 falls**
- `ab_sharpened_kernel_FAILED_900.json` - REJECTED variant: sharper aim kernel, 900 shots, seeds 5000-5899. **808/900 on target, 888 goals, 8 falls**
- `ab_shipped_kick_900_seeds5000.json` - shipped kick on those same 900 seeds, for comparison. **848/900 on target, 881 goals, 3 falls**

## Walk-up match (documented extra, not the headline)

- `eval_match60_learned.json` - approach + kick + celebration, 60 walk-up shots. **31/60 on target, 41 goals, 0 falls**
- `eval_match15_zero_action_baseline.json` - neutral baseline: never walks, never shoots. **0/15 on target, 0 goals, 0 falls**
- `eval_match30_cycle3.json` - 30-shot match after the final co-training round. **15/30 on target, 21 goals, 0 falls**
- `eval_match30_approach_v1_with_stage2_kick.json` - 30-shot match from the first approach policy, kept to show the progression. **0/30 on target, 1 goals, 9 falls**
- `eval_shootout60_match_kick_standing.json` - the match kick judged on the standing shootout (it trades standing accuracy for hand-over robustness). **38/60 on target, 56 goals, 1 falls**

## Earlier checkpoints and failures

- `eval_shootout60_stage1_checkpoint.json` - first run from scratch: its aim penalty taught the robot to tap the ball gently. **26/60 on target, 26 goals, 0 falls**
- `eval_shootout60_ppo.json` - an early aimed checkpoint. **56/60 on target, 60 goals, 0 falls**
- `eval_shootout60_ppo_on_him_machine.json` - the same evaluation run on the HIM machine rather than the Mac. **56/60 on target, 60 goals, 0 falls**
- `eval_shootout60_kick_stage4_standing.json` - kick after the post-walk state-bank run. **55/60 on target, 58 goals, 0 falls**
- `eval_shootout60_single_network_both_phases_FAILED.json` - REJECTED: one network for kick and celebration; it learned to avoid scoring. **16/60 on target, 28 goals, 1 falls**

## Training metrics, one JSON object per PPO iteration

- `train_metrics_stage1_scratch.jsonl` - run 1, from scratch, broken reward. **40 M steps, 2441 iterations**
- `train_metrics_stage2_warmstart.jsonl` - run 2, penalties removed, log-std floored. **20 M steps, 1220 iterations**
- `train_metrics_kick_stage4_postwalk_bank.jsonl` - run 3, resets drawn from real walk-up hand-over states. **16 M steps, 1302 iterations**
- `train_metrics_kick_stage5_aim_precision.jsonl` - run 4, aim-precision bonus added. **20 M steps, 4882 iterations**
- `train_metrics_kick_stage6_aim_precision.jsonl` - run 5, same objective continued; this produced the shipped kick. **20 M steps, 4882 iterations**
- `train_metrics_kick_stage7_sharpened_FAILED.jsonl` - REJECTED run with the sharpened kernel. **20 M steps, 4882 iterations**
- `train_metrics_celebration_stage1.jsonl` - celebration, window only. **12 M steps, 2929 iterations**
- `train_metrics_celebration_stage2.jsonl` - celebration, goal to end of episode. **14 M steps, 3417 iterations**
- `train_metrics_kick_match_specialist.jsonl` - the match kick specialist. **24 M steps, 5859 iterations**
- `train_metrics_approach_final.jsonl` - the walk-up approach policy. **6 M steps, 1953 iterations**

## Configuration, provenance and inputs

- `train_config_stage1.json` - full EnvConfig and PPOConfig for run 1.
- `train_config_stage2.json` - same for run 2.
- `train_config_kick_stage4.json` - same for run 3.
- `train_config_kick_match_specialist.json` - same for the match kick.
- `train_config_celebration_stage2.json` - same for the celebration.
- `train_config_approach_final.json` - same for the approach.
- `WALKING_POLICY_PROVENANCE.md` - how the walking network was trained, and with whose trainer.
- `train_walk_stdout.txt` - walking training log.
- `train_walk_reward_curve.txt` - walking reward curve.
- `walk_onnx_sha256.txt` - hash of the shipped walking network.
- `train_stdout_stage2.txt` - stdout of run 2.
- `state_bank_postwalk.npz` - 400 real hand-over states used as kick-training resets (74 KB, produced by tools/collect_state_bank.py).

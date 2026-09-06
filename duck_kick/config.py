"""All environment, reference-motion, randomization and PPO settings in one explicit place.

Every number that affects training or evaluation lives here so a run can be reproduced from
this file plus the seed. Values are documented inline; units are SI (metres, seconds, radians).
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List
import os

ASSET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "open_duck_mini_v2")
SCENE_XML = os.path.join(ASSET_DIR, "scene_kick.xml")

# Joint order in the MJCF (14 actuated joints). Head joints are held at their home pose.
JOINT_NAMES: List[str] = [
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
    "neck_pitch", "head_pitch", "head_yaw", "head_roll",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
]
LEG_JOINT_IDX: List[int] = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13]   # indices into the 14-vector
LEG_JOINT_NAMES: List[str] = [JOINT_NAMES[i] for i in LEG_JOINT_IDX]
# "home" standing pose from Open_Duck_Playground scene_flat_terrain.xml keyframe.
HOME_POSE: List[float] = [
    0.002, 0.053, -0.63, 1.368, -0.784,
    0.0, 0.0, 0.0, 0.0,
    -0.003, -0.065, 0.635, 1.379, -0.796,
]
FOOT_GEOMS = {"left": "left_foot_bottom_tpu", "right": "right_foot_bottom_tpu"}
FOOT_SITES = {"left": "left_foot", "right": "right_foot"}


@dataclass
class ControlConfig:
    sim_dt: float = 0.002          # MuJoCo physics timestep (10 substeps per control step)
    ctrl_dt: float = 0.02          # 50 Hz policy / control rate
    action_scale: float = 1.0      # rad; joint target = home + action_scale * clip(a, -1, 1), then clipped to joint range
    action_latency_steps: int = 1  # exactly one control step: action computed at t is applied at t+1
    # Actuator model is the MJCF position servo (STS3215 fit from Open_Duck_Playground):
    # kp = 13.37 Nm/rad, kv = 0, force range +/-3.23 Nm, joint damping 0.56, frictionloss 0.068, armature 0.027.
    episode_seconds: float = 4.5   # 225 control steps: ~1 s kick + ball flight, then celebration
    phase_horizon_seconds: float = 4.5  # normalizer of the phase observation (kept fixed when longer match episodes are used)


@dataclass
class ReferenceConfig:
    """Procedural right-leg kick, defined as joint-space offsets from HOME_POSE (see reference.py).

    Signs were verified by forward kinematics on the model (tests/test_reference.py):
      right_hip_pitch < 0  -> right foot forward       right_knee > 0 -> right foot up / forward
      hip_roll > 0 on both legs -> base shifts over the left foot (weight shift for a right-foot kick)
    """
    # Timing/amplitude values were selected by a seeded random search over open-loop rollouts on
    # randomized seeds 0-29 (fast kick + recovery without falling). They are used only as a soft
    # imitation target; the policy may deviate from them.
    t_shift_end: float = 0.1494962882442667
    t_back_end: float = 0.31706176765951827
    t_strike_end: float = 0.37706176765951827
    t_recover_end: float = 0.6956114063486699
    weight_shift_roll: float = 0.07765302176608305
    swing_abduction_roll: float = -0.06541473223890773
    backswing_hip_pitch: float = 0.3700561571386918
    backswing_knee: float = 0.47838412836100996
    strike_hip_pitch: float = -1.0            # reachable: home 0.635 + (-1.0) > joint lower limit -0.524
    strike_knee: float = -0.2599733039034778
    strike_ankle: float = 0.08223375234152544
    left_hip_pitch_compensation: float = -0.05781432472883917


@dataclass
class GoalConfig:
    """Goal on the line x = goal_x, centred on y = 0. Three commanded target zones along the line."""
    goal_x: float = 1.0
    half_width: float = 0.375          # posts at y = +/- 0.375
    crossbar_z: float = 0.25
    zone_centers_y: List[float] = field(default_factory=lambda: [0.25, 0.0, -0.25])   # left, centre, right (robot faces +x, +y is its left)
    zone_half_width: float = 0.125
    zone_names: List[str] = field(default_factory=lambda: ["left", "centre", "right"])


@dataclass
class CelebrationConfig:
    """Procedural celebration reference tracked after the ball crosses the goal line (see celebration.py)."""
    enabled: bool = True
    duration: float = 1.6          # s of celebration after the goal event, then back to home
    ramp: float = 0.25             # s ease in / ease out
    bounce_hz: float = 2.0
    bounce_knee: float = 0.35      # rad knee flex pulse (both legs) -> vertical bounce
    bounce_hip_pitch: float = 0.12 # rad, sign per leg so the trunk stays level
    bounce_ankle: float = 0.10
    head_yaw_amp: float = 0.9      # rad, side-to-side sweep
    head_yaw_hz: float = 1.5
    head_pitch_amp: float = 0.35   # rad nod
    head_pitch_hz: float = 3.0
    head_roll_amp: float = 0.25
    neck_lift: float = -0.2        # rad, lift the head up


@dataclass
class ApproachConfig:
    """Walk-up task: the robot starts behind the ball and must bring the ball into the kick pocket, then trigger the kick."""
    start_dist: List[float] = field(default_factory=lambda: [0.30, 0.60])   # m behind the ball along -x
    start_lateral: float = 0.25          # m, half-width of the start lateral offset relative to the ball
    start_yaw: float = 0.5               # rad, half-width of the start heading
    ball_x: List[float] = field(default_factory=lambda: [0.0, 0.10])       # ball world position (goal at x = 1.0)
    ball_y: List[float] = field(default_factory=lambda: [-0.10, 0.10])
    pocket_x: float = 0.095              # kick pocket centre, ball relative to base in the heading frame
    pocket_y: float = -0.085
    cmd_hold_steps: int = 5              # learner acts at 10 Hz (5 control steps per command)
    max_seconds: float = 8.0             # approach time limit
    settle_seconds: float = 0.5          # zero command before the kick policy takes over
    kick_seconds: float = 3.0            # frozen kick policy runs at most this long (ends early when the ball crosses, stops, or the robot falls)
    trigger_threshold: float = 0.8       # action[2] must exceed this to trigger the kick (a deliberate output, not noise)
    vx_max: float = 0.15                 # walker command ranges (from the walking policy's training)
    yaw_rate_max: float = 1.0
    # learner reward
    w_progress: float = 10.0             # * decrease of the pocket error distance per learner step
    w_heading: float = 2.0               # * (cos(yaw) - 1)
    w_trigger_yaw: float = -3.0          # * yaw^2 (rad^2) at the trigger: the kick policy expects to face the goal
    w_time: float = -0.02                # per learner step
    w_fall: float = -5.0
    w_no_shot: float = -3.0              # timeout without triggering
    disturb_dist: float = 0.03           # m: ball moved by more than this before the kick policy takes over = disturbed shot
    w_disturb: float = -3.0              # penalty for a disturbed shot (its kick outcome is voided)
    w_trigger_err: float = -10.0         # * pocket error (m) at the trigger: firing from 0.3 m away costs 3, from 2 cm costs 0.2
    far_trigger_skip: float = 0.15       # m: triggers with a larger pocket error skip the kick simulation (outcome = nothing)
    w_contact: float = 2.0               # kick outcome: foot touched ball
    w_goal: float = 5.0                  #               goal in any zone
    w_on_target: float = 15.0            #               goal in the commanded zone
    w_kick_fall: float = -3.0


@dataclass
class ObsConfig:
    """Observation = [gyro(3), projected gravity(3), leg joint pos - home(10), head joint pos - home(4),
    leg joint vel(10), head joint vel(4), previous action(14), ball pos rel. to base in heading frame(3),
    ball vel in heading frame(3), phase(2) = (clip(t / t_recover_end, 0, 1), t / episode_seconds),
    commanded target zone one-hot(3), target point rel. to base in heading frame (x, y)(2),
    celebration(2) = (scored flag, clip(time since goal / celebration duration, 0, 1))]  -> 63 dims."""
    gyro_scale: float = 0.5
    joint_vel_scale: float = 0.1
    ball_vel_scale: float = 0.5
    # additive uniform noise half-widths (applied when randomization is enabled)
    noise_gyro: float = 0.10
    noise_gravity: float = 0.03
    noise_joint_pos: float = 0.02
    noise_joint_vel: float = 1.0
    noise_ball_pos: float = 0.005
    noise_ball_vel: float = 0.05


@dataclass
class RandomizationConfig:
    """Light domain randomization, sampled once per episode from the episode's RNG."""
    enabled: bool = True
    floor_friction: List[float] = field(default_factory=lambda: [0.5, 1.0])
    body_mass_scale: List[float] = field(default_factory=lambda: [0.9, 1.1])
    ball_mass_scale: List[float] = field(default_factory=lambda: [0.8, 1.2])
    kp_scale: List[float] = field(default_factory=lambda: [0.9, 1.1])
    joint_frictionloss_scale: List[float] = field(default_factory=lambda: [0.9, 1.1])
    target_probs: List[float] = field(default_factory=lambda: [0.5, 0.25, 0.25])  # training-time sampling of left/centre/right (left oversampled: hardest zone)
    init_yaw: float = 0.0                    # rad, half-width of the initial base yaw (0 = always facing +x)
    init_base_vel: float = 0.0               # m/s, half-width of the initial base xy velocity
    state_bank_path: str = ""                 # optional .npz of post-walk states (qpos, qvel, target); see tools/collect_state_bank.py
    state_bank_prob: float = 0.5              # probability of resetting from the bank instead of the standing pose
    init_joint_noise: float = 0.015           # rad, added to home pose at reset (0.03 tipped a passive stand on 1/40 seeds)
    ball_x: List[float] = field(default_factory=lambda: [0.08, 0.10])     # world x (robot faces +x); 1-3 cm gap to the foot
    ball_y: List[float] = field(default_factory=lambda: [-0.105, -0.065]) # in front of the right foot


@dataclass
class RewardConfig:
    """Per-control-step reward weights. target direction is world +x (the robot's initial heading)."""
    imitation: float = 2.0          # * exp(-imitation_k * mean((q - q_ref)^2)) over all 14 joints; ref = kick (legs) + head home before the goal, celebration after
    liveliness: float = 0.3         # * min(|head yaw velocity|, 3)/3 + min(bounce speed |base vz|, 0.3)/0.3, only during the celebration window
    imitation_k: float = 20.0
    target_progress: float = 20.0   # * clip(decrease of |ball - target point| per step, -0.05, 0.05)  (1 m => +20)
    aim: float = 0.0                # disabled: as a penalty active only above 0.2 m/s it rewarded soft taps; target_progress already rewards velocity toward the target
    goal_on_target: float = 15.0    # once: ball crosses the goal line inside the commanded zone and under the crossbar
    goal_other_zone: float = 1.0    # once: ball crosses the goal line inside the goal but in another zone
    miss: float = 0.0               # once: ball crosses x = goal_x outside the goal mouth (a penalty here taught soft taps; kept at 0)
    aim_precision_width: float = 0.125   # m, kernel width of the bonus below. At the default (= zone half-width) the
                                    # bonus is already 82% of max at the measured 5.5 cm mean error, so it saturates and
                                    # stops pulling; a narrower width keeps a gradient inside the zone.
    aim_precision: float = 0.0      # once, at the crossing: * exp(-((crossing_y - target_y) / aim_precision_width)^2).
                                    # Always >= 0, so unlike a lateral penalty it never rewards avoiding the goal line.
                                    # It adds a gradient the binary zone bonus lacks: shots drifting toward a zone edge
                                    # or just outside it are pulled back. Off (0.0) for the shipped checkpoints unless
                                    # a checkpoint's stored env_cfg says otherwise.
    foot_contact_bonus: float = 2.0 # once per episode, at the first foot<->ball contact
    approach: float = 0.3           # * exp(-|right foot - ball| / 0.05) before the first contact
    tilt: float = 2.0               # * -(g_x^2 + g_y^2) of projected gravity
    height: float = 2.0             # * -|base z - 0.16|
    heading: float = 0.5            # * -(yaw - yaw_to_target)^2: face the commanded zone (yaw_to_target = atan2 of base->target point)
    action_rate: float = 0.1        # * -mean((a_t - a_{t-1})^2)
    torque: float = 0.002           # * -mean(actuator_force^2)
    joint_vel: float = 1e-4         # * -mean(leg qvel^2)
    alive: float = 0.1              # per step
    termination: float = -5.0       # once, when the robot falls


@dataclass
class TerminationConfig:
    min_base_height: float = 0.09       # m
    max_tilt_gravity_z: float = -0.5    # projected gravity z > -0.5 means tilt > 60 degrees


@dataclass
class EnvConfig:
    control: ControlConfig = field(default_factory=ControlConfig)
    reference: ReferenceConfig = field(default_factory=ReferenceConfig)
    obs: ObsConfig = field(default_factory=ObsConfig)
    randomization: RandomizationConfig = field(default_factory=RandomizationConfig)
    goal: GoalConfig = field(default_factory=GoalConfig)
    celebration: CelebrationConfig = field(default_factory=CelebrationConfig)
    approach: ApproachConfig = field(default_factory=ApproachConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    termination: TerminationConfig = field(default_factory=TerminationConfig)
    ball_radius: float = 0.04   # m (matches scene_kick.xml; ball mass 0.05 kg, condim 6 with rolling friction 0.0005)

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class PPOConfig:
    total_steps: int = 10_000_000
    num_envs: int = 64
    num_workers: int = 8
    rollout_steps: int = 64
    epochs: int = 5
    minibatch_size: int = 1024
    lr: float = 3e-4
    lr_final_frac: float = 0.1     # linear decay to lr * frac
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    entropy_coef: float = 0.0
    value_coef: float = 0.5
    max_grad_norm: float = 1.0
    hidden: List[int] = field(default_factory=lambda: [256, 256])
    init_log_std: float = -1.0
    min_log_std: float = -2.0      # exploration floor (std >= 0.135); an unclamped run collapsed to -2.95 and stalled
    checkpoint_every_iters: int = 50
    seed: int = 0

    def to_dict(self) -> Dict:
        return asdict(self)


# Scoring (see evaluate.py). A shot is "on target" when a foot touched the ball, the ball crossed the goal line
# x = goal_x inside the commanded zone (|y - zone_y| <= zone_half_width) below the crossbar, and the robot did not
# fall during the episode. "goal" = crossed the line anywhere inside the goal mouth. The shootout score is the number
# of on-target shots out of N; targets cycle left/centre/right with the episode seed.
KICK_MIN_TRAVEL_M = 0.15   # legacy "kick" metric: contact and >= 0.15 m forward travel without falling

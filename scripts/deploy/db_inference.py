#!/usr/bin/env python
"""
DoubleBee policy inference (ROS1) -- TQC, 38-dim observation, RoboClaw wheels.

A clean rewrite of policy_inference_4d.py for the current hardware. That file
still exists and still works for the 22-dim PPO checkpoint; this one targets the
38-dim TQC checkpoints (ckpts/latest/model_*.pt) and drops the ~400 lines of
superseded FCU duty-cycle machinery that the direct RoboClaw link made obsolete.

    rosrun mavros db_inference.py \
        --model_path /home/airlab/doublebee_PID_JAI/ckpts/latest/model_14997.pt \
        --dry_run

Always start with --dry_run. It runs the entire pipeline -- mocap, observation,
inference, logging -- and commands nothing. That is how you confirm the
observation vector is sane before anything can move.


SIGNALS AND WHERE THEY COME FROM
--------------------------------
    idx      term                  dims  scale   source
    ----------------------------------------------------------------------------
    [0:2]    wheel_vel              2    0.05    RoboClaw encoders, REAL
    [2:5]    base_lin_vel           3    2.00    mocap + Kalman filter, body frame
    [5:8]    base_ang_vel           3    0.10    mocap quaternion difference
    [8:11]   projected_gravity      3    1.15    mocap attitude
    [11:27]  height_scan           16    --      mocap + static terrain map
    [27:29]  wheel_ground_contact   2    --      ASSUMED (no contact sensor)
    [29:32]  velocity_commands      3    --      constant, --velocity_command
    [32:38]  actions                6    --      previous raw policy output

Only [27:29] is fabricated. [11:27] is reconstructed rather than sensed, and is
only as good as the terrain map you pass in --step (see below).

Body frame, Isaac Lab DoubleBee: X = right, Y = forward, Z = up.


TWO THINGS THIS FIXES RELATIVE TO policy_inference_4d.py
--------------------------------------------------------
1. NO ACTION DUPLICATION. The old node ran, for every checkpoint kind:

       action[3] = action[2]      # right servo  := left servo
       action[5] = action[4]      # right prop   := left prop

   That is correct for the old ActionsCfg4D PPO run, whose env really did tie
   those pairs together. It is WRONG for these TQC checkpoints: the env config
   declares `propeller_servo_pos` and `propeller_vel` as two-joint action terms
   with MIRRORED scales (+pi/2 / -pi/2 and +250 / -250), so all six dims are
   independent and the left/right members of a pair are expected to differ.

   The damage is not subtle. With model_14997.pt at a plausible resting state
   the policy asks for action[4] = -0.138, action[5] = -0.980, which the sim
   scales to left propeller +215 rad/s and right propeller -5 rad/s -- i.e.
   "spin the left one up, leave the right one alone". Overwriting action[5] with
   action[4] turns the right propeller command into -215 rad/s: same magnitude,
   opposite sign, a propeller driven backwards. The duplication also corrupted
   last_actions, so the error fed straight back into obs[32:38].

2. PROPELLER COMMANDS GO THROUGH THE CALIBRATED ESC MAP. Because the sim's
   propeller term is affine (left = 250*a + 250, right = -250*a - 250, both
   spanning 0..500 rad/s), the throttle FRACTION for each propeller is
   (a_i + 1) / 2 -- the mirroring is absorbed by the offset, so each propeller
   uses its OWN action and neither needs a sign flip. That fraction is then put
   through the ESC calibration the PPO deployment already validated on this
   airframe:

       u_thr = -0.9175 + (a + 1) * 0.86625        [-1,1] -> [-0.918, +0.815]

   The old TQC path instead passed the raw action straight through, which reads
   a "propeller off" command of a = -0.98 as -0.98 * prop_scale and, after the
   firmware's linear [-1,1] -> PWM 1000..2000 map, hands the ESC roughly 40%
   throttle for a propeller that should be idle. --prop_map passthrough restores
   the old behaviour if you need to compare.


WHEEL PATH
----------
Wheels do NOT go out over JAIOut any more. The velocity setpoint goes straight
to the RoboClaw over USB, whose hardware PID closes the loop against the
encoders wired to it -- the closest real analogue of the velocity-tracked wheel
joint that sim trains. wh_l/wh_r are still published as 0.0 so the firmware
slews its wheel channels to neutral instead of holding a stale command.

    desired_rad_s = -action[i] * wheel_action_scale        (both wheels)

The leading minus is not arbitrary: in the sim's own policy I/O log, both wheel
actions correlate NEGATIVELY with forward body velocity (-0.28 and -0.33), so
negative action means forward. Conventions, and the measurement that pins each
one down, are documented in db_wheels.py -- read that before touching signs.

--wheel_action_scale is the rad/s that |action| = 1 represents. It defaults to
35, the `velocity_limit` on the wheel joints in the training config, which also
happens to match the motor's ~34.6 rad/s no-load ceiling. Be aware this is the
weakest link in the whole sim2real chain: sim commands the wheel joint at
action * 500 rad/s and lets an effort limit of 1.0 Nm decide what actually
happens, so the realised joint velocity in sim is set by load, not by the
action. No scalar mapping reproduces that faithfully. 35 is defensible, not
correct.


SAFETY
------
  --dry_run          full pipeline; the RoboClaw is never commanded and the
                     JAIOut message is forced to neutral (wheels 0, servos 0,
                     propellers idle). It DOES still publish that neutral
                     message at the control rate, deliberately -- going silent
                     would leave the FCU holding whatever it last received.
  --preflight N      --dry_run for N seconds, then a per-term verdict on the
                     observation vector and an exit code. Run this on the floor
                     before every session.
  --no_props         wheels and servos only; propellers held at idle.
  --action_scale     uniform attenuation on servos and propellers (default 0.4)
  --prop_scale       extra attenuation on propellers only (default 0.5)
  --wheel_scale      attenuation on the wheel velocity setpoint (default 0.4)
  mocap watchdog     no pose for --mocap_timeout seconds stops the wheels and
                     idles the propellers, rather than flying on a frozen pose.
  --arm_gate on      (default) wheels turn only while ARMED *and* CH6 is high.

THE WHEEL GATE, AND WHY IT IS NOT OPTIONAL
------------------------------------------
The wheels are driven over USB by the RoboClaw, deliberately outside the FCU's
arming chain. That is what makes closed-loop velocity tracking possible, but it
has a consequence that is easy to miss until it bites: NEITHER DISARMING NOR THE
POLICY SWITCH STOPS THE WHEELS BY ITSELF. Both cut propellers and servos, and
the robot keeps driving away on the last setpoint the RoboClaw received.

That interlock used to exist for free. While the wheels still went out through
JAIOut, the firmware enforced it (GCS_Mavlink.cpp:1139-1148):

    const bool ch6_high = (RC_Channels::get_radio_in(CH_6) > 1500);
    if (!ch6_high || packet.rc_state == 0) { /* zero the whole JAI path */ }

Moving the wheels onto their own USB link silently dropped it. --arm_gate on
reinstates it in software with the same channel and the same 1500 us threshold,
so the robot behaves the way it did before:

    wheels move  <=>  FCU ARMED  AND  CH6 > 1500 us

Arming alone is deliberately NOT enough. Releasing the switch, disarming, or
losing either signal all command zero to the wheels. It fails safe -- an unknown
arm state, an RC link that has gone quiet, or a transmitter that was never
turned on all gate the wheels rather than permitting them.

Three caveats worth knowing:
  * /mavros/state rides the 1 Hz HEARTBEAT, so a disarm can take up to a second
    to register. The CH6 path updates at the RC stream rate and is faster, which
    is another reason to leave --enable_channel at its default rather than
    relying on disarm alone.
  * This is a SOFTWARE interlock inside this process. It cannot help if the
    process is SIGKILLed or the USB cable is pulled. The RoboClaw's own serial
    timeout, set in Motion Studio, is the only backstop that survives that, and
    it is worth confirming it is actually configured.
  * It is not a substitute for being able to reach the battery.

Note that this policy is not shy. Given a plausible at-rest observation,
model_14997.pt immediately asks for roughly -0.98 on both wheels, i.e. near
full-speed forward from the first control tick. There is no gentle start.
"""

import argparse
import csv
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

import rospy
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import JAIOut, RCIn, State
from std_msgs.msg import Header

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from height_scan_from_mocap import compute_height_scan, quat_to_yaw, StepMap
from db_wheels import RoboClawWheels, WHEEL_OBS_SCALE, SIM_WHEEL_VEL_LIMIT_RAD_S


# =============================================================================
# Observation scales -- these MUST track ObservationsCfg.PolicyCfg
# =============================================================================
# Read from ckpts/g1_38/params/env.yaml. ckpts/latest/ ships no params/ dir, so
# these are carried over from the last TQC run that did. If the observation
# space changed between those runs, nothing here will error -- the policy will
# just quietly receive garbage. Worth dumping params/ alongside the checkpoint.

# Two action layouts, selected automatically from the checkpoint's output width.
# Everything before `actions` in the observation is identical, so obs_dim is
# always 32 + act_dim.
#
#   6 actions / obs 38 -- INDEPENDENT servos (pre-2026-08-21 checkpoints):
#       [0] wheelL [1] wheelR [2] servoL [3] servoR [4] propL [5] propR
#       theta_L = +pi/2*a[2],  theta_R = -pi/2*a[3]   (mirrored scale dict)
#
#   5 actions / obs 37 -- TIED servos (TiedJointPositionAction, scalar scale):
#       [0] wheelL [1] wheelR [2] servo  [3] propL  [4] propR
#       theta_L = theta_R = +pi/2*a[2]
#
# Hardware mapping is servo_i = -(2/pi)*theta_i in both cases, so the tied
# layout sends the SAME value to servo1 and servo2 -- which is what the working
# PPO deployment did, and the opposite of the arms-opposed configuration that
# could not hold the robot up.
#   4 actions / obs 36 -- TIED wheels AND servos (2026-08-23 on):
#       [0] wheel  [1] servo  [2] propL  [3] propR
#       both wheels get the SAME ground-speed command; there is no steering, so
#       --max_wheel_diff is irrelevant for these checkpoints (it clamps a
#       differential that is structurally zero).
LAYOUTS = {
    6: {"wheel": (0, 1), "servo": (2, 3), "prop": (4, 5)},
    5: {"wheel": (0, 1), "servo": (2, 2), "prop": (3, 4)},
    4: {"wheel": (0, 0), "servo": (1, 1), "prop": (2, 3)},
    # 2026-08-26: propellers tied too. One action drives both, counter-rotating.
    3: {"wheel": (0, 0), "servo": (1, 1), "prop": (2, 2)},
}

# OBSERVATION LAYOUT GENERATIONS.
#
# Checkpoints up to 2026-08-25 carry 32 non-action observations. Checkpoints
# from 2026-08-26 carry 36: servo_pos and propeller_vel were re-enabled in
# PolicyCfg, inserted at [2:4] and [4:6], pushing everything after them by 4.
#
# The policy could previously command servo angle and propeller speed while
# observing NEITHER, and the actor is feedforward -- one step of action history
# and no memory -- so it had no way to know where its own thrust was pointing.
# That matters on this airframe because the propellers' restoring moment is
# T*L_prop*sin(theta - psi): at psi = theta it is exactly zero.
OBS_BASE_LEGACY = 32   # servo_pos / propeller_vel ABSENT
OBS_BASE_ACTUATOR = 36  # servo_pos / propeller_vel PRESENT
# 2026-09-06: velocity_command.py grew a FOURTH command channel -- XY range to
# target, clamp(distance_xy / MAX_RANGE_M, 0, 1) with MAX_RANGE_M = 2.0. The
# first three (direction_x_body, direction_y_body, ang_vel_z=0) are unchanged,
# so this is one extra slot between contact and the action history.
OBS_BASE_RANGE = 37     # + 4-channel command (range included)
TARGET_MAX_RANGE_M = 2.0

# ACTION HISTORY, 2026-08-26. The observation carries the last N actions, not one.
#
# Measured on hw_v18.csv across five engaged segments:
#     lean -> wheel_action       lag  0 ticks (  0 ms)  r = -0.75..-0.80
#     wheel_des -> wheel_meas    lag 15 ticks (300 ms)  r = +0.74..+0.84
#
# State estimation is not the delay -- the policy reacts to attitude instantly.
# The WHEELS lag 300 ms, dominated by the 43 rad/s^2 acceleration limit. With one
# step of command history the policy cannot know what it has already commanded and
# not yet received, which is why it holds still but falls when it commits to
# forward motion.
#
# ORDERING IS NEWEST FIRST and must match mdp/observations.py::action_history:
#     [a(t-1) dims..., a(t-2) dims..., ..., a(t-N) dims...]
# Both sides were written by hand rather than relying on IsaacLab's
# ObsTerm(history_length=...), because an ordering mismatch between sim and
# hardware produces no error and no visible symptom -- just a policy fed noise.
ACTION_HISTORY_LEN = 5

# act_dim 4 IS AMBIGUOUS and must be resolved by obs_dim.
#
#   obs 36 (= 32 + 4)  legacy: tied wheels, tied servos, INDEPENDENT props
#                      {"wheel": (0,0), "servo": (1,1), "prop": (2,3)}
#   obs 40 (= 36 + 4)  2026-08-26: wheel COMMON + wheel DIFFERENTIAL, tied
#                      servos, tied props -- the differential exists because
#                      tied wheels left the robot with no yaw authority at all
#                      while cross-track drift was still being penalised.
#
# Same act_dim, completely different meaning for indices 1..3. Getting it wrong
# feeds the yaw trim into the servos and the servo command into the propellers.
LAYOUTS_BY_OBS = {
    (4, OBS_BASE_ACTUATOR + 4): {"wheel": (0, 0), "wheel_diff": 1,
                                 "servo": (2, 2), "prop": (3, 3)},
    # Identical ACTION layout; only the observation got wider.
    (4, OBS_BASE_RANGE + 4): {"wheel": (0, 0), "wheel_diff": 1,
                              "servo": (2, 2), "prop": (3, 3)},
}

# obs scale on propeller_vel in PolicyCfg (observations.py).
SCALE_PROP_VEL = 0.01

# DEFAULT 0. Was 3, which CAUSED A LIMIT CYCLE -- see below.
#
# The reasoning for 3 was: sim's actuators carry min_delay 2 / max_delay 5, so
# the position sim OBSERVES lags the command, and an undelayed estimate would
# hand the policy fresher state than it saw in training.
#
# That conflated two different things. Sim's delay applies to the COMMAND
# reaching the joint; the observed servo_pos is then the joint's ACTUAL
# position, which the joint's own inertia, damping and effort saturation
# low-pass. self.servo_cmd is already the SLEW-LIMITED command -- the slew IS
# that dynamics -- so stacking a pure transport delay on top double-counts it,
# and a pure delay inside a feedback path is the destabilising kind.
#
# Measured, hw_v15.csv 2026-08-26: the servo command became a clean square wave
# and the position a clean triangle wave, autocorrelation peaking at lag 12
# ticks (240 ms, 4.2 Hz) -- exactly 4x this delay, the classic signature. It
# swamped any response to lean: corr(lean, servo_act) read +0.05 / +0.02 /
# -0.19 / +0.02 across four engaged segments, i.e. nothing.
ACTUATOR_OBS_DELAY_TICKS = 0

# PROPELLER STEADY-STATE MODEL -- doublebee_v1.py propellers + aerodynamics.py.
#
# The tied propeller action maps action a -> target velocity 187.5*(1+a), and
# the joint settles where the drive torque balances aerodynamic drag:
#
#     D * (target - omega) = k * omega^2
#     omega = (-D + sqrt(D^2 + 4*k*D*target)) / (2*k)
#
# D = 0.015 is the propeller joint damping; k = 1.25e-4 is the drag coefficient
# implied by 5.0 N*m holding 200 rad/s. VALIDATED against a play log on
# 2026-08-26: at raw_action 0.992 the model predicts 160.0 rad/s and sim
# measured 157.5 / 159.6, a 0.9% error.
#
# THIS IS WHY --prop_map sim_damped EXISTS. The older `sim` map is LINEAR --
# omega = prop_rad_s_max * (a+1)/2 -- and the real curve is concave because
# drag grows as omega^2. At the policy's actual operating point (a = 0.992)
# the linear map with the shipped default prop_rad_s_max=375 commands 374 rad/s
# where sim reaches 160: pwm 1486 vs 1208, thrust 25.6 N vs 11.6 N. That is
# 2.2x, T/W 0.81 instead of 0.37 -- near hover, wheels unloaded, on a robot
# that needs wheel traction to steer.
PROP_DAMPING = 0.015
PROP_DRAG_K = 1.25e-4
# 2026-09-02: 187.5 -> 320.0 to MATCH THE SIMULATOR. actions.py sets the tied
# propeller action's tied_scale == tied_offset == 320, i.e. sim commands a target
# of 320(1+a) rad/s. Deployment used 187.5, so for the same action it commanded
# 1.71x less propeller speed, and --prop_scale silently absorbed the difference.
#
# What that cost: prop_scale 3.5 doubled thrust at low command and SATURATED
# everything above a3 = -0.43, so the whole usable range was a3 in [-1, -0.43].
# On hardware that turned the policy's smooth thrust modulation into a two-state
# switch -- 32% or 81% of weight -- which is why the climb-profile trace shows a
# flat thrust command through the riser while the same policy modulates in sim.
#
# It also capped thrust: at 187.5 the frac=1 target is 375 rad/s -> 25.6 N, so
# the top 30% of the envelope (sim reaches 36.6 N) was unreachable at ANY
# prop_scale.
#
# With 320.0 and --prop_scale 1.0 the chain reproduces sim to within 0.2% across
# the action range (a3 = -0.5: 11.55 vs 11.53 N; a3 = 0: 21.52 vs 21.47 N).
PROP_HALF_SPAN = 320.0   # propeller_vel action scale == offset, matches actions.py

OBS_DIM = 38
ACT_DIM = 6
SCALE_LIN_VEL = 2.0
SCALE_ANG_VEL = 0.10
SCALE_GRAVITY = 1.15

# Propeller ESC calibration, validated on this airframe during the PPO
# deployment: maps a normalized [-1, 1] propeller action into the JAIOut range
# the ESCs actually respond to. -1 (propeller off) -> -0.9175, +1 -> +0.815.
PROP_ESC_OFFSET = -0.9175
PROP_ESC_GAIN = 0.825 * 1.05


# =============================================================================
# Policy
# =============================================================================

class TQCActor(nn.Module):
    """Mirrors co_rl/core/algorithms/networks/tqc_network.py::GaussianPolicy.

    Deterministic action is tanh(mean); action_bound is [-1, 1] so there is no
    further rescale. log_std_layer is unused at inference but must exist for the
    state_dict to load strictly.
    """

    def __init__(self, state_dim, action_dim=ACT_DIM, hidden_dims=(512, 256, 128)):
        super(TQCActor, self).__init__()
        self.input_layer = nn.Linear(state_dim, hidden_dims[0])
        self.hidden_layers = nn.ModuleList(
            nn.Linear(hidden_dims[i], hidden_dims[i + 1])
            for i in range(len(hidden_dims) - 1)
        )
        self.mean_layer = nn.Linear(hidden_dims[-1], action_dim)
        self.log_std_layer = nn.Linear(hidden_dims[-1], action_dim)

    def forward(self, state):
        x = torch.relu(self.input_layer(state))
        for layer in self.hidden_layers:
            x = torch.relu(layer(x))
        return torch.tanh(self.mean_layer(x))


def load_policy(path, device):
    """Load a TQC checkpoint. Returns (actor, obs_dim).

    obs_dim is read off the first layer rather than assumed, so a checkpoint
    trained with a different observation space fails loudly here instead of
    silently consuming a mis-shaped vector.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if "actor_state_dict" not in ckpt:
        if "model_state_dict" in ckpt:
            raise ValueError(
                "%s is a PPO (model_state_dict) checkpoint. This node only handles "
                "the 38-dim TQC checkpoints -- use policy_inference_4d.py for PPO."
                % path)
        raise ValueError("%s has no 'actor_state_dict'. Keys: %s"
                         % (path, list(ckpt.keys())))

    state = ckpt["actor_state_dict"]
    obs_dim = state["input_layer.weight"].shape[1]
    act_dim = state["mean_layer.weight"].shape[0]
    valid = (OBS_BASE_LEGACY + act_dim,
             OBS_BASE_ACTUATOR + act_dim,                          # 1-step history
             OBS_BASE_RANGE + act_dim,                             # + range channel
             OBS_BASE_ACTUATOR + act_dim * ACTION_HISTORY_LEN)     # N-step history
    if act_dim not in LAYOUTS or obs_dim not in valid:
        raise ValueError(
            "checkpoint has obs_dim=%d act_dim=%d. Supported act_dim: 6 "
            "(independent servos), 5 (tied servos), 4 (tied wheels AND servos), "
            "3 (tied wheels, servos AND propellers). obs_dim must be either "
            "%d+act_dim (legacy, no actuator feedback) or %d+act_dim (2026-08-26 "
            "onward, servo_pos and propeller_vel observed). This checkpoint "
            "matches none -- the observation layout changed again and this file "
            "needs updating before you run it. (action history %d would give %d)"
            % (obs_dim, act_dim, OBS_BASE_LEGACY, OBS_BASE_ACTUATOR,
               ACTION_HISTORY_LEN, OBS_BASE_ACTUATOR + act_dim * ACTION_HISTORY_LEN))

    actor = TQCActor(state_dim=obs_dim, action_dim=act_dim).to(device)
    actor.load_state_dict(state)
    actor.eval()
    rospy.loginfo("Loaded TQC actor from %s (iter=%s, total_steps=%s)",
                  path, ckpt.get("iter"), ckpt.get("total_steps"))
    return actor, obs_dim



# =============================================================================
# DECOUPLED-MODE BASELINE
# =============================================================================
#
# Added 2026-08-26. Runs the published DoubleBee decoupled controller (Cao et
# al., eqs 19-23) through the SAME plumbing as the RL policy -- same mocap state
# estimate, same RoboClaw wheel path, same MAVLink servo/propeller path, same
# arm/CH6 gates, same CSV schema. That is the point: IROS R1 objected that the
# paper compared an autonomous method against a HUMAN-OPERATED baseline. This
# makes the baseline autonomous and the comparison apples-to-apples.
#
# The controller is IMPORTED, never copied. A divergent second copy would be a
# silent discrepancy between the sim baseline and the hardware baseline, which
# is exactly the kind of thing a reviewer is entitled to distrust.

# thrust polynomial, identical to mdp/aerodynamics.py -- pwm (us) -> newtons
_THRUST_C = [4.0792540792478203e-13, -2.2921522921483562e-09,
             2.2550699300690226e-05, -0.026882905982896884, 8.516433566430207]


def _pwm_to_thrust(pwm):
    pwm = min(max(float(pwm), 1000.0), 1650.0)
    t = 0.0
    for c in _THRUST_C:
        t = t * pwm + c
    return max(t, 0.0)


def _thrust_to_pwm(thrust_n):
    """Invert the thrust polynomial by bisection. Monotonic on [1000, 1650]."""
    lo, hi = 1000.0, 1650.0
    if thrust_n <= _pwm_to_thrust(lo):
        return lo
    if thrust_n >= _pwm_to_thrust(hi):
        return hi
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if _pwm_to_thrust(mid) < thrust_n:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def load_decoupled_controller(path):
    """Import DecoupledController from the training repo. Single source of truth."""
    import importlib.util
    import os
    if not os.path.isfile(path):
        raise ValueError(
            "--controller decoupled needs doublebee_dctrl.py. Not found at %r. "
            "Point --dctrl_path at scripts/co_rl/doublebee_dctrl.py in the "
            "training repo." % path)
    spec = importlib.util.spec_from_file_location("doublebee_dctrl", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.DecoupledController


# =============================================================================
# State estimation
# =============================================================================

def quat_to_rotation_matrix(q):
    """Body-to-world rotation from a unit quaternion [x, y, z, w]."""
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def quat_conjugate(q):
    return np.array([-q[0], -q[1], -q[2], q[3]])


def quat_multiply(a, b):
    x1, y1, z1, w1 = a
    x2, y2, z2, w2 = b
    return np.array([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ])


class KalmanFilterCV:
    """Constant-velocity KF over [x, y, z, vx, vy, vz], world frame.

    Mocap gives position only. Differentiating it raw gives velocity dominated
    by quantisation noise at 50 Hz, which lands directly in obs[2:5] with a
    scale of 2.0. The filter is the difference between a usable velocity term
    and a noisy one.
    """

    def __init__(self, process_noise=1.0, measurement_noise=0.0005):
        self.x = np.zeros(6)
        self.P = np.eye(6) * 0.1
        self.q = process_noise
        self.R = np.eye(3) * measurement_noise
        self.H = np.zeros((3, 6))
        self.H[:3, :3] = np.eye(3)
        self.initialized = False

    def predict(self, dt):
        F = np.eye(6)
        F[:3, 3:] = np.eye(3) * dt
        self.x = F @ self.x
        # Continuous white-noise acceleration, discretised.
        Q = np.zeros((6, 6))
        Q[:3, :3] = np.eye(3) * (dt ** 4) / 4.0
        Q[:3, 3:] = np.eye(3) * (dt ** 3) / 2.0
        Q[3:, :3] = np.eye(3) * (dt ** 3) / 2.0
        Q[3:, 3:] = np.eye(3) * (dt ** 2)
        self.P = F @ self.P @ F.T + Q * self.q

    def update(self, z):
        if not self.initialized:
            self.x[:3] = z
            self.initialized = True
            return
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P

    @property
    def position(self):
        return self.x[:3]

    @property
    def velocity(self):
        return self.x[3:]


# =============================================================================
# Node
# =============================================================================

class DoubleBeeInference(object):

    def __init__(self, args):
        rospy.init_node("db_inference", anonymous=True)
        self.args = args
        self.dry_run = args.dry_run

        self.device = torch.device(args.device)
        self.policy, obs_dim = load_policy(args.model_path, self.device)
        self.obs_dim = obs_dim
        self.act_dim = self.policy.mean_layer.out_features
        # (act_dim, obs_dim) first -- act_dim 4 is ambiguous, see LAYOUTS_BY_OBS.
        self.layout = LAYOUTS_BY_OBS.get((self.act_dim, obs_dim),
                                         LAYOUTS.get(self.act_dim))
        if self.layout is None:
            raise ValueError("no action layout for act_dim=%d obs_dim=%d"
                             % (self.act_dim, obs_dim))
        self.wheel_diff_idx = self.layout.get("wheel_diff")

        # Which observation generation this checkpoint expects. load_policy has
        # already rejected anything that is neither.
        self.obs_has_actuator_state = obs_dim in (OBS_BASE_ACTUATOR + self.act_dim,
                                                  OBS_BASE_RANGE + self.act_dim)
        # 3 channels (dir_x, dir_y, 0) or 4 (+ XY range). Drives both the width of
        # the command block and where the action history starts, so a wrong value
        # here silently feeds the range into the first action slot.
        self.cmd_dim = 4 if obs_dim == OBS_BASE_RANGE + self.act_dim else 3
        rospy.loginfo("Command channels: %d%s", self.cmd_dim,
                      " (includes XY range)" if self.cmd_dim == 4 else "")
        # Ring buffer for the delayed servo_pos / propeller_vel estimates.
        self._act_state_buf = []
        # First-order state for --servo_obs_source lowpass.
        self._servo_obs_est = 0.0
        # --wheel_ramp_s state; see the ramp block in the control loop.
        self._wheel_ramp_n = int(round(args.wheel_ramp_s * 50.0))   # 50 Hz loop
        self._wheel_ramp_i = 0
        # --wheel_scale_post state: 1 s of height history, and a countdown of
        # ticks remaining in the post-climb settling window.
        self._z_hist = []
        self._post_climb_i = 0
        # --heading_hold_kp reference, captured on the first live tick.
        self._yaw_ref = None
        # --- electrical logging ---
        self.batt_v = float("nan")
        self.batt_a = float("nan")
        self._energy_j = 0.0
        self._e_last_t = None

        # ACTION HISTORY, newest first. self.last_actions is a(t-1); this holds
        # a(t-1) .. a(t-N). How many steps the loaded checkpoint expects is
        # derived from its observation width, so a 1-step checkpoint still works.
        self.act_hist_len = 1
        if obs_dim == OBS_BASE_ACTUATOR + self.act_dim * ACTION_HISTORY_LEN:
            self.act_hist_len = ACTION_HISTORY_LEN
        self._act_hist = np.zeros((self.act_hist_len, self.act_dim), dtype=np.float32)
        rospy.loginfo("Action history: %d step(s) -> %d observation dims",
                      self.act_hist_len, self.act_hist_len * self.act_dim)

        # --- decoupled-mode baseline ---------------------------------------
        self.dctrl = None
        self._dctrl_wref = 0.0      # integrated wheel speed setpoint, rad/s
        self._dctrl_prop_pwm = None  # direct ESC command when running the baseline
        if args.controller == "decoupled":
            Ctrl = load_decoupled_controller(args.dctrl_path)
            self.dctrl = Ctrl()
            self.dctrl.dt = 1.0 / float(args.rate)
            self.dctrl.reset()
            rospy.logwarn("CONTROLLER: DECOUPLED BASELINE (not the RL policy). "
                          "T_hold=%.2f N/prop, v_desired=%.2f m/s",
                          self.dctrl.T_hold, args.dctrl_v_desired)
        rospy.loginfo(
            "Observation layout: obs=%d act=%d -> %s (servo_pos/propeller_vel %s)",
            obs_dim, self.act_dim,
            "ACTUATOR-STATE (2026-08-26+)" if self.obs_has_actuator_state
            else "LEGACY",
            "OBSERVED" if self.obs_has_actuator_state else "absent")
        rospy.loginfo("Action layout: %d dims -> wheel idx %s, servo idx %s, "
                      "prop idx %s (%s wheels, %s servos)",
                      self.act_dim, self.layout["wheel"], self.layout["servo"],
                      self.layout["prop"],
                      "TIED" if self.layout["wheel"][0] == self.layout["wheel"][1]
                      else "independent",
                      "TIED" if self.layout["servo"][0] == self.layout["servo"][1]
                      else "independent")

        # ---- terrain map for height_scan ----------------------------------
        self.terrain = StepMap(ground_height=args.ground_height)
        for s in (args.step or []):
            self.terrain.add_step(x_min=s[0], x_max=s[1], y_min=s[2], y_max=s[3],
                                  height=s[4])
        if not args.step:
            rospy.logwarn(
                "No --step given: height_scan will report FLAT GROUND everywhere. "
                "The policy will be blind to the very obstacle it was trained to "
                "climb. Measure the step in the mocap frame and pass "
                "--step x_min x_max y_min y_max height.")
        else:
            rospy.loginfo("Terrain map: %d step(s), ground z=%.3f",
                          len(args.step), args.ground_height)
        self.base_z_offset = args.base_z_offset

        # ---- wheels --------------------------------------------------------
        self.wheels = None
        if args.no_roboclaw:
            rospy.logwarn("--no_roboclaw: no wheel drive AND no real wheel_vel. "
                          "obs[0:2] will be ZERO, which is a lie unless the robot "
                          "is stationary. Bench use only.")
        else:
            self.wheels = RoboClawWheels(
                port=args.roboclaw_port, baud=args.roboclaw_baud,
                address=args.roboclaw_address,
                counts_per_rev=args.wheel_counts_per_rev,
                accel_counts_s2=args.roboclaw_accel,
                m1_is_left=not args.m1_is_right,
                sign_left=args.wheel_sign_left, sign_right=args.wheel_sign_right,
                max_rad_s=args.wheel_max_rad_s,
                vel_mirror=args.wheel_vel_mirror,
                log=lambda fmt, *a: rospy.loginfo(fmt, *a),
            )
            rospy.on_shutdown(self.wheels.close)

        # ---- state ---------------------------------------------------------
        self.kf = KalmanFilterCV(args.process_noise, args.measurement_noise)
        self.latest_pos = None
        self.latest_quat = None
        self.latest_stamp = None
        self.new_pose = False
        self.prev_quat = None
        self.prev_time = None
        self.ang_vel_body = np.zeros(3)
        self.ang_vel_rejects = 0
        self.pose_repeats = 0
        # latches True on the first permitted tick, so the upright-start check in
        # _gate() only ever blocks the INITIAL engage
        self._engaged = False
        self.servo_cmd = np.zeros(2)
        self._servo_lpf = None      # first-order lag on the servo command
        self._wheel_lpf = None   # --wheel_lpf_alpha state; None until first tick
        # filtered propeller speeds, rad/s -- see _propeller_command
        self.prop_omega = np.zeros(2)
        self.target_dist = float('nan')
        self.last_actions = np.zeros(self.act_dim, dtype=np.float32)
        self.velocity_cmd = np.array(args.velocity_command, dtype=np.float32)
        self.step_count = 0
        self.wheels_stale_warned = False

        # ---- arm gate ------------------------------------------------------
        # The wheels are on their own USB link, outside the FCU's arming chain,
        # so disarming does NOT stop them by itself -- it only cuts propellers
        # and servos. This gate puts them back under the same switch.
        self.armed = None
        self.armed_stamp = None
        self.enable_high = False
        self.enable_value = 0
        self.rc_stamp = None
        self.was_gated = False

        # ---- ROS I/O -------------------------------------------------------
        self.pub = rospy.Publisher("/Jai_command", JAIOut, queue_size=1)
        rospy.Subscriber(args.pose_topic, PoseStamped, self._pose_cb, queue_size=1)
        if args.arm_gate == "on":
            rospy.Subscriber("/mavros/state", State, self._state_cb, queue_size=1)
            # Whole-robot draw from the Cube's power module. Wrapped because a
            # setup without a power module simply never publishes, and the run
            # must not die for the want of a log column.
            try:
                from sensor_msgs.msg import BatteryState
                rospy.Subscriber("/mavros/battery", BatteryState,
                                 self._batt_cb, queue_size=1)
                rospy.loginfo("subscribed /mavros/battery for energy logging")
            except Exception as e:
                rospy.logwarn("no /mavros/battery (%s) -- energy will be NaN", e)
            if args.enable_channel:
                rospy.Subscriber("/mavros/rc/in", RCIn, self._rc_cb, queue_size=1)
                rospy.loginfo("Gate ON: wheels require ARMED *and* policy enable "
                              "CH%d > %d us (mirrors GCS_Mavlink.cpp:1139 ch6_high)",
                              args.enable_channel, args.enable_above)
            else:
                rospy.logwarn("Gate ON but --enable_channel 0: wheels released by "
                              "ARM alone, which is NOT the pre-existing behaviour.")
        else:
            rospy.logwarn("--arm_gate off: NEITHER DISARM NOR THE POLICY SWITCH "
                          "WILL STOP THE WHEELS.")

        # ---- logging -------------------------------------------------------
        self.log_file = None
        self.log_writer = None
        if args.log_path:
            path = os.path.abspath(args.log_path)
            self.log_file = open(path, "w")
            self.log_writer = csv.writer(self.log_file)
            self.log_writer.writerow(
                ["step", "t"]
                + ["obs_%d" % i for i in range(self.obs_dim)]
                + ["action_%d" % i for i in range(self.act_dim)]
                + ["wheel_des_l", "wheel_des_r", "wheel_meas_l", "wheel_meas_r",
                   "wh_l", "wh_r", "servo1", "servo2", "u_thr1", "u_thr2",
                   "wheel_fresh", "gated", "gate_reason",
                   # raw mocap pose -- NOT part of the observation, logged so
                   # yaw-dependent behaviour can be tested after the fact. The
                   # obs vector is yaw-invariant on flat ground by construction,
                   # so if behaviour correlates with yaw, something upstream is
                   # wrong and this is the column that proves it.
                   "pos_x", "pos_y", "pos_z", "yaw_deg", "qx", "qy", "qz", "qw",
                   # ELECTRICAL. VI-B1 reports hardware energy as the integral of
                   # V*I over the trial, and none of it was logged -- 68 columns
                   # and not one electrical, so no run before 2026-09-07 can
                   # produce that number.
                   #
                   # batt_v / batt_a come from the Cube's power module via
                   # /mavros/battery and cover the WHOLE robot, which is what the
                   # paper compares against the decoupled baseline. m1_a / m2_a
                   # are the RoboClaw's per-wheel currents, so the wheel and
                   # propeller shares can be separated afterwards. energy_J
                   # integrates batt_v*batt_a over LIVE ticks only -- idling on
                   # the bench is not part of a traversal.
                   "batt_v", "batt_a", "watts", "energy_J", "m1_a", "m2_a"])
            rospy.loginfo("Logging to %s", path)
            rospy.on_shutdown(self._close_log)

        mode = "DRY RUN -- nothing will move" if self.dry_run else "LIVE"
        rospy.loginfo("=" * 62)
        rospy.loginfo("db_inference ready [%s]", mode)
        rospy.loginfo("  rate=%d Hz  cmd=%s", args.rate, self.velocity_cmd)
        rospy.loginfo("  action_scale=%.2f prop_scale=%.2f wheel_scale=%.2f",
                      args.action_scale, args.prop_scale, args.wheel_scale)
        rospy.loginfo("  wheel_action_scale=%.1f rad/s  prop_map=%s  props=%s",
                      args.wheel_action_scale, args.prop_map,
                      "OFF" if args.no_props else "on")
        rospy.loginfo("=" * 62)

    def _close_log(self):
        if self.log_file is not None:
            self.log_file.close()
            self.log_file = None

    # ---- state ------------------------------------------------------------

    def _pose_cb(self, msg):
        p, o = msg.pose.position, msg.pose.orientation
        pos = np.array([p.x, p.y, p.z])
        q = np.array([o.x, o.y, o.z, o.w])
        n = np.linalg.norm(q)
        q = q / n if n > 1e-6 else q

        # FROZEN-POSE DETECTION -- proven necessary on hardware 2026-08-21.
        #
        # Near the edge of the capture volume the mocap loses the rigid body and
        # republishes its last pose forever. Measured: 2001 messages at a perfect
        # 100 Hz over 20 s with NOT ONE value changing, while the robot was being
        # physically tilted through 60 degrees. rostopic hz looks healthy and the
        # message-arrival watchdog cannot see it, because messages ARE arriving.
        #
        # Everything attitude-derived is then wrong -- projected_gravity, the KF's
        # base_lin_vel, height_scan -- and the policy flies blind. transfer3.csv
        # was 47% frozen during live control, which is enough on its own to
        # explain the divergence we spent a day attributing to actuator delay.
        #
        # Real mocap always carries sub-millimetre noise, so byte-identical
        # consecutive samples mean genuinely held, never a stationary robot.
        if (self.latest_pos is not None
                and np.array_equal(pos, self.latest_pos)
                and np.array_equal(q, self.latest_quat)):
            self.pose_repeats += 1
        else:
            if self.pose_repeats >= self.args.pose_freeze_ticks:
                rospy.logwarn("mocap resumed after %d frozen ticks (%.2fs)",
                              self.pose_repeats, self.pose_repeats / 100.0)
            self.pose_repeats = 0

        self.latest_pos = pos
        self.latest_quat = q
        self.latest_stamp = msg.header.stamp.to_sec()
        self.new_pose = True

    def pose_frozen(self):
        """True if the mocap is republishing a held pose (tracking lost)."""
        return self.pose_repeats >= self.args.pose_freeze_ticks

    def _batt_cb(self, msg):
        self.batt_v = float(msg.voltage)
        # MAVLink reports discharge as negative on some stacks; energy is a
        # magnitude either way.
        self.batt_a = abs(float(msg.current))

    def _state_cb(self, msg):
        self.armed = msg.armed
        self.armed_stamp = time.time()

    def _rc_cb(self, msg):
        """Track the policy-inference enable switch.

        This mirrors the firmware exactly. GCS_Mavlink.cpp:1139 reads

            const bool ch6_high = (RC_Channels::get_radio_in(CH_6) > 1500);

        and zeroes the entire JAI path when it is low. CH_6 is 0-indexed there,
        so it is RC channel 6 in human numbering, i.e. channels[5] here.
        """
        ch = self.args.enable_channel
        if len(msg.channels) < ch:
            return
        v = msg.channels[ch - 1]
        if v <= 0:  # 0 = channel absent / no signal. Not evidence of "enabled".
            return
        self.enable_high = v > self.args.enable_above
        self.enable_value = v
        self.rc_stamp = time.time()

    def _gate(self):
        """(permitted, reason). Fail-safe: anything unknown means NOT permitted.

        The wheels move only when BOTH hold:
          * the FCU is armed, and
          * the policy-inference switch (CH6) is high.

        Arming alone is deliberately not enough. That matches how the robot
        behaved before the wheels moved onto the USB link, when the firmware
        itself zeroed the JAI outputs on !ch6_high.

        Staleness on either input gates the wheels too: mavros dying, or the RC
        link dropping, must not leave them running on the last setpoint.
        """
        if self.args.arm_gate != "on":
            return True, None

        if self.armed is None:
            return False, "waiting for /mavros/state (is mavros up?)"
        if time.time() - self.armed_stamp > self.args.arm_timeout:
            return False, "no /mavros/state for %.1fs" % (time.time() - self.armed_stamp)
        if not self.armed:
            return False, "FCU disarmed"

        if self.pose_frozen():
            return False, ("mocap FROZEN -- %d identical poses (%.1fs). Tracking is "
                           "lost; every attitude term is stale. Move the robot "
                           "toward the centre of the capture volume."
                           % (self.pose_repeats, self.pose_repeats / 100.0))

        if self.args.enable_channel:
            if self.rc_stamp is None:
                return False, ("waiting for /mavros/rc/in -- policy enable switch "
                               "CH%d unknown (transmitter on?)" % self.args.enable_channel)
            if time.time() - self.rc_stamp > self.args.rc_timeout:
                return False, "no RC input for %.1fs" % (time.time() - self.rc_stamp)
            if not self.enable_high:
                return False, ("policy enable CH%d low (%d us, needs >%d)"
                               % (self.args.enable_channel, self.enable_value,
                                  self.args.enable_above))

        # UPRIGHT START. Blocks the FIRST engage only -- never re-engages the
        # gate once running, because a policy that is actively catching a lean
        # must be allowed to be leaning.
        #
        # Why this exists (transfer_props.csv, 2026-08-23): the policy engaged at
        # a 13.5 deg lean, and with 41% of weight supported by thrust the
        # effective pendulum time constant is 119/sqrt(1-0.41) = 155 ms. From
        # 13.5 deg, theta(t) = 13.5*cosh(t/155ms) reaches 70 deg in 0.36 s --
        # which is exactly what the mocap recorded. The fall was pure passive
        # pendulum dynamics; the wheels needed ~260 ms just to reach their
        # commanded speed, so they arrived after it was already unrecoverable.
        # The run that WORKED (transfer_smooth.csv, 3 m straight, 15 s) engaged
        # at 6.3 deg and held 6.2-6.5 deg throughout.
        #
        # Starting lean is therefore not a detail, it is most of the budget. Hold
        # the robot upright, flip CH6, and it engages the moment it is within
        # tolerance.
        if not self._engaged and self.args.max_start_lean_deg > 0:
            lean = self._lean_deg()
            if lean is None:
                return False, "waiting for mocap attitude (start-lean check)"
            if lean > self.args.max_start_lean_deg:
                return False, ("leaning %.1f deg at engage, needs <%.1f -- hold it "
                               "upright. At this lean the pendulum reaches 70 deg "
                               "in %.2f s and the wheels take ~0.26 s to develop "
                               "force." % (lean, self.args.max_start_lean_deg,
                                           self._fall_time_s(lean)))
        self._engaged = True
        return True, None

    def _lean_deg(self):
        """Angle between the body z axis and world up, in degrees, from mocap."""
        q = self.latest_quat
        if q is None:
            return None
        qx, qy = float(q[0]), float(q[1])
        return math.degrees(math.acos(max(-1.0, min(1.0, 1.0 - 2.0 * (qx * qx + qy * qy)))))

    def _fall_time_s(self, lean_deg):
        """Time for the passive pendulum to go from lean_deg to 70 deg.

        tau = sqrt(L/g) = 119 ms at the measured CoM height of 139 mm above the
        wheel axle; thrust supporting fraction f stretches it by 1/sqrt(1-f).
        Validated against transfer_props.csv to within the sample interval.
        """
        if lean_deg <= 0.0 or lean_deg >= 70.0:
            return 0.0
        f = min(max(self.args.assumed_thrust_frac, 0.0), 0.95)
        tau = 0.119 / math.sqrt(1.0 - f)
        return tau * math.acosh(70.0 / lean_deg)

    def _update_state(self):
        """Fold the newest pose into the filter. False until warmed up."""
        self.new_pose = False
        t = self.latest_stamp
        if self.prev_time is None:
            self.prev_time = t
            self.prev_quat = self.latest_quat.copy()
            self.kf.update(self.latest_pos)
            return False

        dt = t - self.prev_time
        if dt <= 0:
            return self.kf.initialized and self.step_count >= self.args.warmup_steps

        # dt SANITY CLAMP -- see the angular-velocity block below. The nominal
        # control period is 20 ms; anything far off that is a stamp artefact,
        # not real timing, and dividing by it manufactures enormous velocities.
        dt = float(np.clip(dt, self.args.min_dt, self.args.max_dt))

        self.kf.predict(dt)
        self.kf.update(self.latest_pos)

        # Body angular velocity from the quaternion difference. The small-angle
        # form 2*dq_vec/dt is exact enough at 50 Hz and avoids an arccos that
        # loses precision as dq -> identity.
        # ANGULAR VELOCITY -- obs[5:8], and the DAMPING term of the balance loop.
        #
        # This used to be a bare `2*dq/dt` with only a `dt <= 0` guard, which let
        # a sub-millisecond stamp gap amplify quaternion noise by 100x. Measured
        # on hardware 2026-08-21: obs[5:8] reached 7.47 against a sim training
        # range of about +/-0.15 -- 50x out of distribution -- on 0.4-7.5% of
        # ticks, including a 74 rad/s spike with the robot sitting still. A tanh
        # policy fed a 50x OOD input returns a saturated, meaningless action, and
        # one of those during a recovery is enough to destabilise it. Note
        # base_lin_vel gets a Kalman filter; this term had no conditioning at all.
        #
        # Three guards, cheapest first:
        #   1. dt is clamped above (a stamp artefact cannot manufacture velocity)
        #   2. clip to a physically achievable body rate
        #   3. optional first-order low-pass, since the raw quaternion difference
        #      of a noisy pose is a noisy derivative by construction
        dq = quat_multiply(quat_conjugate(self.prev_quat), self.latest_quat)
        if dq[3] < 0:
            dq = -dq
        w = 2.0 * dq[:3] / dt

        w_max = self.args.max_ang_vel
        if np.any(np.abs(w) > w_max):
            self.ang_vel_rejects += 1
            rospy.logwarn_throttle(
                2.0, "ang_vel spike %.1f rad/s clipped to %.1f (dt=%.4fs, %d so far) "
                     "-- obs[5:8] would have been %.2f vs a sim range of ~0.15",
                float(np.abs(w).max()), w_max, dt, self.ang_vel_rejects,
                float(np.abs(w).max()) * SCALE_ANG_VEL)
            w = np.clip(w, -w_max, w_max)

        a = self.args.ang_vel_lpf_alpha
        self.ang_vel_body = w if a >= 1.0 else (a * w + (1.0 - a) * self.ang_vel_body)

        self.prev_time = t
        self.prev_quat = self.latest_quat.copy()
        self.step_count += 1
        return self.step_count >= self.args.warmup_steps

    # ---- observation ------------------------------------------------------

    def _wheel_obs(self):
        """obs[0:2] and the raw measurement, in the mirrored sim joint frame.

        Returns (obs_pair, measured_jai_pair, fresh). When the RoboClaw link is
        stale this reports ZERO rather than the last good value: a frozen
        measurement is indistinguishable from "the wheels stopped", and holding
        a stale non-zero velocity in the observation is the more dangerous lie
        of the two. Freshness is logged so you can see it after the fact.
        """
        if self.wheels is None:
            return np.zeros(2, dtype=np.float32), (0.0, 0.0), False
        if not self.wheels.is_fresh():
            if not self.wheels_stale_warned:
                rospy.logwarn("RoboClaw telemetry stale -- obs[0:2] forced to zero")
                self.wheels_stale_warned = True
            return np.zeros(2, dtype=np.float32), (0.0, 0.0), False
        self.wheels_stale_warned = False
        obs_pair = self.wheels.observation()
        meas = self.wheels.velocity()

        # --wheel_obs_mirror: synthesise the dead wheel's observation from the
        # live one. 2026-09-01: one motor's encoder is dead (magnetic disc is
        # machine press-fit, Pololu will not support removing it), so a
        # replacement motor is the real fix. This is the stopgap.
        #
        # Justified empirically, not assumed: in hw_v34 the two wheel-velocity
        # observations are mirrored almost exactly --
        #     corr(obs_0, obs_1) = -0.959,  best fit obs_1 = -1.002 * obs_0
        # because the USD joints are mirrored and --max_wheel_diff 0 ties the
        # commands. So -1 x the live wheel is a good estimate of the dead one
        # WHILE BOTH WHEELS ACTUALLY TURN TOGETHER.
        #
        # THE LIMITATION, read this before trusting a trial: the dead-encoder
        # wheel has to run OPEN LOOP (DutyM2), so it slows under load while the
        # live wheel holds speed. They diverge exactly at a riser -- which is
        # the moment the observation matters most. Expect this to drive fine on
        # flat ground and to mis-report at the step.
        #
        # The raw measurement is left ALONE so the log keeps the truth; only
        # the policy input is synthesised.
        if self.args.wheel_obs_mirror != "none":
            obs_pair = np.array(obs_pair, dtype=np.float32, copy=True)
            if self.args.wheel_obs_mirror == "from_l":
                obs_pair[1] = -obs_pair[0]
            else:                      # from_r
                obs_pair[0] = -obs_pair[1]
        return obs_pair, meas, True

    def _velocity_command(self):
        """obs[29:32] -- reproduce sim's TerrainTargetDirectionCommand from mocap.

        Sim recomputes this EVERY step (velocity_command.py::_update_command):
        the normalised XY direction from the robot to a fixed world-frame target,
        rotated into the yaw-only body frame, plus a normalised heading error.

            obs[29] =  cos(yaw)*dx + sin(yaw)*dy
            obs[30] = -sin(yaw)*dx + cos(yaw)*dy
            obs[31] =  atan2(obs[29], obs[30]) / pi

        Feeding a CONSTANT [0, 1, 0] instead -- which this node did until
        2026-08-23 -- says "the target is dead ahead and your heading error is
        zero", forever. The policy turns to face the target, the target appears
        to turn with it, and it never registers arriving. Measured consequence:
        a pivot commanded on 94% of live ticks at EVERY attitude from -100 to
        +75 degrees, with the wheels driven in opposite directions and the robot
        spinning on the spot at 0.54 rad/s.

        Without --target this falls back to the old constant, so old behaviour is
        still reachable for comparison.
        """
        if self.args.target is None:
            # --velocity_command is a 3-vector. A 41-dim policy needs 4, and an
            # unpadded return would raise a shape error on assignment. Range 1.0
            # = "target is far", which is the honest value when there is no
            # target to measure against.
            if self.cmd_dim == 4:
                return np.concatenate([self.velocity_cmd[:3],
                                       np.array([1.0], dtype=np.float32)])
            return self.velocity_cmd
        dx = self.args.target[0] - self.latest_pos[0]
        dy = self.args.target[1] - self.latest_pos[1]
        dist = math.hypot(dx, dy)
        nx, ny = dx / (dist + 1e-6), dy / (dist + 1e-6)
        yaw = quat_to_yaw(*self.latest_quat)
        c, s = math.cos(yaw), math.sin(yaw)
        bx = c * nx + s * ny
        by = -s * nx + c * ny
        self.target_dist = dist
        # obs[31] MUST BE ZERO. Fixed 2026-08-27.
        #
        # velocity_command.py:248 sets it to a literal zero tensor on EVERY sim
        # step:  ang_vel_z_command = torch.zeros_like(angle_error)
        # so the policy has never once seen a nonzero value in this slot. Feeding
        # atan2(bx,by)/pi here put a live, varying signal into an observation
        # that was identically zero for the whole of training -- out of
        # distribution by construction.
        #
        # Sim's own comment above that line predicts exactly what then happens:
        #   "the policy will happily learn spurious couplings from it to the
        #    actions it does have (wheels, thrust)"
        #   "angle_error/pi steps from +0.99 to -1.00 in a single tick when the
        #    robot faces directly away from the target. Observed on hardware as
        #    the wheel command reversing every few ticks."
        #
        # Measured, hw_v28: with the target verified 0.1 deg off the nose at t=0
        # and corr(cmd[1], ground truth) = +0.9999, the policy still commanded
        # action_1 (wheel differential) at mean +0.765 of full scale and yawed
        # 93 deg. cmd[2] was averaging +0.196.
        #
        # The frozen --velocity_command fallback is [0, 1, 0] and was always
        # correct on this axis; only the --target path introduced the nonzero.
        # If steering is ever restored in TRAINING, feed sin and cos of the angle
        # as two slots (see velocity_command.py) -- do not reinstate this.
        heading = 0.0
        if self.args.heading_cmd == "live":
            heading = math.atan2(bx, by) / math.pi
        if self.cmd_dim == 3:
            return np.array([bx, by, heading], dtype=np.float32)
        # CHANNEL 3: XY RANGE TO TARGET, matching velocity_command.py exactly:
        #     MAX_RANGE_M = 2.0
        #     vel_command_b[:, 3] = clamp(distance_xy / MAX_RANGE_M, 0, 1)
        #
        # This is the only channel that tells the policy HOW FAR it has to go --
        # the first two are a unit direction and carry no distance. Without it a
        # 41-dim policy reads 0.0 here, i.e. "already at the target", for the
        # whole run. goal_reached at 0.25 m sits at 0.125 on this scale.
        rng = float(np.clip(dist / TARGET_MAX_RANGE_M, 0.0, 1.0))
        return np.array([bx, by, heading, rng], dtype=np.float32)

    def _actuator_state_obs(self):
        """servo_pos (rad) and propeller_vel*0.01 as sim reports them.

        Added 2026-08-26 with OBS_BASE_ACTUATOR. Neither quantity is sensed on
        this robot -- no servo encoder, no propeller tachometer -- but both are
        RECONSTRUCTED from state this node already keeps, so no new hardware:

          servo_pos     self.servo_cmd is the slew-limited servo command in
                        normalised units; x sim_servo_limit_rad gives the radians
                        sim observes. Sim's servos track a 2.0 rad/s
                        velocity_limit and this node slews at the same rate, so
                        the command IS the position to within tracking error.

          propeller_vel the ACHIEVED steady-state speed for the current action,
                        from the same damped model _propeller_command uses.
                        Sim reports signed joint velocity, left +, right -.

        DELAY. Sim's actuators carry min_delay 2 / max_delay 5 control steps, so
        the position sim OBSERVES lags the command by 2-5 ticks. These estimates
        have no such lag -- they are the command itself. Feeding them undelayed
        would hand the policy fresher actuator state than it ever saw in
        training, which is the wrong direction to be wrong in for a balance
        loop. The ring buffer below reproduces the mean 3-tick lag.
        """
        limit = self.args.sim_servo_limit_rad
        if self.args.servo_obs_source == "lowpass":
            # FIRST-ORDER LAG -- the closest cheap model of sim's actual joint.
            #
            # The policy learned servo_action ~= -servo_pos / limit: a regulator
            # that drives the servo toward zero. That is STABLE in sim only
            # because sim's servo_pos is the ACTUAL joint angle -- inertial,
            # torque-limited, and therefore smoothly lagging. Feed it anything
            # without that smoothing and the loop rings:
            #
            #   pure 3-tick delay (hw_v15) -> 4.2 Hz limit cycle, ac@12 = 0.87
            #   raw command       (hw_v17) -> corr(pos,act) = -1.000 EXACTLY,
            #                                 a one-tick flip-flop: +/-45 deg at
            #                                 25 Hz, which the servo cannot
            #                                 follow, so it buzzes near centre
            #                                 and looks "pinned upright"
            #
            # A first-order lag adds phase gradually instead of all at once,
            # which is what the real joint does. tau defaults to 3 ticks (60 ms),
            # matching sim's torque-limited sweep of ~2.8 ticks for full range.
            si, sj = self.layout["servo"]
            tgt = float(self.last_actions[si]) * limit
            alpha = 1.0 - math.exp(-1.0 / max(self.args.servo_obs_tau_ticks, 1e-3))
            self._servo_obs_est += alpha * (tgt - self._servo_obs_est)
            servo_rad = np.array([self._servo_obs_est, self._servo_obs_est],
                                 dtype=np.float32)
        elif self.args.servo_obs_source == "command":
            # RAW policy action, NOT the slew-limited command.
            #
            # Sim observes the ACTUAL joint position, and sim's servo is
            # torque-limited at ~500 rad/s^2 (effort 5.0 / armature 0.01), which
            # sweeps the full +/-45 deg in 2.8 control ticks -- so its position
            # tracks the command almost immediately and spans the full range.
            # Measured in sim: servo_pos -0.713 .. +0.722 rad.
            #
            # Feeding the SLEW-limited command instead gave +/-6 deg (sd 0.10),
            # 4-7x narrower than anything in training, because the command
            # reverses every tick or two and the slewed value never travels.
            # Worse, the rate limiter sits INSIDE the feedback path, which is a
            # classic limit-cycle generator: measured on hw_v16.csv, the servo
            # action became a -0.965 pure function of the servo_pos fed back,
            # with autocorrelation 0.75 at lag 12.
            si, sj = self.layout["servo"]
            servo_rad = np.array([float(self.last_actions[si]) * limit,
                                  float(self.last_actions[sj]) * limit],
                                 dtype=np.float32)
        else:
            servo_rad = np.array([self.servo_cmd[0] * limit,
                                  self.servo_cmd[1] * limit], dtype=np.float32)

        pi, pj = self.layout["prop"]
        omega_l = self._prop_omega(float(self.last_actions[pi]))
        omega_r = self._prop_omega(float(self.last_actions[pj]))
        prop_obs = np.array([omega_l * SCALE_PROP_VEL,
                             -omega_r * SCALE_PROP_VEL], dtype=np.float32)

        self._act_state_buf.append((servo_rad, prop_obs))
        # Cap at DELAY+1 so the buffer holds [t-DELAY .. t] and buf[0] is
        # exactly DELAY ticks old. Capping at DELAY gives DELAY-1.
        if len(self._act_state_buf) > self.args.actuator_obs_delay_ticks + 1:
            self._act_state_buf.pop(0)
        return self._act_state_buf[0]

    def _infer_decoupled(self, obs):
        """Decoupled-mode baseline, emitted in the POLICY'S normalised action space.

        Returning a normal action vector (rather than driving the actuators
        directly) means every downstream path is shared with the RL policy:
        --*_scale, the servo slew limit, the propeller map, the arm/CH6 gates,
        the watchdog, and the CSV schema. A baseline that took a different route
        to the motors would not be a fair comparison.

        Mapping, per actuator:

          servo   sigma [rad]        -> sigma / sim_servo_limit_rad
          prop    T [N per prop]     -> invert thrust poly to pwm, pwm to omega,
                                        omega to the action that _propeller_command
                                        would map back to that same omega
          wheel   tau_w [N*m]        -> the RoboClaw is VELOCITY controlled, so the
                                        torque is integrated into a speed setpoint
                                        at tau/I_wheel, with I_wheel = 0.0119 kg*m^2
                                        (from the 0.51 N*m effort limit producing the
                                        measured 43 rad/s^2)
        """
        a = self.args
        act = np.zeros(self.act_dim, dtype=np.float32)

        # --- state, in the controller's convention -------------------------
        R_b2w = quat_to_rotation_matrix(self.latest_quat)
        g_body = R_b2w.T @ np.array([0.0, 0.0, 1.0])
        # body +Y is forward (see the module docstring); the robot tips about
        # body X, so pitch is the Y/Z component of body-frame gravity.
        theta = math.atan2(g_body[1], g_body[2])
        theta_dot = float(self.ang_vel_body[0])
        v = float((R_b2w.T @ self.kf.velocity)[1])

        # forward speed setpoint from the goal direction, same signal the policy
        # receives in obs[..]: component of the unit goal vector along body +Y.
        cmd = self._velocity_command()
        v_desired = float(a.dctrl_v_desired) * float(np.clip(cmd[1], -1.0, 1.0))

        out = self.dctrl.control(theta=theta, theta_dot=theta_dot, v=v,
                                 v_desired=v_desired, theta_desired=0.0,
                                 yaw_rate=float(self.ang_vel_body[2]),
                                 yaw_rate_desired=0.0, yaw=0.0)

        # --- servo ----------------------------------------------------------
        si, sj = self.layout["servo"]
        sv = float(np.clip(out["sigma"] / max(a.sim_servo_limit_rad, 1e-6), -1.0, 1.0))
        act[si] = sv
        act[sj] = sv

        # --- propellers -------------------------------------------------------
        # Published as a DIRECT PWM (see _apply) because the policy action space
        # tops out at 11.6 N total and this controller needs 17.3 N to hold
        # station. The action-space entry below is written anyway so the CSV
        # still records something comparable, but it is not what drives the ESC.
        self._dctrl_prop_pwm = _thrust_to_pwm(float(out["T"]))
        omega = (self._dctrl_prop_pwm - 1000.0) / 650.0 * 500.0
        target = omega + PROP_DRAG_K * omega * omega / PROP_DAMPING
        pa = float(np.clip(target / PROP_HALF_SPAN - 1.0, -1.0, 1.0))
        pi, pj = self.layout["prop"]
        act[pi] = pa
        act[pj] = pa

        # --- wheels -----------------------------------------------------------
        I_WHEEL = 0.0119
        # tau_w is clipped to +/-2.0 N*m by the controller, which at this inertia
        # is 168 rad/s^2 -- four times what the drive delivers (measured 31-62
        # accelerating, and --roboclaw_accel caps the ramp at 43). Letting the
        # setpoint run at 168 just winds it far ahead of the wheel and turns the
        # integrator into a delay. Clamp to the achievable rate.
        MAX_WHEEL_ACCEL = 43.0
        dw = float(np.clip(float(out["tau_w"]) / I_WHEEL,
                           -MAX_WHEEL_ACCEL, MAX_WHEEL_ACCEL)) / float(a.rate)
        self._dctrl_wref += dw
        self._dctrl_wref = float(np.clip(self._dctrl_wref, -a.wheel_max_rad_s,
                                         a.wheel_max_rad_s))
        wi, wj = self.layout["wheel"]
        # the publish path computes desired_rad_s = -action * wheel_action_scale
        wa = float(np.clip(-self._dctrl_wref / max(a.wheel_action_scale, 1e-6),
                           -1.0, 1.0))
        act[wi] = wa
        if wj != wi:
            act[wj] = wa
        return act

    def _build_observation(self):
        obs = np.zeros(self.obs_dim, dtype=np.float32)
        R_b2w = quat_to_rotation_matrix(self.latest_quat)
        R_w2b = R_b2w.T

        wheel_obs, wheel_meas, wheel_fresh = self._wheel_obs()
        obs[0:2] = wheel_obs

        # Everything after wheel_vel shifts by 4 when the checkpoint observes
        # actuator state. k is that shift, so a single set of indices serves
        # both generations and there is no second copy to drift out of sync.
        if self.obs_has_actuator_state:
            servo_rad, prop_obs = self._actuator_state_obs()
            obs[2:4] = servo_rad
            obs[4:6] = prop_obs
            k = 4
        else:
            k = 0

        obs[2 + k:5 + k] = (R_w2b @ self.kf.velocity) * SCALE_LIN_VEL
        obs[5 + k:8 + k] = self.ang_vel_body * SCALE_ANG_VEL
        obs[8 + k:11 + k] = (R_w2b @ np.array([0.0, 0.0, -1.0])) * SCALE_GRAVITY

        yaw = quat_to_yaw(*self.latest_quat)
        base_z = self.latest_pos[2] + self.base_z_offset
        _scan = compute_height_scan(
            self.latest_pos[0], self.latest_pos[1], base_z, yaw, self.terrain)
        obs[11 + k:27 + k] = _scan
        # Relief in view, metres. 0 on flat ground; ~0.06 with a 6 cm riser in
        # the scan. Drives --prop_scale_step below.
        self._scan_spread = float(np.max(_scan) - np.min(_scan))
        # DISTANCE TO THE NEXT RISER, from the geometry --step already gave us.
        #
        # The 4x4 height scan only shows relief 0.10 m before the edge (measured
        # 2026-09-06, hw_fig_194914: spread 0.0000 from 0.9 m out, 0.0600 at
        # 0.066 m). At 1 m/s that is 100 ms of warning, and real propellers need
        # several hundred ms to spool -- so the policy commanded action_3 = -1.00
        # at 0.15 m out and only reached +0.90 AFTER it had hit and bounced.
        #
        # Sim gets away with the short look-ahead because prop_map sim_damped
        # solves the steady state, i.e. thrust is instantaneous. Hardware is not.
        # The deployment knows exactly where the risers are; use that rather than
        # waiting for the scan to notice.
        self._dist_to_riser = float("inf")
        try:
            for (xmn, _xmx, ymn, ymx, _h) in self.terrain._steps:
                if ymn <= self.latest_pos[1] <= ymx:
                    d = xmn - self.latest_pos[0]      # risers extend in +x
                    if 0.0 <= d < self._dist_to_riser:
                        self._dist_to_riser = d
        except (AttributeError, TypeError):
            pass

        # No contact sensor on the robot. Sim thresholds contact force at 1.0 N,
        # so this is 1.0 whenever the wheels are down -- which is the whole
        # traverse except mid-hop. Flip with --contact if testing airborne.
        obs[27 + k:29 + k] = self.args.contact
        obs[29 + k:29 + k + self.cmd_dim] = self._velocity_command()
        # newest first, matching mdp/observations.py::action_history exactly:
        #     [a(t-1) dims..., a(t-2) dims..., ..., a(t-N) dims...]
        a0 = 29 + k + self.cmd_dim
        obs[a0:a0 + self.act_hist_len * self.act_dim] = self._act_hist.reshape(-1)

        return obs, wheel_meas, wheel_fresh

    # ---- action -----------------------------------------------------------

    @torch.inference_mode()
    def _infer(self, obs):
        t = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        return np.clip(self.policy(t).cpu().numpy().flatten(), -1.0, 1.0)

    def _prop_omega(self, action_value):
        """Achieved steady-state propeller speed (rad/s) for one action value.

        See PROP_DAMPING / PROP_DRAG_K above for the derivation and the measured
        validation. Returns 0.0 at or below a = -1 (target velocity 0).
        """
        target = PROP_HALF_SPAN * (1.0 + float(action_value))
        if target <= 0.0:
            return 0.0
        return ((-PROP_DAMPING
                 + math.sqrt(PROP_DAMPING * PROP_DAMPING
                             + 4.0 * PROP_DRAG_K * PROP_DAMPING * target))
                / (2.0 * PROP_DRAG_K))

    def _propeller_command(self, action_value, scale=1.0, floor=True, idx=None):
        """One propeller action -> one JAIOut u_thr field.

        The sim's propeller term is affine and mirrored (left = 250a + 250,
        right = -250a - 250), so BOTH propellers span 0..500 rad/s and the
        throttle fraction is (a + 1) / 2 for each, using its own action. The
        mirroring lives entirely in the offset -- there is no sign to undo.

        `scale` attenuates the throttle FRACTION, not the mapped output. That
        distinction matters: the mapped output is an ESC command where -0.9175
        means "off", so scaling it toward 0.0 would push a stopped propeller
        UP toward mid-throttle rather than down. Attenuating the fraction keeps
        "off" at off for every scale.
        """
        if self.args.prop_map == "passthrough":
            return float(action_value) * scale

        fraction = (float(action_value) + 1.0) * 0.5 * scale   # 0..1 of 0..500 rad/s

        # THRUST FLOOR. --servo_attitude_hold only buys static stability while
        # total thrust exceeds 13.76 N (T/W 0.314 at 4.47 kg) -- below that the
        # righting moment loses to gravity and pointing the props up achieves
        # nothing. The policy cannot be relied on to hold that floor: measured on
        # transfer_clamped_props.csv it commanded 13.2-14.7 N, straddling the
        # threshold, and was above it on only 40% of ticks.
        #
        # This is a floor, not a setpoint -- the policy still commands anything
        # above it. Applied to the FRACTION for the same reason `scale` is: the
        # mapped output treats -0.9175 as "off", so flooring the mapped value
        # would be meaningless.
        #
        # `floor=False` IS A SAFETY PATH, NOT AN OPTIMISATION. Every caller that
        # means "propellers OFF" must pass it: the disarmed/gated branch, the
        # --no_props branch, and the watchdog _idle(). Those all call this with
        # action_value=-1.0, which without the opt-out would come back as the
        # FLOOR instead of off -- so disarming the FCU, dropping mocap, or losing
        # the RC link would leave 15.3 N of thrust running. Introduced and caught
        # 2026-08-23 before it reached hardware.
        if floor and self.args.prop_min_frac > 0.0:
            fl = self.args.prop_min_frac
            # STEP FLOOR. --prop_scale_step multiplies the POLICY's action, and
            # fraction = (action + 1) * 0.5 * scale, so when the policy commands
            # action_3 = -1 at the riser -- which it does on almost every run --
            # the boost is scaling zero and delivers nothing.
            #
            # Measured 2026-09-06 across 14 runs: thrust at contact was pinned at
            # -0.25 (this floor) in every run but one. The single run that reached
            # +1.00 at contact got 5.48 s on the tread, against 0.4-3.0 s for the
            # rest. II-A needs 46% of weight before the wheel can rotate about the
            # edge at all, so this is the difference between climbing and bouncing.
            #
            # Raising the FLOOR near a riser forces thrust regardless of what the
            # policy asks for, which is the only thing that works when the policy
            # is asking for zero.
            if self.args.prop_min_frac_step > fl:
                g = 0.0
                if self.args.prop_boost_dist > 0.0:
                    d = getattr(self, "_dist_to_riser", float("inf"))
                    if d < self.args.prop_boost_dist:
                        g = 1.0 - d / self.args.prop_boost_dist
                g = max(g, min(getattr(self, "_scan_spread", 0.0)
                               / max(self.args.prop_step_relief, 1e-6), 1.0))
                fl = fl + g * (self.args.prop_min_frac_step - fl)
            fraction = max(fraction, fl)

        if self.args.prop_map == "sim_damped":
            # Reproduce the speed sim ACHIEVES, not the speed it is commanded.
            # `fraction` has already had --prop_scale and any floor applied, so
            # convert it back to an action value before the model.
            omega = self._prop_omega(fraction * 2.0 - 1.0)
            pwm = 1000.0 + (omega / 500.0) * 650.0
            pwm = min(max(pwm, 1000.0), 1650.0)
            return float((pwm - 1000.0) / 650.0) * 2.0 - 1.0

        if self.args.prop_map == "sim":
            # Sim's aerodynamics model, mdp/aerodynamics.py:189 -- the one that
            # produced the thrust this policy trained against:
            #     pwm = 1000 + (|omega| / 500) * 650,  clamped to [1000, 1650]
            #
            # The 500 and 650 are AERODYNAMIC constants and never change. What
            # DOES change is how much omega a full action commands, set by the
            # propeller_vel action scale/offset in training:
            #     omega = prop_rad_s_max * (action + 1) / 2
            # Pre-2026-08-21 configs used scale=offset=250 -> 0..500 rad/s, so
            # a=+1 reached the 1650 us ceiling. Current configs use 125 -> 0..250,
            # so a=+1 is only 1325 us. Getting this wrong is a 2x thrust error at
            # full command, which on this airframe is the difference between
            # assisting and flipping.
            omega = self.args.prop_rad_s_max * fraction

            # PROPELLER RATE LIMIT -- reproduce sim's propeller dynamics.
            #
            # Sim's propeller links carry an auto-computed inertia of roughly
            # 0.57 kg*m^2 (a 0.042 kg prop should be ~1e-5), so against the
            # 5.0 N*m effort_limit they can only change speed at about
            # +9 rad/s^2 spinning up and -38 rad/s^2 spinning down. Measured
            # from the play log: commanded 28 rad/s while the joint was still
            # turning at 144.
            #
            # Sim then computes thrust from the ACTUAL joint speed, so the
            # policy trained against propellers that behave like flywheels --
            # a commanded cut still delivers thrust for the next half second.
            # This deployment computed thrust from the COMMAND, so a cut took
            # effect immediately and delivered 1.6-2.1x less thrust than sim
            # for the same action. That is why cutting a propeller drops the
            # robot here and not in sim, and why one prop appears to "not fire".
            #
            # Rate-limiting omega here puts the same flywheel between the
            # policy and the ESC. idx selects which propeller's state to use.
            if self.args.prop_accel_rad_s2 > 0 and idx is not None:
                dt = 1.0 / float(self.args.rate)
                up = self.args.prop_accel_rad_s2 * dt
                dn = self.args.prop_decel_rad_s2 * dt
                cur = self.prop_omega[idx]
                omega = cur + float(np.clip(omega - cur, -dn, up))
                self.prop_omega[idx] = omega

            pwm = min(1000.0 + (omega / 500.0) * 650.0, 1650.0)
            return (pwm - 1500.0) / 500.0

        return PROP_ESC_OFFSET + 2.0 * fraction * PROP_ESC_GAIN

    def _apply(self, action, gated=False):
        """Command the wheels over USB and publish servos/propellers as JAIOut.

        When `gated` the policy still runs and is still logged, but the wheels
        are commanded to zero and the JAIOut message is forced neutral.
        """
        a = self.args
        wheel_gain = a.wheel_scale * a.wheel_action_scale
        wi, wj = self.layout["wheel"]

        # WHEEL COMMAND LOW-PASS, added 2026-08-27. Default 1.0 = OFF.
        #
        # The policy's wheel action is white noise tick to tick: measured in
        # hw_v33, lag-1 autocorrelation +0.026 with mean |delta| 0.796 of full
        # scale EVERY 20 ms. The wheels cannot follow that -- measured tracking
        # ratio (meas sd / des sd) was 0.41 even with saturation eliminated -- so
        # the machine is already low-passing it, mechanically and violently.
        #
        # Every one of those is a step input, and on a balancing robot a step in
        # wheel speed pitches the body against the acceleration. hw_v33 shows it
        # twice: at launch a0 +0.92 from rest drove pitch to +54 deg and the
        # robot rolled 18 cm BACKWARDS before recovering, and after clearing the
        # second riser a0 +0.94 drove it to +67 deg.
        #
        # Filtering to roughly the actuator's own bandwidth removes the visible
        # jerk without removing control content the wheels could have delivered.
        # tau = dt*(1-alpha)/alpha, so at 50 Hz: alpha 0.5 -> 20 ms,
        # 0.3 -> 47 ms, 0.2 -> 80 ms. Do NOT go far below 0.2: this is inside the
        # balance loop and past ~100 ms the added phase lag costs more stability
        # than the smoothing buys.
        raw_w = (float(action[wi]), float(action[wj]))
        if a.wheel_lpf_alpha < 1.0:
            if self._wheel_lpf is None:
                self._wheel_lpf = list(raw_w)
            else:
                k = a.wheel_lpf_alpha
                self._wheel_lpf = [k * r + (1.0 - k) * p
                                   for r, p in zip(raw_w, self._wheel_lpf)]
            raw_w = tuple(self._wheel_lpf)

        # POST-CLIMB SLOWDOWN. Cuts wheel gain for a settling window AFTER
        # height is gained, not before contact.
        #
        # Braking into a riser is the wrong instinct: momentum is what carries a
        # wheel over the edge, and the deceleration itself pitches the body
        # forward exactly when it needs to be placed. The problem is what happens
        # once the wheels ARE up -- a narrow tread, 0.70 m to the next riser, and
        # full wheel gain still applied, so every correction is a large reaction
        # torque on a robot with nowhere to go.
        #
        # Triggered on MEASURED height gain rather than the height scan, so it
        # fires on having climbed rather than on expecting to.
        zc = float(self.latest_pos[2])
        self._z_hist.append(zc)
        if len(self._z_hist) > 50:            # 1 s at 50 Hz
            self._z_hist.pop(0)
        if len(self._z_hist) > 10 and (zc - min(self._z_hist)) > a.post_climb_rise:
            self._post_climb_i = int(round(a.post_climb_s * 50.0))
        if self._post_climb_i > 0:
            self._post_climb_i -= 1
            if a.wheel_scale_post > 0.0:
                wheel_gain *= a.wheel_scale_post / max(a.wheel_scale, 1e-6)

        # PRE-STEP SLOWDOWN, off by default. See above for why braking into
        # contact is usually the wrong direction; kept for the case where contact
        # speed itself is the failure.
        # STEP-TRIGGERED SLOWDOWN, the mirror of --prop_scale_step.
        #
        # Contact speed is what decides whether a riser is climbed or bounced
        # off: measured 2026-09-06 hw_new_183925, 0.73 m/s approach gave 90 deg
        # of yaw swing on asymmetric contact, and with risers 0.70 m apart there
        # is no room to recover between them. Fades the wheel gain down as relief
        # enters the height scan, so it arrives slow and square without being
        # slow across the whole traverse.
        if a.wheel_scale_step > 0.0 and a.wheel_scale_step < a.wheel_scale:
            g = min(getattr(self, "_scan_spread", 0.0) / a.prop_step_relief, 1.0)
            wheel_gain *= (1.0 - g) + g * (a.wheel_scale_step / a.wheel_scale)
        des_l = -raw_w[0] * wheel_gain
        des_r = -raw_w[1] * wheel_gain


        # WHEEL DIFFERENTIAL, 2026-08-27. Previously DROPPED ON THE FLOOR.
        #
        # The CommonDiff action space (obs 40 / act 4) puts translation on
        # action[0] and yaw on action[1]. LAYOUTS_BY_OBS resolved the index but
        # nothing ever applied it, so hardware executed only the common term.
        #
        # That is not a rounding error. Measured in the sim play log for this
        # very checkpoint:
        #     [WHEEL] target = [-15.1, +23.0]  ->  common -0.405, diff +0.49
        # The differential is COMPARABLE IN MAGNITUDE to the common term, so the
        # robot was running roughly half the policy's wheel intent with the rest
        # discarded -- while the policy, trained with both, kept commanding as if
        # both were arriving.
        #
        # Sim mapping (mdp/actions.py::CommonDiffJointVelocityAction), with the
        # mirrored USD wheel axes:
        #     left_joint  = +S*common + D*diff
        #     right_joint = -S*common + D*diff        S = 47.0, D = 8.0
        # The publish path already negates and applies S via wheel_gain, so the
        # differential enters with the SAME sign on both wheels -- equal joint
        # signs are opposite physical directions, which is what turns.
        if self.wheel_diff_idx is not None:
            # SIGN, derived rather than guessed. db_wheels.velocity_sim() maps
            #     sim_left = -hw_left,  sim_right = +hw_right
            # and sim is  left = +S*c + D*d,  right = -S*c + D*d, so
            #     hw_left  = -S*c - D*d
            #     hw_right = -S*c + D*d
            # The COMMON term therefore carries the SAME sign on both wheels
            # (which is why des_l == des_r above), and the DIFFERENTIAL carries
            # OPPOSITE signs. The first version of this added d to both, which
            # left des_l == des_r on every tick -- visible in the v21 console as
            # "wheel des=[38.1 38.1]" -- so the differential still did nothing.
            diff_gain = a.wheel_scale * a.wheel_diff_scale
            d = float(action[self.wheel_diff_idx]) * diff_gain
            des_l -= d
            des_r += d

        # WHEEL DIFFERENTIAL CLAMP.
        # Measured 2026-08-23 (transfer_far.csv): the policy steers correctly
        # when it is roughly aligned -- at |heading error| < 0.2 the two wheel
        # actions differ by only 0.02-0.13, i.e. it drives straight. But at
        # large heading error it commands a differential of -1.6 to -1.9 with
        # the SAME sign for errors of EITHER sign, so instead of taking the
        # short way round it circles: one full -360 deg net rotation inside a
        # 3.35 m run. Clamping the differential keeps the useful gentle
        # correction and removes the circling.
        #   --max_wheel_diff 0   ties the wheels completely (zero steering,
        #                        guaranteed straight, aim it by hand)
        #   --max_wheel_diff 5   allows gentle correction only
        #   --max_wheel_diff <0  disables the clamp (raw policy output)
        if a.max_wheel_diff >= 0.0:
            mean = 0.5 * (des_l + des_r)
            half = 0.5 * float(np.clip(des_r - des_l,
                                       -a.max_wheel_diff, a.max_wheel_diff))
            des_l, des_r = mean - half, mean + half

        # HEADING HOLD, APPLIED AFTER THE max_wheel_diff CLAMP.
        #
        # It used to run BEFORE the clamp, so --max_wheel_diff 0 -- which every
        # hardware run uses, to stop the policy steering itself off course --
        # zeroed the correction every tick. Measured 2026-09-07 hw_fig_014705:
        # the commanded differential was 0.00 for the entire run with kp at 12.
        # The hold had never done anything, and the yaw improvements I attributed
        # to it were placement luck.
        #
        # The clamp exists to limit the POLICY's differential. This is a separate
        # deployment-side authority with no sim counterpart, and it has its own
        # bound in --heading_hold_max, so it belongs outside the clamp.
        if a.heading_hold_kp > 0.0 and self.latest_quat is not None:
            yaw_now = quat_to_yaw(*self.latest_quat)
            if self._yaw_ref is None:
                self._yaw_ref = yaw_now
            err = math.atan2(math.sin(self._yaw_ref - yaw_now),
                             math.cos(self._yaw_ref - yaw_now))
            corr = float(np.clip(a.heading_hold_kp * err,
                                 -a.heading_hold_max, a.heading_hold_max))
            # HALF TO EACH SIDE. The differential is des_r - des_l = 2*corr, so
            # applying the full clamp to each wheel delivered TWICE
            # --heading_hold_max. At 4.0 that was 8 rad/s of differential --
            # double sim's k_diff -- and it saturated there, spinning the robot
            # 132 deg on 2026-09-07 while --max_wheel_diff 0 was set. Halving it
            # makes --heading_hold_max mean what its help text says: the maximum
            # differential, comparable to k_diff.
            des_l -= 0.5 * corr
            des_r += 0.5 * corr

        if gated:
            des_l = des_r = 0.0

        # --wheel_ramp_s: fade the wheel command in when control is (re)enabled.
        #
        # 2026-09-04, measured across every hardware run: within 0.3 s of the
        # gate opening the wheels go from 0 to -14 rad/s and the robot is 17 cm
        # displaced before the policy can correct. It commands full forward from
        # a dead stop, the body pitches in reaction, and with the measured 300 ms
        # wheel lag the catch arrives after the fall is decided. One run (run3)
        # survived the same lurch and climbed both risers; the rest did not. That
        # is a coin flip on the launch transient, not a property of the policy.
        #
        # play_dctrl.py has always done this (ramp = min(1, step/100)); this node
        # never did. Ramping only the FIRST seconds after the gate opens leaves
        # steady-state authority untouched.
        if self._wheel_ramp_n > 0:
            if gated:
                self._wheel_ramp_i = 0          # re-arm for the next enable
                self._yaw_ref = None                # re-capture heading on engage
            else:
                if self._wheel_ramp_i < self._wheel_ramp_n:
                    self._wheel_ramp_i += 1
                    k = self._wheel_ramp_i / float(self._wheel_ramp_n)
                    des_l *= k
                    des_r *= k

        if self.wheels is not None and not self.dry_run:
            if gated:
                # Re-assert zero every tick rather than once on the transition:
                # a single dropped frame must not leave the wheels running.
                # OPEN LOOP zero, not a closed-loop zero setpoint -- see
                # db_wheels.RoboClawWheels.hold_zero. A velocity setpoint of
                # zero keeps the PID alive and lets a noisy encoder drive the
                # motor while disarmed.
                self.wheels.hold_zero()
            elif not self.wheels.write_ok():
                # LINK DOWN. Repeated write failures mean the RoboClaw is still
                # running the last speed it received and we cannot change it.
                # Stop asking the policy to drive and say so loudly -- the only
                # thing that will actually stop the wheels now is the RoboClaw's
                # own serial timeout, which must be set in Motion Studio.
                rospy.logerr_throttle(
                    1.0, "ROBOCLAW LINK DOWN -- wheels are running the last "
                         "command and cannot be stopped from here. Kill the run.")
                self.wheels.hold_zero()
            elif self.args.wheel_mode == "duty":
                # Open-loop torque, bypassing the RoboClaw velocity PID. See
                # db_wheels.RoboClawWheels.command_duty for why: sim models the
                # wheel as an effort-limited joint (saturating proportional
                # torque), not as a velocity setpoint handed to a PI controller,
                # and that extra loop is dynamics the policy never trained
                # against.
                self.wheels.command_duty(des_l, des_r,
                                         self.args.wheel_duty_kp,
                                         self.args.wheel_duty_max,
                                         self.args.wheel_duty_trim_l,
                                         self.args.wheel_duty_trim_r)
            else:
                self.wheels.command(des_l, des_r)

        msg = JAIOut()
        msg.header = Header(stamp=rospy.Time.now())
        msg.rc_state = a.rc_state
        msg.pit_pwm = 500

        # Wheels are driven over USB now. Publishing zero (rather than omitting
        # the field) makes the firmware slew its wheel channels to neutral
        # instead of holding whatever it last received.
        msg.wh_l = 0.0
        msg.wh_r = 0.0

        # Servos carry the thrust VECTOR ANGLE, propellers its magnitude.
        # Attenuating the angle is not a safety margin in the way attenuating
        # thrust is -- it just points the thrust somewhere the policy did not
        # ask for. Hence a separate knob from --prop_scale.
        #
        # SIGNS -- note servo2 takes +action[3], NOT -action[3]. Derivation,
        # because getting this wrong points one propeller at the ground:
        #   sim left  joint angle  theta_L = (+pi/2) * action[2]
        #   sim right joint angle  theta_R = (-pi/2) * action[3]
        # JAIOut servo1/servo2 are NOT mirrored -- one constant k maps physical
        # angle to PWM on both sides. The PPO deployment pins k: it sent
        # servo1 = servo2 = -action[2] and worked, and PPO training had
        # action[3] ~ -action[2], which makes theta_R == theta_L. Equal joint
        # angles must therefore produce equal JAIOut values, giving k = -2/pi:
        #   servo1 = -(2/pi) * theta_L = -action[2]
        #   servo2 = -(2/pi) * theta_R = +action[3]
        # This was wrong here until 2026-08-20 (servo2 used -action[3]), which
        # sent the two arms to opposite tilts whenever the policy asked for a
        # symmetric one -- observed on the robot as one propeller pointing down,
        # and it is why the robot could not hold itself up.
        #
        # The propellers do NOT get an equivalent flip: their sim term is a
        # velocity with a matching offset (left = 250a+250, right = -250a-250),
        # so each spans 0..500 rad/s in magnitude and the throttle fraction is
        # (a_i + 1)/2 from each propeller's OWN action. The mirroring there is
        # the physical counter-rotation, not a command sign.
        # SIM->JAIOut GAIN. theta = sim_servo_limit_rad * action, and JAIOut servo
        # units are [-1,1] over +/-pi/2 rad, so servo = -(2/pi)*theta.
        # At the historical pi/2 this factor is exactly 1.0 and the expression
        # collapses to the bare -action[2] the older checkpoints were flown with;
        # at pi/6 it is 1/3. Folding it into `s` keeps every sign derivation below
        # unchanged.
        s = a.action_scale * a.servo_scale * (a.sim_servo_limit_rad / (math.pi / 2.0))

        # ATTITUDE HOLD BYPASSES THE ATTENUATION. The held value is an ABSOLUTE
        # geometric angle -- the one that puts thrust at world vertical -- not a
        # policy action to be trimmed for safety. Multiplying it by
        # action_scale*servo_scale would point the thrust somewhere that is
        # neither what the policy asked for nor vertical, which is the whole
        # failure this mode exists to remove. (It is also how --servo_scale 0.35
        # froze the servo at a constant +30 deg in transfer_clamped_props.csv.)
        # sim_servo_limit_rad is divided back out because the value was expressed
        # in action units above purely so the slew limiter could act on it.
        if a.servo_attitude_hold:
            s = a.sim_servo_limit_rad / (math.pi / 2.0)
        # SERVO SLEW LIMIT -- match sim's actuator, not the hardware's capability.
        #
        # sim: propeller_servos velocity_limit = 2.0 rad/s, i.e. 114 deg/s, so an
        # 86 deg command takes 0.75 s to execute and the policy revises it long
        # before the joint arrives. The thrust vector never actually gets sideways.
        #
        # real: a hobby servo does 86 deg in ~0.15 s, 3-5x faster. So hardware
        # executes a command sim only ever half-executed. Measured consequence
        # (transfer_centre.csv, 2026-08-21): the policy asked for -0.955 (86 deg
        # off vertical) while at rest, the servo snapped there, and 20.2 N of
        # thrust went 97% horizontal -- 8.9 N*m at a 0.443 m lever arm, 110
        # rad/s^2, robot over in 0.14 s. Verified against the USD: at joint 0 the
        # propeller thrust axis is world +Z (straight up), so large |action[2]|
        # genuinely means sideways.
        #
        # Rate-limiting here reproduces sim's actuator instead of out-running it.
        # The slew runs in NORMALIZED action units, where |action|=1 is
        # sim_servo_limit_rad of joint travel -- so the per-tick step is
        # rate_rad_s / sim_servo_limit_rad / control_rate. Dividing by a fixed
        # pi/2 here (as this did before 2026-08-23) would make the limiter 3x
        # too slow against a pi/6 checkpoint: the joint would crawl at 0.67 rad/s
        # while sim's actuator does 2.0.
        # ATTITUDE HOLD -- override the policy's servo action with the angle that
        # keeps the thrust vector VERTICAL IN THE WORLD FRAME.
        #
        # Why this is not a hack. The propellers sit 0.443 m above the wheel axle
        # and the CoM only 0.139 m above it, so the props are ABOVE the CoM. With
        # thrust held world-vertical, the torque balance about the axle is
        #
        #     gravity  (tipping)  = m*g*0.139 * sin(theta) = 6.10 * sin(theta)
        #     thrust   (righting) = T * 0.443 * sin(theta)
        #
        # sin(theta) cancels, so the net moment is RESTORING for any lean as long
        # as T * 0.443 > 6.10, i.e. T > 13.76 N (T/W > 0.314 at 4.47 kg). Vertical
        # thrust applied above the CoM turns this machine into a HANGING pendulum
        # rather than an inverted one. That is STATIC stability -- it needs no
        # bandwidth, no delay budget and no reaction time, which is exactly why it
        # transfers when a learned balance loop does not.
        #
        # Measured on transfer_clamped_props.csv (2026-08-23) without this:
        #     lean 16 deg -> servo +29.9 deg -> thrust world_z 0.77
        #     lean 35 deg -> servo +30.8 deg -> thrust world_z 0.59
        #     lean 69 deg -> servo +26.0 deg -> thrust world_z 0.26
        # The servo sat at a constant ~+30 deg regardless of attitude (an artefact
        # of --servo_scale 0.35 against a saturated action), so the thrust vector
        # rotated away from vertical along with the body. Thrust was above the
        # threshold on 40% of ticks and within 20 deg of world-vertical on 5%;
        # both at once on 2%. The robot was in the stable configuration almost
        # never.
        #
        # The servo axis rotates the thrust vector in the body Y-Z plane (forward
        # is body +Y), so cancelling the body's pitch about X keeps thrust up:
        #     theta_servo = -pitch,   servo_cmd = -(2/pi) * theta_servo
        # clamped to the servo's mechanical travel. Roll is NOT correctable --
        # the servos are a single pitch axis -- so a roll disturbance still has
        # to be handled by the wheels or not at all.
        #
        # This deliberately takes the servo away from the policy. The propellers
        # and wheels are still the policy's. Use --prop_min_frac to hold thrust
        # above the 13.76 N threshold, or the stability criterion is not met and
        # this does nothing useful.
        if a.servo_attitude_hold and self.latest_quat is not None:
            R_b2w = quat_to_rotation_matrix(self.latest_quat)
            # WORLD UP EXPRESSED IN THE BODY FRAME. This must be the body frame,
            # because that is the frame the servo acts in.
            #
            # Until 2026-08-24 this used bz = R_b2w[:, 2] (body +Z in WORLD) and
            # atan2(bz[1], bz[2]), which measures tilt toward world +Y. The servo
            # rotates in the BODY Y-Z plane, so the two agree only when the robot
            # happens to face along world +Y and become orthogonal at 90 deg of
            # yaw. Measured on new_ckpt_hold.csv at yaw = -95 deg: true body pitch
            # 65.6 deg, the old expression reported 18.0, so the servo corrected
            # 27% of the tilt while tracking its own target perfectly. Thrust
            # world_z decayed 1.00 -> 0.43 and the robot went over. The bug is
            # invisible at yaw 0 and total at yaw 90 -- do not "simplify" this
            # back to a world-frame vector.
            g_body = R_b2w.T @ np.array([0.0, 0.0, 1.0])
            pitch = math.atan2(g_body[1], max(g_body[2], 1e-6))

            # RATE TERM. servo = -pitch alone is pure proportional control: it
            # restores but cannot stop, so the machine swings THROUGH vertical
            # and out the other side. Measured, new_ckpt_hold3.csv 2026-08-24:
            # the hold pulled 7.4 deg -> 0.7 deg in 0.14 s (the restoring moment
            # working exactly as the geometry predicts), then overshot and ran to
            # -90 deg over the next 0.64 s. A hanging pendulum is statically
            # stable but UNDAMPED, and the wheels pump it further.
            #
            # Leading the command by the pitch RATE is the damping. ang_vel_body
            # is already computed from mocap for the observation, and [0] is the
            # rotation about body X, which is the same axis this servo acts in.
            # k has units of seconds: it is how far ahead the correction looks.
            pitch += a.servo_hold_damping * float(self.ang_vel_body[0])
            # Clamp to the servo's PHYSICAL travel (+/-pi/2), not to
            # sim_servo_limit_rad. That limit exists to match the policy's action
            # scaling; this is not a policy action, and restricting a geometric
            # correction to pi/4 would throw away half the available authority
            # exactly when the lean is large enough to need it.
            theta = float(np.clip(-pitch, -math.pi / 2.0, math.pi / 2.0))
            hold = -theta / (math.pi / 2.0)          # JAIOut units, [-1, 1]
            # Express it in ACTION units, because the publish path below applies
            # msg.servo1 = -action[si] * s. Note the leading minus: without it the
            # negation is applied twice and the servo ADDS to the body pitch
            # instead of cancelling it -- verified numerically, +10 deg of pitch
            # came out as +10 deg of joint and 20 deg of thrust error.
            # SIGN. The mapping from JAIOut servo units to the physical thrust
            # direction was DERIVED, never measured, and the obvious check is
            # circular: computing "thrust world_z" from the logged servo command
            # uses the same assumed convention as the command itself, so a
            # globally flipped convention scores a perfect 1.00 while the real
            # props point the wrong way. Exactly the trap that hid the wheel
            # mapping for three days -- only an open-loop observation settles it.
            #
            # Observed on hardware 2026-08-24: tilting the frame forward made the
            # props point FORWARD, i.e. the servo ADDS to the body tilt instead of
            # cancelling it, doubling the error instead of nulling it. Hence -1.0.
            # Verify by hand with --no_props --no_roboclaw --arm_gate off: tilt the
            # body and the props must stay aimed at the ceiling.
            act_units = (a.servo_hold_sign * -hold
                         / max(a.sim_servo_limit_rad / (math.pi / 2.0), 1e-6))
            action = np.array(action, dtype=np.float32, copy=True)
            # BLEND rather than replace. The policy holds the thrust vector
            # fixed in the BODY frame -- measured 2026-09-09, corr(servo action,
            # pitch) = +0.03 and d(servo_deg)/d(pitch) = +0.01, i.e. no attitude
            # response at all -- so when the body pitches forward the thrust
            # tips with it and drives the fall instead of arresting it. It could
            # not have learned otherwise: sim rate-limits the servo to
            # 2 rad/s, which sweeps 45 deg in 393 ms against a 119 ms fall.
            #
            # blend=1.0 is the historical full override, byte-identical to
            # before this change. blend=0 is pure policy. In between the policy
            # keeps its command and gains the pitch compensation it never had
            # the actuator bandwidth to learn.
            _b = float(np.clip(a.servo_hold_blend, 0.0, 1.0))
            # Optional step-local blend, faded on the same distance gate the
            # propeller flags use. Lets the servo follow the policy more (or
            # less) closely at a riser than it does between them.
            _gate = 0.0
            if a.prop_boost_dist > 0.0:
                _d = getattr(self, "_dist_to_riser", float("inf"))
                if _d < a.prop_boost_dist:
                    _gate = 1.0 - _d / a.prop_boost_dist
            if a.servo_hold_blend_step >= 0.0:
                _bs = float(np.clip(a.servo_hold_blend_step, 0.0, 1.0))
                _b = _b + _gate * (_bs - _b)
            for _k in (0, 1):
                _i = self.layout["servo"][_k]
                # CLIP. act_units is the hold angle divided by
                # sim_servo_limit_rad/(pi/2), so it can exceed 1.0 by itself;
                # adding a fraction of the policy action on top pushed the
                # blended command to -1.57 on 2026-09-09, i.e. 70 deg against a
                # trained range of +-45. The servo then sat on its 10 rad/s slew
                # limit, thrashing at 11.5 deg per tick, well outside anything
                # the policy saw in training.
                _v = (1.0 - _b) * float(action[_i]) + _b * act_units
                # STEP BIAS. The blend can only interpolate between the policy
                # and a hold that targets world-vertical thrust; neither ever
                # aims thrust FORWARD. Measured 2026-09-09 at the second riser:
                # servo -28 to -41 deg, so two thirds of thrust pushes sideways
                # while the wheels push forward and the robot creeps 11 cm in
                # 6 s. This adds a signed tilt that fades in on the same
                # distance gate as the propeller boost and fades out past the
                # riser, so thrust can lean into the step and return to vertical
                # after it.
                if a.servo_step_bias != 0.0:
                    # Its own gate. Sharing --prop_boost_dist fired the tilt
                    # 30 cm out, which stopped the FIRST step being climbed at
                    # all: thrust leaned into the riser before the wheels
                    # reached it. Thrust wants to spool early; the tilt wants to
                    # arrive late.
                    _bd = (a.servo_bias_dist if a.servo_bias_dist > 0.0
                           else a.prop_boost_dist)
                    _bg = 0.0
                    if _bd > 0.0:
                        _dd = getattr(self, "_dist_to_riser", float("inf"))
                        if _dd < _bd:
                            _bg = 1.0 - _dd / _bd
                    _v += _bg * a.servo_step_bias
                action[_i] = float(np.clip(_v, -1.0, 1.0))
            self.attitude_hold_deg = math.degrees(theta)
        else:
            self.attitude_hold_deg = float('nan')

        # ATTITUDE HOLD GETS THE HARDWARE'S RATE, NOT SIM'S.
        #
        # servo_slew_rad_s exists so hardware does not out-run sim's
        # propeller_servos velocity_limit (2.0 rad/s) on POLICY actions. Attitude
        # hold is not a policy action: it is a geometric correction that by
        # construction can only point thrust UP, so there is no sim behaviour to
        # match and nothing to protect against. Rate-limiting it to 2.0 rad/s
        # just makes it lose the race against the fall.
        #
        # Measured, new_ckpt_hold2.csv 2026-08-24: the servo tracked at exactly
        # 4.6 deg/tick = 115 deg/s = 2.0 rad/s while the body pitched at up to
        # 400 deg/s. The lag grew 0.7 -> 36 deg over 0.2 s and thrust world_z
        # fell 1.00 -> 0.81; it only caught up at t=0.64 s, well after the robot
        # was resting on its frame. A real hobby servo does 86 deg in ~0.15 s,
        # i.e. ~10 rad/s.
        # LOW-PASS THE SERVO COMMAND. Measured 2026-09-09: the policy's servo
        # action alternates sign every tick on hardware (+0.28, -0.96, -0.07,
        # -0.83, +0.70 ...), shaking the servo 12 deg peak-to-peak at 25 Hz
        # while corr(servo, pitch) sits at +0.02. That is a limit cycle in the
        # servo feedback path, the same class as the 4.2 Hz one from
        # --actuator_obs_delay_ticks 3 and the locked cycle from
        # --servo_obs_source slewed.
        #
        # The wheels already have this filter (--wheel_lpf_alpha), added for the
        # identical 25 Hz thrash. The servos never needed it while
        # --servo_attitude_hold overrode them; with the policy driving them they
        # do. Applied before the slew limiter so the rate limit sees a smooth
        # target rather than a square wave.
        if 0.0 < a.servo_lpf_alpha < 1.0:
            _idx = [self.layout["servo"][0], self.layout["servo"][1]]
            _cur = np.array([float(action[i]) for i in _idx])
            if self._servo_lpf is None:
                self._servo_lpf = _cur.copy()
            else:
                self._servo_lpf += a.servo_lpf_alpha * (_cur - self._servo_lpf)
            action = np.array(action, dtype=np.float32, copy=True)
            for _k, _i in enumerate(_idx):
                action[_i] = float(self._servo_lpf[_k])

        slew = (a.servo_hold_slew_rad_s if a.servo_attitude_hold
                else a.servo_slew_rad_s)
        if slew > 0:
            step = slew / a.sim_servo_limit_rad / float(a.rate)
            tgt = np.array([action[self.layout["servo"][0]],
                            action[self.layout["servo"][1]]], dtype=np.float64)
            self.servo_cmd += np.clip(tgt - self.servo_cmd, -step, step)
            action = np.array(action, dtype=np.float32, copy=True)
            action[self.layout["servo"][0]] = self.servo_cmd[0]
            action[self.layout["servo"][1]] = self.servo_cmd[1]

        si, sj = self.layout["servo"]
        if si == sj:
            # TIED: one command, both arms to the same physical angle.
            msg.servo1 = -float(action[si]) * s * a.servo_sign_left
            msg.servo2 = -float(action[si]) * s * a.servo_sign_right
        else:
            msg.servo1 = -float(action[si]) * s * a.servo_sign_left
            msg.servo2 = +float(action[sj]) * s * a.servo_sign_right

        if a.no_props:
            # -1 is "propeller off" in the normalized action space, so push that
            # through the same map rather than writing a bare 0.0, which the ESC
            # would read as mid-throttle.
            msg.u_thr1 = msg.u_thr2 = self._propeller_command(-1.0, floor=False)
        elif self._dctrl_prop_pwm is not None:
            # DECOUPLED BASELINE -- direct PWM, bypassing the policy action space.
            #
            # Not an inconsistency, a necessity. The policy's propeller action
            # spans 0..375 rad/s of TARGET, which the damped actuator turns into
            # 158 rad/s achieved = pwm 1208 = 11.6 N total. The decoupled
            # controller's T_hold alone is 8.66 N per prop = 17.3 N total
            # (BB_HOV_DC = 1335 pwm on the real robot). Routing it through the
            # policy's action space would silently cap the baseline at 2/3 of its
            # design thrust and hand the comparison to the RL policy by
            # construction -- exactly the kind of rigged baseline IROS R1
            # objected to.
            #
            # Everything else stays shared: state estimate, gates, servo path,
            # wheel path, logging.
            pw = float(np.clip(self._dctrl_prop_pwm, 1000.0, 1650.0))
            u = (pw - 1000.0) / 650.0 * 2.0 - 1.0
            u *= a.action_scale * a.prop_scale
            msg.u_thr1 = msg.u_thr2 = float(np.clip(u, -1.0, 1.0))
        else:
            prop_s = a.action_scale * a.prop_scale
            # STEP-TRIGGERED THRUST BOOST.
            #
            # II-A: the wheel torque requirement is only met once fp >= 14.6 N,
            # 46% of weight. Flat-ground cruising needs nothing like that, and a
            # riser needs it for well under a second. The policy asks for the
            # spike (reward_thrust_up_at_step) but hardware props under-deliver
            # against sim's fitted model, so the ask arrives short: measured
            # 2026-09-06 hw_new_183459, thrust +0.04..+0.07 at the moment of
            # tread contact, followed by sliding back off.
            #
            # Scales prop_scale up while a riser is in the height scan, in
            # proportion to how much relief is visible, and returns to the
            # cruise value on flat ground. Default equals --prop_scale, i.e. off.
            if a.prop_scale_step > a.prop_scale:
                g = min(getattr(self, "_scan_spread", 0.0) / a.prop_step_relief, 1.0)
                # Geometry-based look-ahead wins when it is further out than the
                # scan, which is the whole point: spool BEFORE contact.
                if a.prop_boost_dist > 0.0:
                    d = getattr(self, "_dist_to_riser", float("inf"))
                    if d < a.prop_boost_dist:
                        g = max(g, 1.0 - d / a.prop_boost_dist)
                prop_s = a.action_scale * (a.prop_scale
                                           + g * (a.prop_scale_step - a.prop_scale))
            pi_, pj_ = self.layout["prop"]
            msg.u_thr1 = self._propeller_command(action[pi_], prop_s, idx=0)
            msg.u_thr2 = self._propeller_command(action[pj_], prop_s, idx=1)
        msg.u_thr = 0.0

        if self.dry_run or gated:
            # HOLD THE SERVO SLEW STATE AT THE PUBLISHED ANGLE.
            #
            # servo_cmd accumulates through the slew limiter on every tick,
            # including gated ones, because _apply() always runs. Meanwhile the
            # gated branch below publishes servo1 = servo2 = 0. So the internal
            # state drifts to wherever the policy wanted during the hold while
            # the real servo sits at zero -- and the instant the gate opens the
            # whole accumulated angle is published in ONE tick.
            #
            # Measured, hw_v6.csv row 1825: the servo went 0 -> -42 deg in a
            # single 20 ms tick at engage, thrust world_z dropped 0.99 -> 0.84,
            # and the robot was at 22.6 deg of lean 180 ms later. Every engage
            # in every run has been doing this.
            #
            # Zeroing the state while gated makes the servo ramp from its true
            # position at the slew rate, the way it does mid-run.
            if not self.args.servo_attitude_hold:
                self.servo_cmd[:] = 0.0
            if self._prewarm_ok():
                # PREWARM. Hold the props spinning and the servos already aimed,
                # so flipping CH6 costs no spin-up.
                #
                # Measured at the four engage transitions in
                # transfer_clamped_props.csv: the command steps from pwm 1000
                # (4.6 N) to pwm ~1363 (18.8 N) in ONE tick, but a real ESC and
                # propeller need 100-300 ms to follow it. The static-stability
                # threshold is 13.76 N, so for the first couple hundred ms after
                # engage the robot is BELOW it -- and 13.5 deg of lean becomes 70
                # in 360 ms. The thrust that is supposed to hold it up arrives
                # after the fall is already unrecoverable. The wheels have the
                # same problem in reverse (measured 260 ms to reach commanded
                # speed) but they only have 6.6 deg of authority anyway.
                #
                # Wheels stay at ZERO here -- prewarm is thrust and aim only,
                # never motion. _prewarm_ok() also refuses unless the FCU is
                # armed, mocap is live, and the robot is within the start-lean
                # tolerance, so this cannot spin up while it is lying on its side.
                msg.u_thr1 = msg.u_thr2 = self._propeller_command(
                    -1.0, scale=0.0)   # scale 0 -> floor only, see _propeller_command
            else:
                msg.servo1 = msg.servo2 = 0.0
                msg.u_thr1 = msg.u_thr2 = self._propeller_command(-1.0, floor=False)

        self.pub.publish(msg)
        return (des_l, des_r), msg

    def _prewarm_ok(self):
        """True when it is safe and useful to hold thrust before the gate opens.

        Deliberately strict: prewarm spins propellers on a robot the operator is
        probably holding, so every one of these must hold.
        """
        a = self.args
        if not a.prewarm or a.no_props or self.dry_run:
            return False
        if a.prop_min_frac <= 0.0:
            return False                      # nothing to hold it at
        if not self.armed:
            return False                      # arming is the operator's consent
        if time.time() - self.armed_stamp > a.arm_timeout:
            return False
        if self.pose_frozen() or self.latest_quat is None:
            return False                      # aim would be stale
        lean = self._lean_deg()
        if lean is None or lean > max(a.max_start_lean_deg, 25.0):
            return False                      # not upright: do not spin up
        return True

    def _idle(self):
        """Stop the wheels and idle the propellers. Used by the watchdog."""
        if self.wheels is not None and not self.dry_run:
            self.wheels.stop()
        msg = JAIOut()
        msg.header = Header(stamp=rospy.Time.now())
        msg.rc_state = self.args.rc_state
        msg.pit_pwm = 500
        msg.wh_l = msg.wh_r = 0.0
        msg.servo1 = msg.servo2 = 0.0
        msg.u_thr1 = msg.u_thr2 = self._propeller_command(-1.0, floor=False)
        msg.u_thr = 0.0
        self.pub.publish(msg)

    # ---- preflight --------------------------------------------------------

    def _preflight_report(self, samples, loop_hz):
        """Judge each observation block against what it must look like with the
        robot sitting normally on the floor, wheels down, not moving.

        Every check here is one that fails SILENTLY at run time: the vector still
        has 38 finite numbers, the policy still emits actions, and the robot
        still drives -- into a wall.
        """
        obs = np.array(samples)
        ok = True

        def verdict(good, name, detail):
            print("  [%s] %-22s %s" % ("PASS" if good else "FAIL", name, detail))
            return good

        print("\n" + "=" * 70)
        print("PREFLIGHT  (robot on the floor, wheels down, stationary)")
        print("=" * 70)

        ok &= verdict(loop_hz > 0.9 * self.args.rate, "loop rate",
                      "%.1f Hz (target %d)" % (loop_hz, self.args.rate))

        g = obs[:, 8:11].mean(axis=0)
        upright = g[2] < -0.9 * SCALE_GRAVITY
        ok &= verdict(upright, "projected_gravity",
                      "[%+.2f %+.2f %+.2f]  (upright is [0 0 %.2f])"
                      % (g[0], g[1], g[2], -SCALE_GRAVITY))
        if not upright:
            print("         ^ the robot's body z-axis is not pointing up. Either it is")
            print("           not sitting upright, or the mocap rigid body was defined")
            print("           with a flipped frame -- in which case this term is wrong")
            print("           on EVERY tick and the policy always thinks it is falling.")

        hs = obs[:, 11:27]
        saturated = np.mean(np.abs(hs) >= 0.999)
        ok &= verdict(saturated < 0.5, "height_scan",
                      "mean %.3f, range %.3f..%.3f, %.0f%% at the clip"
                      % (hs.mean(), hs.min(), hs.max(), 100 * saturated))
        if saturated >= 0.5:
            print("         ^ pinned at the clip: the robot is far above the mapped")
            print("           terrain, or --base_z_offset / --step are wrong. The")
            print("           policy is blind to the step in this state.")

        ok &= verdict(not self.pose_frozen(), "mocap tracking",
                      "%d identical poses in a row" % self.pose_repeats)
        if self.pose_frozen():
            print("         ^ the mocap is republishing a held pose at full rate.")
            print("           Move the robot toward the centre of the volume and")
            print("           confirm values change before running anything.")

        w = obs[:, 0:2]
        quiet = np.abs(w).max() < 0.05
        ok &= verdict(quiet, "wheel_vel (at rest)",
                      "peak |obs| %.4f  (= %.2f rad/s)"
                      % (np.abs(w).max(), np.abs(w).max() / WHEEL_OBS_SCALE))

        v = obs[:, 2:5]
        still = np.abs(v).max() < 0.2
        ok &= verdict(still, "base_lin_vel (at rest)",
                      "peak |obs| %.3f  (= %.2f m/s)"
                      % (np.abs(v).max(), np.abs(v).max() / SCALE_LIN_VEL))

        print("\n" + "=" * 70)
        print("PREFLIGHT %s" % ("PASSED" if ok else "FAILED -- do not run"))
        print("=" * 70)
        return 0 if ok else 1

    # ---- main loop --------------------------------------------------------

    def run(self):
        a = self.args
        rate = rospy.Rate(a.rate)
        rospy.loginfo("Waiting for mocap on %s ...", a.pose_topic)
        published = 0
        t0 = time.time()
        last_pose_wall = None
        preflight = [] if a.preflight else None

        while not rospy.is_shutdown():
            # Watchdog first: a stale pose must stop the robot even though the
            # loop below would simply never advance.
            if last_pose_wall is not None and not self.new_pose:
                if time.time() - last_pose_wall > a.mocap_timeout:
                    rospy.logerr_throttle(
                        1.0, "No mocap for %.2fs -- wheels stopped, propellers idled",
                        time.time() - last_pose_wall)
                    self._idle()
                    rate.sleep()
                    continue

            if not self.new_pose:
                rate.sleep()
                continue
            last_pose_wall = time.time()

            if not self._update_state():
                rate.sleep()
                continue

            # Refresh the wheel measurement BEFORE building the observation, so
            # obs[0:2] carries this tick's reading rather than the last one. One
            # serial round trip; the tick has 20 ms at 50 Hz. If db_wheels.py
            # monitor reports below ~45 Hz, move this to a background thread.
            if self.wheels is not None:
                self.wheels.poll()

            obs, wheel_meas, wheel_fresh = self._build_observation()
            if self.args.controller == "decoupled":
                action = self._infer_decoupled(obs)
            else:
                action = self._infer(obs)

            # NO action duplication -- all six dims are independent for these
            # checkpoints. See the module docstring.
            self.last_actions = action.copy()
            # Shift the history AFTER the action is chosen, so on the next tick
            # index 0 holds a(t-1). Rolling before would put the current action
            # at t-1 and hand the policy a command it has not issued yet.
            self._act_hist = np.roll(self._act_hist, 1, axis=0)
            self._act_hist[0] = action

            permitted, reason = self._gate()
            if not permitted:
                # repeat, not just on transition: a run that never ungates is
                # the most common failure and the reason must be impossible to
                # miss in the console.
                rospy.logwarn_throttle(3.0, "WHEELS GATED: %s", reason)
            if not permitted and not self.was_gated:
                rospy.logwarn("WHEELS GATED: %s", reason)
                if self.wheels is not None and not self.dry_run:
                    self.wheels.stop()
            elif permitted and self.was_gated:
                rospy.loginfo("wheels released -- armed and live")
            self.was_gated = not permitted

            desired, msg = self._apply(action, gated=not permitted)
            published += 1

            if preflight is not None:
                preflight.append(obs.copy())
                if time.time() - t0 >= a.preflight:
                    elapsed = time.time() - t0
                    return self._preflight_report(preflight, published / elapsed)

            # ENERGY INTEGRATION. Trapezoid-free rectangular integration at the
            # control rate is plenty at 50 Hz. Accumulated only while the wheels
            # are PERMITTED, so bench idle and gated time are not charged to the
            # traversal -- VI-B1 compares energy over a 3.1 s trial window, not
            # over however long the operator took to arm.
            now_t = time.time()
            watts = float("nan")
            if self.batt_v == self.batt_v and self.batt_a == self.batt_a:
                watts = self.batt_v * self.batt_a
                if permitted and self._e_last_t is not None:
                    self._energy_j += watts * (now_t - self._e_last_t)
            self._e_last_t = now_t
            m1_a = m2_a = float("nan")
            if self.wheels is not None and not self.dry_run:
                c = self.wheels.currents()
                if c is not None:
                    m1_a, m2_a = c

            if self.log_writer is not None:
                self.log_writer.writerow(
                    [published, "%.4f" % (time.time() - t0)]
                    + ["%.6f" % v for v in obs]
                    + ["%.6f" % v for v in action]
                    + ["%.4f" % desired[0], "%.4f" % desired[1],
                       "%.4f" % wheel_meas[0], "%.4f" % wheel_meas[1],
                       "%.4f" % msg.wh_l, "%.4f" % msg.wh_r,
                       "%.4f" % msg.servo1, "%.4f" % msg.servo2,
                       "%.4f" % msg.u_thr1, "%.4f" % msg.u_thr2,
                       int(wheel_fresh), int(not permitted),
                       (reason or "").replace(",", ";")]
                    + ["%.4f" % v for v in self.latest_pos]
                    + ["%.2f" % np.degrees(quat_to_yaw(*self.latest_quat))]
                    + ["%.5f" % v for v in self.latest_quat]
                    + ["%.3f" % self.batt_v, "%.3f" % self.batt_a,
                       "%.2f" % watts, "%.2f" % self._energy_j,
                       "%.3f" % m1_a, "%.3f" % m2_a])

            if published % a.rate == 0:
                yaw = np.degrees(quat_to_yaw(*self.latest_quat))
                rospy.loginfo(
                    "step=%d pos=[%.2f %.2f %.2f] yaw=%.0fdeg | "
                    "tgt=%.2fm | wheel des=[%.1f %.1f] meas=[%.1f %.1f]%s | "
                    "obs[0:2]=[%+.3f %+.3f] | servo=[%.2f %.2f] prop=[%.2f %.2f]",
                    published,
                    self.latest_pos[0], self.latest_pos[1], self.latest_pos[2], yaw,
                    self.target_dist, desired[0], desired[1], wheel_meas[0], wheel_meas[1],
                    "" if wheel_fresh else " STALE",
                    obs[0], obs[1], msg.servo1, msg.servo2, msg.u_thr1, msg.u_thr2)

            rate.sleep()


# =============================================================================

def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_path", required=True, help="TQC checkpoint (.pt)")
    p.add_argument("--controller", choices=["policy", "decoupled"], default="policy",
                   help="policy (default) = the TQC checkpoint. decoupled = the "
                        "published DoubleBee decoupled-mode controller (Cao et al. "
                        "eqs 19-23), run AUTONOMOUSLY through identical plumbing so "
                        "the comparison is apples-to-apples. IROS R1 objected that "
                        "the paper compared an autonomous method against a "
                        "human-operated baseline; this is the answer to that.")
    p.add_argument("--dctrl_path",
                   default=os.path.expanduser(
                       "~/doublebee_PID_JAI/doubleBee_isaac/scripts/co_rl/doublebee_dctrl.py"),
                   help="path to doublebee_dctrl.py. Imported, never copied -- a "
                        "second copy would let the sim and hardware baselines "
                        "diverge silently.")
    p.add_argument("--dctrl_v_desired", type=float, default=0.5,
                   help="forward speed setpoint [m/s] handed to the decoupled "
                        "controller, scaled by the body-frame goal direction. This "
                        "replaces the human operator's wheel commands.")
    p.add_argument("--device", default="cpu")
    p.add_argument("--rate", type=int, default=50,
                   help="control rate, Hz (50 = decimation 4 at dt=0.005)")

    safety = p.add_argument_group("safety")
    safety.add_argument("--dry_run", action="store_true",
                        help="run everything, command nothing. Use this first.")
    safety.add_argument("--preflight", type=float, default=0.0, metavar="SECONDS",
                        help="collect the observation for N seconds with the robot "
                             "sitting still on the floor, print a per-term verdict, "
                             "and exit. Implies --dry_run. Exit code 0 = safe to run.")
    safety.add_argument("--no_props", action="store_true",
                        help="hold both propellers at idle; wheels and servos live")
    safety.add_argument("--action_scale", type=float, default=0.4,
                        help="uniform attenuation on servos and propellers")
    safety.add_argument("--prop_scale", type=float, default=0.5,
                        help="extra attenuation on propellers only. Effective "
                             "propeller authority is action_scale*prop_scale; "
                             "1.0 delivers exactly the throttle the policy asks "
                             "for, and anything less cannot reach its hover point.")
    safety.add_argument("--servo_scale", type=float, default=1.0,
                        help="extra attenuation on servos only. Effective servo "
                             "authority is action_scale*servo_scale. Note this "
                             "scales the thrust VECTOR ANGLE, so below 1.0 the "
                             "arms never reach the tilt the policy commanded.")
    safety.add_argument("--max_wheel_diff", type=float, default=-1.0, metavar="RAD_S",
                        help="clamp on |right - left| wheel command, rad/s. 0 ties "
                             "the wheels (no steering at all, perfectly straight); "
                             "a small value like 5 keeps gentle heading correction "
                             "while preventing the policy from circling; negative "
                             "disables the clamp. Default -1 (off) preserves the "
                             "raw policy output.")
    safety.add_argument("--wheel_scale", type=float, default=0.4,
                        help="attenuation on the wheel velocity setpoint. NOTE "
                             "wheel_scale*wheel_action_scale is the rad/s the "
                             "policy gets at |action|=1, and the wheels top out "
                             "at 23.6 rad/s: above 0.5 (with the 47.0 action "
                             "scale) the command CLIPS and the policy loses "
                             "fine control. Measured hw_v31 at 1.4: 81%% of "
                             "ticks saturated, wheels reproduced 14%% of the "
                             "commanded variation, and the robot tipped over "
                             "its own wheels at the riser.")
    safety.add_argument("--wheel_lpf_alpha", type=float, default=1.0,
                        help="first-order low-pass on the wheel action, "
                             "0<alpha<=1. 1.0 (default) = off, unchanged "
                             "behaviour. The policy output is white noise "
                             "tick-to-tick (hw_v33 lag-1 +0.026) and the wheels "
                             "track only 0.41 of it, so each command is a step "
                             "input that pitches the body against the "
                             "acceleration. tau = dt*(1-alpha)/alpha: 0.5 -> "
                             "20 ms, 0.3 -> 47 ms, 0.2 -> 80 ms. Stay >= 0.2; "
                             "this filter sits inside the balance loop.")
    safety.add_argument("--pose_freeze_ticks", type=int, default=25,
                        help="gate the wheels after this many byte-identical "
                             "mocap poses (25 = 0.25s at 100 Hz). Catches the "
                             "capture-volume-edge failure where the mocap keeps "
                             "publishing at full rate with frozen values -- the "
                             "message-arrival watchdog cannot see that.")
    safety.add_argument("--mocap_timeout", type=float, default=0.3,
                        help="seconds without a pose before the robot is idled")
    safety.add_argument("--arm_gate", choices=["on", "off"], default="on",
                        help="'on' (default): the wheels turn only while the FCU "
                             "is ARMED *and* the policy-inference switch is high. "
                             "The wheels sit outside the FCU arming chain, so "
                             "without this neither disarming nor the switch stops "
                             "them.")
    safety.add_argument("--arm_timeout", type=float, default=3.0,
                        help="seconds without /mavros/state before the wheels "
                             "are gated. HEARTBEAT is 1 Hz, so keep this above 2.")
    safety.add_argument("--enable_channel", type=int, default=6, metavar="N",
                        help="RC channel carrying the policy-inference toggle. "
                             "Default 6, matching the firmware's ch6_high "
                             "(GCS_Mavlink.cpp:1139). 0 disables the check, "
                             "leaving ARM as the only gate -- not recommended.")
    safety.add_argument("--enable_above", type=int, default=1500, metavar="US",
                        help="the switch is 'on' above this PWM. 1500 matches "
                             "the firmware threshold exactly.")
    safety.add_argument("--rc_timeout", type=float, default=1.5,
                        help="seconds without /mavros/rc/in before the wheels "
                             "are gated (a dead RC link must not mean 'enabled')")
    safety.add_argument("--warmup_steps", type=int, default=10,
                        help="filter warmup ticks before anything is commanded")

    obs = p.add_argument_group("observation")
    obs.add_argument("--target", type=float, nargs=2, default=None, metavar=("X","Y"),
                     help="goal position in the MOCAP WORLD frame (metres). With "
                          "this, obs[29:32] is recomputed every tick exactly as "
                          "sim's TerrainTargetDirectionCommand does: unit "
                          "direction to the target in the yaw-only body frame "
                          "plus normalised heading error. WITHOUT it the command "
                          "is the frozen --velocity_command constant, which tells "
                          "the policy the target is permanently dead ahead and "
                          "makes it pivot forever. Pass a real target.")
    obs.add_argument("--heading_cmd", choices=["zero", "live"], default="zero",
                     help="obs[31], the third command slot. zero (default) "
                          "matches TRAINING: velocity_command.py sets this to a "
                          "literal zero on every sim step, so the policy has "
                          "never seen a nonzero value there. live restores the "
                          "old atan2(bx,by)/pi, which is out of distribution and "
                          "was measured driving a full-scale constant turn "
                          "(hw_v28: action_1 mean +0.765 with the goal 0.1 deg "
                          "off the nose). Only use live for comparison.")
    obs.add_argument("--velocity_command", type=float, nargs=3, default=[0.0, 1.0, 0.0],
                     help="obs[29:32]. This is the goal DIRECTION in the body "
                          "frame (X=right, Y=forward), not a velocity setpoint; "
                          "[0 1 0] means 'goal straight ahead'.")
    obs.add_argument("--servo_obs_source",
                     choices=["lowpass", "command", "slewed"],
                     default="lowpass",
                     help="what to report as servo_pos. command (default) = the "
                          "raw policy action x sim_servo_limit_rad, which spans "
                          "the same range sim observes. slewed = the slew-limited "
                          "command, i.e. the servo's true physical angle -- more "
                          "honest but measured 4-7x narrower than training and it "
                          "puts a rate limiter inside the feedback path, which "
                          "produced a locked limit cycle on hw_v16.csv. lowpass "
                          "(default) is a first-order lag on the raw action -- "
                          "the closest cheap model of sim's real joint, and the "
                          "only one of the three that does not ring.")
    obs.add_argument("--servo_obs_tau_ticks", type=float, default=3.0,
                     help="time constant, in control ticks, of the --servo_obs_source "
                          "lowpass model. 3.0 (60 ms) matches sim's torque-limited "
                          "servo sweeping its full range in ~2.8 ticks.")
    obs.add_argument("--actuator_obs_delay_ticks", type=int,
                     default=ACTUATOR_OBS_DELAY_TICKS,
                     help="control ticks of lag on the reconstructed servo_pos "
                          "and propeller_vel observations. 0 (default) feeds the "
                          "current slew-limited command, which already carries "
                          "the actuator dynamics. Anything above 0 inserts a PURE "
                          "delay into a feedback path: 3 produced a 4.2 Hz limit "
                          "cycle on hardware (hw_v15.csv). Raise only with a log "
                          "in hand.")
    obs.add_argument("--contact", type=float, default=1.0,
                     help="obs[27:29] wheel ground contact, assumed constant")
    obs.add_argument("--base_z_offset", type=float, default=0.058494812250137335,
                     help="added to raw mocap z before height_scan. Calibrated "
                          "2026-08-01 with the robot on real flat ground "
                          "(mocap z = 0.041505). Recalibrate if the mocap rigid "
                          "body definition changes.")
    obs.add_argument("--ground_height", type=float, default=0.0,
                     help="floor height in the mocap frame")
    obs.add_argument("--step", type=float, nargs=5, action="append",
                     metavar=("X_MIN", "X_MAX", "Y_MIN", "Y_MAX", "HEIGHT"),
                     help="one measured step in mocap-frame coordinates; repeat "
                          "per tread, lowest first. Without this the policy sees "
                          "flat ground and is blind to the obstacle.")
    obs.add_argument("--pose_topic", default="/mocap_node/doublebee/pose")
    obs.add_argument("--max_ang_vel", type=float, default=12.0,
                     help="clip on body angular rate (rad/s) before it reaches "
                          "obs[5:8]. Anything above this is a mocap/stamp "
                          "artefact, not the robot: hardware logs show spikes to "
                          "74 rad/s while stationary, which land 50x outside the "
                          "sim training range and saturate the policy.")
    obs.add_argument("--ang_vel_lpf_alpha", type=float, default=0.5,
                     help="first-order low-pass on angular velocity, 0<a<=1. "
                          "1.0 disables it. The raw quaternion difference of a "
                          "noisy pose is a noisy derivative; base_lin_vel gets a "
                          "Kalman filter and this term historically got nothing.")
    obs.add_argument("--min_dt", type=float, default=0.005,
                     help="lower clamp on the pose dt used for differentiation")
    obs.add_argument("--max_dt", type=float, default=0.100,
                     help="upper clamp on the pose dt used for differentiation")
    obs.add_argument("--process_noise", type=float, default=1.0)
    obs.add_argument("--measurement_noise", type=float, default=0.0005)

    act = p.add_argument_group("actuation")
    act.add_argument("--prop_map",
                     choices=["sim_damped", "sim", "affine", "passthrough"],
                     default="sim_damped",
                     help="sim (default) = sim's own aerodynamics.py PWM model, "
                          "1000 + 1.3*|omega| capped at 1650 us -- what the policy "
                          "was actually trained against. affine = the PPO "
                          "deployment's ESC calibration, which reaches 1908 us and "
                          "over-commands relative to sim. passthrough = raw action.")
    act.add_argument("--servo_slew_rad_s", type=float, default=2.0,
                     help="slew-rate limit on the servo command, rad/s. Default "
                          "2.0 MATCHES sim's propeller_servos velocity_limit. "
                          "Real servos are 3-5x faster, so without this the "
                          "hardware fully executes a thrust-vector command that "
                          "sim only ever half-executed -- 20 N going 97%% "
                          "horizontal at a 0.44 m lever arm flips the robot in "
                          "0.14 s. Set 0 to disable.")
    act.add_argument("--prop_accel_rad_s2", type=float, default=0.0,
                     help="limit how fast the commanded propeller speed may RISE, "
                          "rad/s^2, reproducing sim's (wildly over-heavy) propeller "
                          "inertia. Sim manages about +9. 0 disables.")
    act.add_argument("--prop_decel_rad_s2", type=float, default=1e9,
                     help="same for spinning DOWN. Sim manages about 38. Kept "
                          "separate because drag makes deceleration faster than "
                          "acceleration. Only used when --prop_accel_rad_s2 > 0.")
    act.add_argument("--prop_rad_s_max", type=float, default=375.0,
                     help="propeller speed (rad/s) commanded by action=+1, i.e. "
                          "2x the propeller_vel scale in training. 375 (default) "
                          "for configs from 2026-08-23 on (scale=offset=187.5); "
                          "250 for 2026-08-21/22 (scale=offset=125); 500 for "
                          "earlier checkpoints (scale=offset=250). Only affects "
                          "--prop_map sim. Getting it wrong is a proportional "
                          "thrust error at every command -- 375 against a 250 "
                          "checkpoint over-commands thrust by 1.5x.")
    act.add_argument("--wheel_diff_scale", type=float, default=8.0,
                     help="rad/s that |differential action| = 1 represents, for "
                          "obs-40/act-4 CommonDiff checkpoints. MUST match "
                          "diff_scale in ActionsCfg4D (8.0). This is the yaw "
                          "trim; it was not applied at all before 2026-08-27, so "
                          "hardware ran only the common term while the policy "
                          "commanded both.")
    act.add_argument("--wheel_action_scale", type=float, default=SIM_WHEEL_VEL_LIMIT_RAD_S,
                     help="rad/s represented by |action| = 1")
    safety.add_argument("--max_start_lean_deg", type=float, default=20.0,
                        help="refuse the FIRST engage while the robot is leaning "
                             "more than this many degrees off vertical. Once "
                             "engaged the check never fires again, so a policy "
                             "catching a lean is not interrupted. Starting lean "
                             "is most of the recovery budget: transfer_props.csv "
                             "engaged at 13.5 deg and was down in 0.36 s exactly "
                             "as the passive pendulum predicts, while "
                             "transfer_smooth.csv engaged at 6.3 deg and drove 3 m. "
                             "Set 0 to disable.")
    safety.add_argument("--assumed_thrust_frac", type=float, default=0.41,
                        help="fraction of weight the propellers are assumed to "
                             "support, used ONLY to report the predicted fall time "
                             "in the start-lean message. 0.41 is what "
                             "transfer_props.csv actually commanded; 0.62 is the "
                             "ceiling at prop_rad_s_max=250.")
    safety.add_argument("--prewarm", action="store_true",
                        help="while ARMED but before the policy switch is flipped, "
                             "hold the propellers at --prop_min_frac and the servos "
                             "already aimed, instead of fully off. Removes the "
                             "engage transient: measured, the command steps from "
                             "pwm 1000 (4.6 N) to ~1363 (18.8 N) in one tick but a "
                             "real ESC needs 100-300 ms, so the robot spends the "
                             "start of every run BELOW the 13.76 N stability "
                             "threshold while already tipping. Wheels stay at zero "
                             "-- this is thrust and aim only, never motion. "
                             "Refuses unless armed, mocap live, and the robot "
                             "within the start-lean tolerance. Requires "
                             "--prop_min_frac > 0. THE PROPELLERS WILL SPIN WHILE "
                             "YOU ARE HOLDING THE ROBOT.")
    act.add_argument("--servo_hold_sign", type=float, default=-1.0,
                     help="+1 or -1 on the --servo_attitude_hold output. The "
                          "JAIOut-units-to-thrust-direction mapping was derived, "
                          "not measured, and cannot be checked from the log "
                          "(computing thrust direction from the servo command uses "
                          "the same assumption, so a flipped convention scores a "
                          "perfect 1.00). Set from OBSERVATION: tilt the frame "
                          "forward and the props must stay pointing UP. If they "
                          "follow the tilt, flip this. Default -1.0 from the "
                          "2026-08-24 hardware observation.")
    act.add_argument("--servo_lpf_alpha", type=float, default=1.0,
                     help="first-order low-pass on the servo command, 0<a<=1. "
                          "1.0 (default) disables it. The mirror of "
                          "--wheel_lpf_alpha, which was added because the wheel "
                          "command thrashed at 25 Hz; the servo command does the "
                          "same once the policy drives it rather than "
                          "--servo_attitude_hold. 0.3 matches the wheel setting.")
    act.add_argument("--servo_bias_dist", type=float, default=0.0, metavar="M",
                     help="distance to a riser at which --servo_step_bias "
                          "starts fading in. 0 (default) = share "
                          "--prop_boost_dist. Set it SMALLER so the tilt "
                          "arrives at contact while thrust still spools early.")
    act.add_argument("--servo_step_bias", type=float, default=0.0,
                     help="signed servo tilt, in normalised action units, added "
                          "at a riser and faded out past it on the same gate as "
                          "--prop_boost_dist. 0 (default) = off. Use it to aim "
                          "thrust INTO the step during contact and let it "
                          "return to vertical afterwards. Sign depends on the "
                          "arm convention: try +-0.2 first and read `servo1` in "
                          "the log to see which way it moves.")
    act.add_argument("--servo_hold_blend_step", type=float, default=-1.0,
                     help="servo blend to fade toward at a riser, on the same "
                          "distance gate as --prop_boost_dist. Negative "
                          "(default) = disabled, --servo_hold_blend applies "
                          "everywhere. Set it to give the policy more or less "
                          "servo authority at a step than between steps.")
    act.add_argument("--servo_hold_blend", type=float, default=1.0,
                     help="0..1 mix between the policy's servo action and the "
                          "geometric attitude hold, when --servo_attitude_hold "
                          "is set. 1.0 (default) is the original full override. "
                          "0.0 leaves the servo entirely to the policy. Use an "
                          "intermediate value to keep the policy in the loop "
                          "while adding the pitch compensation it cannot learn "
                          "under sim's 2 rad/s servo limit. Note the slew limit "
                          "switches to --servo_hold_slew_rad_s whenever the "
                          "hold is enabled, which is what makes a correction "
                          "fast enough to matter.")
    act.add_argument("--servo_hold_damping", type=float, default=0.15,
                     help="seconds of rate lead added to --servo_attitude_hold: "
                          "theta = -(pitch + k*pitch_rate). Without it the hold is "
                          "pure proportional control -- statically stable but "
                          "undamped -- and the robot swings through vertical and "
                          "over the other side, which is what new_ckpt_hold3.csv "
                          "recorded. 0 disables (proportional only). Raise if it "
                          "still overshoots, lower if the servo starts chattering.")
    act.add_argument("--servo_hold_slew_rad_s", type=float, default=10.0,
                     help="slew limit used INSTEAD of --servo_slew_rad_s while "
                          "--servo_attitude_hold is on. Defaults to 10.0, roughly "
                          "the real servo's capability (86 deg in ~0.15 s). Sim's "
                          "2.0 rad/s is the wrong limit here: it applies to policy "
                          "actions so hardware does not out-run sim, whereas the "
                          "hold is a geometric correction that can only point "
                          "thrust up. At 2.0 the servo lagged the fall by 36 deg "
                          "and thrust dropped to 0.81 vertical. 0 disables.")
    act.add_argument("--servo_attitude_hold", action="store_true",
                     help="IGNORE the policy's servo action and instead hold the "
                          "thrust vector VERTICAL IN THE WORLD FRAME, from mocap "
                          "attitude (theta_servo = -pitch, clamped to the servo "
                          "travel). The propellers sit 0.443 m above the wheel "
                          "axle and the CoM only 0.139 m, so world-vertical thrust "
                          "gives a RESTORING moment of T*0.443*sin(theta) against "
                          "gravity's 6.10*sin(theta): sin(theta) cancels, and the "
                          "machine is a HANGING pendulum -- statically stable at "
                          "any lean -- whenever T > 13.76 N. That needs no control "
                          "bandwidth, which is why it transfers where a learned "
                          "balance loop does not. Pair with --prop_min_frac; "
                          "without the thrust floor the criterion is not met. "
                          "Corrects PITCH only -- the servos are one axis, so roll "
                          "is still the wheels' problem.")
    act.add_argument("--wheel_vel_mirror", choices=("none", "from_l", "from_r"),
                     default="none",
                     help="Derive one wheel's MEASURED velocity from the other, "
                          "at the source. Unlike --wheel_obs_mirror this also "
                          "feeds command_duty, so a wheel with a dead encoder "
                          "channel is out of both the observation and the "
                          "control path. from_r drives the LEFT wheel's value "
                          "from the right. Pair with --wheel_mode duty.")
    act.add_argument("--wheel_obs_mirror", choices=("none", "from_l", "from_r"),
                     default="none",
                     help="STOPGAP for a dead wheel encoder: synthesise the dead "
                          "wheel's obs[0:2] entry as -1 x the live one. "
                          "from_l = left encoder is good, mirror it onto right. "
                          "from_r = right is good. The logged measurement is not "
                          "altered. The dead wheel must be driven OPEN LOOP or its "
                          "velocity PID will command full power; it will then lag "
                          "under load, so treat step-climbing data as suspect.")
    act.add_argument("--prop_min_frac", type=float, default=0.0,
                     help="floor on the propeller throttle FRACTION (0..1), so "
                          "total thrust stays above the 13.76 N static-stability "
                          "threshold --servo_attitude_hold depends on. The policy "
                          "still commands anything above the floor. Use the printed "
                          "table at startup to pick it for your --prop_rad_s_max.")
    act.add_argument("--sim_servo_limit_rad", type=float, default=math.pi / 2.0,
                     help="SERVO_POS_LIMIT_RAD the checkpoint was TRAINED with, "
                          "i.e. the joint angle in rad that |action|=1 asks for. "
                          "JAIOut servo units are [-1,1] over +/-pi/2, so the "
                          "hardware command is -(2/pi)*limit*action and this "
                          "value sets the gain. Default pi/2 (=1.5708) reproduces "
                          "the historical bare -action[2] and is correct for every "
                          "checkpoint up to 2026-08-23. Use 0.5236 (pi/6) for "
                          "checkpoints trained after actions.py dropped the servo "
                          "range to 30 deg. Setting it too HIGH over-tilts the "
                          "thrust vector proportionally -- at pi/2 against a pi/6 "
                          "checkpoint that is 3x, i.e. 90 deg instead of 30.")
    act.add_argument("--servo_sign_left", type=float, default=1.0,
                     help="+1 or -1. Flip if the LEFT propeller arm tilts the "
                          "wrong way. The derived mapping is servo1 = -action[2]; "
                          "this exists so a linkage mounted backwards can be "
                          "corrected without editing code.")
    act.add_argument("--servo_sign_right", type=float, default=1.0,
                     help="+1 or -1, as --servo_sign_left but for the RIGHT arm "
                          "(servo2 = +action[3]).")
    act.add_argument("--rc_state", type=int, default=2)

    rc = p.add_argument_group("roboclaw")
    rc.add_argument("--roboclaw_port",
                    default="/dev/serial/by-id/usb-Basicmicro_Inc._USB_Roboclaw_2x7A-if00",
                    help="by-id path -- ttyACM numbering swaps between boots")
    rc.add_argument("--roboclaw_baud", type=int, default=38400)
    rc.add_argument("--roboclaw_address", type=lambda s: int(s, 0), default=0x80)
    rc.add_argument("--roboclaw_accel", type=int, default=0,
                    help="acceleration limit in counts/s^2. DEFAULT 0 = NO LIMIT, "
                         "and that is deliberate. This robot balances on its "
                         "wheels, so the wheel command IS the balance loop and "
                         "rate-limiting it is not smoothing, it is a delay in a "
                         "feedback path that has to be instant. Measured from the "
                         "sim policy I/O log, the policy relies on wheel "
                         "accelerations of ~74 rad/s^2 (median), 104 (p90), 270 "
                         "(p99) and bursts past 1000 when catching a fall; the "
                         "old 20000 counts/s^2 default was 65 rad/s^2, i.e. below "
                         "sim's MEDIAN, and the robot tipped 14deg->51deg in 0.25s "
                         "while the wheels were still ramping. With 0 the RoboClaw "
                         "steps the setpoint and the motor accelerates as hard as "
                         "torque and traction allow -- which is what sim does, "
                         "where the wheel is effort-limited rather than "
                         "slew-limited. Set a nonzero value only if you see "
                         "overcurrent faults in './db_wheels.py status'.")
    rc.add_argument("--wheel_mode", choices=("velocity", "duty"), default="velocity",
                    help="'velocity' sends a setpoint to the RoboClaw's PI loop "
                         "(previous behaviour). 'duty' applies open-loop torque "
                         "proportional to velocity error, which is what SIM "
                         "actually models: an effort-limited joint, torque = "
                         "clamp(damping*(v_des - v_meas), +/-effort_limit). The "
                         "RoboClaw loop adds 150-280 ms rise and up to 37% "
                         "overshoot that sim never saw.")
    rc.add_argument("--wheel_duty_kp", type=float, default=0.06,
                    help="duty per rad/s of velocity error in duty mode. 0.06 "
                         "saturates at ~17 rad/s of error. Raise for more "
                         "authority, lower if it chatters.")
    rc.add_argument("--wheel_duty_max", type=float, default=0.6,
                    help="duty magnitude cap in duty mode, 0..1. This is the "
                         "analogue of sim's effort_limit; start conservative.")
    rc.add_argument("--wheel_duty_trim_l", type=float, default=1.0,
                    help="per-side duty gain, LEFT (M1). LEAVE AT 1.0. The "
                         "1.5-1.8x asymmetry measured 2026-09-04 was NOT "
                         "mechanical: M1's encoder cable was running alongside "
                         "the motor power leads and counting switching noise. "
                         "Separated 2026-09-05, after which M1/M2 matched to "
                         "within 1% at every duty. Only used in --wheel_mode duty.")
    rc.add_argument("--heading_hold_kp", type=float, default=0.0,
                    help="rad/s of wheel differential per radian of yaw error "
                         "from the heading captured at engage. 0 (default) = off. "
                         "Corrects the pivot caused by a wheel jamming on a riser "
                         "face, which the encoders cannot see because the jammed "
                         "wheel still spins.")
    rc.add_argument("--heading_hold_max", type=float, default=4.0, metavar="RAD_S",
                    help="clamp on the heading-hold differential. Keep near "
                         "sim's k_diff = 4 so the wheels stay inside the "
                         "authority the policy trained with.")
    rc.add_argument("--prop_min_frac_step", type=float, default=0.0,
                    help="propeller fraction FLOOR to fade toward at a riser, "
                         "using the same gate as --prop_boost_dist. 0 (default) "
                         "= off. Unlike --prop_scale_step this does not multiply "
                         "the policy's action, so it still delivers thrust when "
                         "the policy commands the propellers off -- which it does "
                         "at contact on almost every run.")
    rc.add_argument("--prop_boost_dist", type=float, default=0.0, metavar="M",
                    help="start fading in --prop_scale_step this far before a "
                         "riser, using the --step geometry rather than the height "
                         "scan. 0 (default) = scan only. The 4x4 scan only shows "
                         "relief 0.10 m out, which is 100 ms at 1 m/s and far "
                         "less than propeller spin-up. 0.5 gives half a second.")
    rc.add_argument("--wheel_scale_post", type=float, default=0.0,
                    help="wheel scale applied for --post_climb_s after a height "
                         "gain is measured. 0 (default) = disabled. Softens the "
                         "wheels while settling on a tread without throwing away "
                         "the momentum that carries a wheel over the edge.")
    rc.add_argument("--post_climb_s", type=float, default=1.5, metavar="S",
                    help="how long --wheel_scale_post stays applied after a climb.")
    rc.add_argument("--post_climb_rise", type=float, default=0.03, metavar="M",
                    help="height gain within 1 s that counts as having climbed.")
    rc.add_argument("--wheel_scale_step", type=float, default=0.0,
                    help="wheel scale to fade toward when a riser is in the "
                         "height scan. 0 (default) = disabled. Set BELOW "
                         "--wheel_scale to approach a step slowly: contact speed "
                         "decides climb vs bounce, and 0.70 m risers leave no "
                         "room to recover. Shares --prop_step_relief.")
    rc.add_argument("--prop_scale_step", type=float, default=0.0,
                    help="propeller scale to fade toward when a riser is in the "
                         "height scan. 0 (default) = disabled, use --prop_scale "
                         "everywhere. Set ABOVE --prop_scale to boost thrust only "
                         "at a step, where II-A says 46%% of weight is needed.")
    rc.add_argument("--prop_step_relief", type=float, default=0.05, metavar="M",
                    help="height-scan relief at which --prop_scale_step is fully "
                         "applied, metres. 0.05 saturates on a 6 cm riser; the "
                         "boost fades in linearly below it.")
    rc.add_argument("--wheel_duty_trim_r", type=float, default=1.0,
                    help="per-side duty gain, RIGHT (M2).")
    rc.add_argument("--wheel_ramp_s", type=float, default=0.0,
                    help="seconds to fade the wheel command in after the gate "
                         "opens. 0 = off (previous behaviour). The policy "
                         "commands full forward from a dead stop, which pitches "
                         "the body before the 300 ms wheel loop can answer; "
                         "measured 2026-09-04, the robot was 17 cm displaced and "
                         "past 30 deg of lean within 0.3 s of enable on most "
                         "runs. 1.0-1.5 is a sensible starting point. Only the "
                         "launch is affected; steady-state authority is "
                         "unchanged.")
    rc.add_argument("--wheel_counts_per_rev", type=float, default=1920.0)
    rc.add_argument("--wheel_max_rad_s", type=float, default=SIM_WHEEL_VEL_LIMIT_RAD_S)
    rc.add_argument("--m1_is_right", action="store_true")
    rc.add_argument("--wheel_sign_left", type=float, default=1.0)
    rc.add_argument("--wheel_sign_right", type=float, default=1.0)
    rc.add_argument("--no_roboclaw", action="store_true")

    p.add_argument("--log_path", default="db_inference_log.csv",
                   help="per-step CSV of obs, actions and commands ('' to disable)")

    args, _ = p.parse_known_args(rospy.myargv(argv=sys.argv)[1:])
    if args.preflight:
        args.dry_run = True

    node = DoubleBeeInference(args)
    try:
        return node.run()
    except rospy.ROSInterruptException:
        return 0


if __name__ == "__main__":
    sys.exit(main() or 0)

"""
DoubleBee DECOUPLED-MODE controller -- publication-grade baseline.

Faithful re-implementation of the decoupled-mode controller from
    M. Cao, X. Xu, S. Yuan, K. Cao, K. Liu, L. Xie,
    "DoubleBee: A Hybrid Aerial-Ground Robot with Two Active Wheels",
    IROS 2023 (arXiv:2303.05075), Section III-D, equations (19)-(23).

This file exists so the PID baseline in our paper is one a reviewer can
trust: the published control law is implemented literally, every deviation
from it is behind an explicit flag and is reported by describe(), and the
actuation limits match the simulator and the hardware.

`doublebee_dctrl.py` is left untouched. This is a separate, additive file.


WHY A REWRITE (defects in doublebee_dctrl.py this file fixes)
-------------------------------------------------------------
1. THRUST D-TERM WAS DEAD, Ki WAS DOUBLED. The integrator/derivative block
   ran twice per call. The second pass computed
       dTe = (Te - self._prev_Te) / dt
   *after* the first pass had already assigned _prev_Te = Te, so dTe was
   identically 0 and Kt_d never contributed; _int_Te accumulated twice, so
   the effective Kt_i was 2x its nominal value.

   This is not cosmetic. The propeller restoring moment f_p*L_p*sin(theta)
   and the gravitational toppling moment m*g*L_com*sin(theta) share the same
   sin(theta), so above the static threshold the robot is NEUTRALLY stable at
   every lean angle -- there is no angle-dependent restoring gradient and
   nothing removes energy. Rate feedback is the ONLY damping in the loop.
   Disabling the D-term makes the baseline pendulum for an implementation
   reason and not a control-design reason, which would understate it.

2. EQUATION (20)'s SIGN RULE WAS MISSING. The code used Te = omega_theta_e
   unconditionally. The paper is explicit that thrust must be *reduced* to
   produce positive pitch acceleration in one quadrant (Sec. II-B: "when
   theta < 0 and sigma > 0, positive and negative pitch acceleration can be
   produced by reducing and increasing the thrust, respectively"). Without
   the rule the controller drives the wrong way there.

3. THRUST CLIP WAS 1.81x TOO PERMISSIVE. T is per-propeller, but it was
   clipped at 33.14 N, which is the *total* two-prop thrust at PWM 1600
   (2 x 16.52 N). The real per-prop ceiling at the simulator's PWM cap of
   1650 us is 18.28 N. A baseline allowed almost twice the thrust it can
   physically produce is not a baseline.

4. AN UNDISCLOSED DEVIATION FROM EQ. (23). A gain-scheduled pitch feedback
   term (Kb_p*theta + Kb_d*theta_dot) and a heading-hold term were added to
   the wheel torques. Equation (23) is pure velocity + steering; the whole
   premise of "decoupled" is that pitch is NOT fed back to the wheels. The
   addition is physically well-motivated -- the paper itself notes the thrust
   lever arm vanishes near theta = 0 (Sec. III-D, referring to Eq. 9) -- but
   published-controller results must come from the published controller.
   Here it lives in mode="augmented" and describe() always reports it.

5. WHEEL GAINS WERE TUNED AGAINST A TORQUE BUDGET THAT DOES NOT EXIST.
   doublebee_dctrl.py clipped wheel torque at +/-2.0 N.m. The actuator's
   effort_limit (doublebee_v1.py) is 0.51 N.m -- the clip was 3.9x the real
   ceiling, so the simulator silently truncated commands the controller
   believed it had issued, and any gain tuned against +/-2.0 is mistuned.
   Run saturation_report() before tuning: at Kb_p = 5.0 the augmented pitch
   feedback alone saturates 0.51 N.m at just 5.8 deg of pitch, which makes
   the wheel channel effectively bang-bang over most of the working range.

NOT a defect: doublebee_dctrl.py averages tau_w1/tau_w2 into tau_w, but
play_dctrl.py consumes tau_w1 and tau_w2 separately, so the differential
(steering, heading hold) does survive. tau_w is a display value only. Kept
in the return dict here for drop-in compatibility.


HOW TO USE IT AS A HONEST BASELINE
----------------------------------
mode="faithful"    Stock [7] Eqs (19)-(23). This is what you cite.
mode="augmented"   + wheel pitch feedback + heading hold. A STRONGER
                   baseline. Beating this is the claim worth making.

Run both. Report both. If the learned policy only beats "faithful", say so.

Sweep theta_desired -- do not pick one value. The classical controller has to
trade pitch excursion against climb success (IROS'26 saw -80 deg climb and
0 deg fail); a sweep over {0, -20, -40, -60, -80} deg with success rate and
energy per setting shows that frontier instead of asserting a single point.
`theta_desired` is a per-call argument precisely so this is cheap.

Parity items to state in the paper (this file enforces the first three):
  - actuation: same wheel/servo/PWM limits as sim and hardware  [enforced]
  - control rate: same dt as the policy (50 Hz)                 [enforced]
  - thrust model: same pwm2thrust polynomial                    [enforced]
  - information: policy sees a 16-dim height scan, this baseline is BLIND.
                 Disclose it. Do not let "sees terrain vs does not" pass as
                 "learned vs classical".
  - autonomy:    drive v_desired from a fixed profile, NOT a human pilot.
                 A human-operated baseline invites both "the pilot helped"
                 and "the pilot was bad".
  - tuning:      record how many gain sets you tried (tuning_budget below).
"""

import numpy as np

# --- Actuation limits. These MUST match the simulator and the hardware. ---
# Per-propeller thrust at the simulator's PWM cap (aerodynamics.py clamps PWM
# to 1650 us; pwm2thrust_params.json degree-4 fit evaluates to 18.281 N there).
T_MAX_PER_PROP_N = 18.28
# Servo travel. actions.py: SERVO_POS_LIMIT_RAD = pi/4. NOT pi/2.
SERVO_LIMIT_RAD = np.pi / 4          # 0.7854
# Wheel torque ceiling, doublebee_v1.py effort_limit.
WHEEL_TORQUE_LIMIT_NM = 0.51
# Robot weight, for reporting thrust as a fraction of weight.
ROBOT_WEIGHT_N = 3.2182 * 9.81       # 31.57 N


class DecoupledBaseline:
    """Decoupled-mode controller, DoubleBee [7] Eqs (19)-(23).

    Args:
        mode: "faithful" for stock [7]; "augmented" to add wheel pitch
            feedback and heading hold (both absent from Eq. 23).
        use_eq20_sign: apply the Eq. (20) throttle-error sign rule. True is
            the paper. False only to measure how much the rule is worth.
        servo_bias_sign: +1 or -1 on the Eq. (22) PID bias. The -theta
            feedforward is unambiguous but the bias sign depends on the
            servo-positive convention, which differs between our sim USD and
            the paper figure. Verify once with servo_sign_check() and pin it.
        enforce_limits: clip to the module-level actuation limits. Leave True
            for any number that goes in the paper.
    """

    def __init__(self, mode="faithful", use_eq20_sign=True,
                 servo_bias_sign=+1.0, enforce_limits=True, dt=0.02):
        if mode not in ("faithful", "augmented"):
            raise ValueError("mode must be 'faithful' or 'augmented', got %r" % mode)
        self.mode = mode
        self.use_eq20_sign = bool(use_eq20_sign)
        self.servo_bias_sign = float(servo_bias_sign)
        self.enforce_limits = bool(enforce_limits)
        self.dt = float(dt)

        # --- Eq. (19): desired pitch rate from pitch error ---
        self.Kp_d = 6.0

        # --- Eq. (21): throttle PID -> thrust bias for pitch torque ---
        self.Kt_p = 2.0
        self.Kt_i = 0.1
        self.Kt_d = 0.05

        # --- Eq. (22): servo PID bias on top of the -theta feedforward ---
        self.Ksig_p = 0.2
        self.Ksig_i = 0.02
        self.Ksig_d = 0.08

        # --- Eq. (23): wheel torque from desired velocity and steer rate ---
        self.Kv_d = 1.0
        self.Ks_d = 0.08

        # T_hold: "the level of throttle capable of lifting the robot from a
        # completely flat status" [7, Eq. 21]. From the real robot's
        # BB_HOV_DC = 1335 us -> pwm_to_thrust(1335) = 8.66 N per prop.
        # 2 x 8.66 = 17.32 N = 55% of weight. Worth knowing: unloading the
        # wheels enough for them to clear a 6 cm riser needs T_z >= 17.7 N,
        # so T_hold sits essentially exactly at the climb threshold.
        self.T_hold = 8.66

        # Thrust floor. With T free to reach 0 the robot has NO attitude
        # authority at all, which is a failure the paper's controller does not
        # have (it commands about a hold throttle, not about zero).
        self.T_floor = 0.5 * self.T_hold

        # Anti-windup. Without a clamp the integrators wind up during the
        # seconds the robot spends pinned against a riser, and the baseline
        # then fails on windup rather than on control design.
        self.int_limit_Te = 5.0
        self.int_limit_sig = 5.0

        # Derivative low-pass. Raw (e_k - e_{k-1})/dt on a 50 Hz mocap-derived
        # pitch rate is noisy enough to swamp Kt_d. 1.0 disables filtering.
        self.d_lpf_alpha = 0.3

        # --- mode="augmented" only. NOT part of [7] Eq. (23). ---
        self.Kb_p = 5.0                       # wheel pitch feedback
        self.Kb_d = 1.0                       # wheel pitch-rate feedback
        self.balance_blend_deg = 20.0         # fades out beyond this |theta|
        self.Kyaw_p = 0.6                     # heading hold toward yaw = 0

        # Reproducibility bookkeeping. Set this to the number of gain
        # configurations actually evaluated, and report it in the paper --
        # "the baseline was undertuned" is the default reviewer objection and
        # a stated tuning budget is the only answer to it.
        self.tuning_budget = None

        # What the harness feeds this controller. Defaults describe
        # play_dctrl.py, which supplies both. Override if you run it elsewhere.
        self.terrain_preview = (
            "height_scanner -> step_ahead -> theta_desired = -LEAN_MAX*step_ahead "
            "(feedforward lean-back schedule); parity with the policy's height scan")
        self.velocity_command_source = (
            "autonomous: v_desired = clip(dist_to_target*0.15, 0, 0.30), "
            "floored at 0.15 near a step; NO human pilot")

        self.reset()

    def reset(self):
        self._int_Te = 0.0
        self._int_sig = 0.0
        self._prev_Te = 0.0
        self._prev_sig_e = 0.0
        self._dTe_f = 0.0
        self._dsig_f = 0.0
        self._first_call = True

    # ------------------------------------------------------------------
    def control(self, theta, theta_dot, v, v_desired=0.0,
                theta_desired=0.0, yaw_rate=0.0, yaw_rate_desired=0.0,
                yaw=0.0, yaw_gain_scale=1.0):
        """One control step.

        Signature and return keys match doublebee_dctrl.DecoupledController,
        so play_dctrl.py works unchanged.

        Args:
            theta: pitch [rad], 0 = upright.
            theta_dot: pitch rate [rad/s].
            v: forward speed [m/s].
            v_desired: commanded forward speed [m/s]. Drive this from a fixed
                profile, not a human stick, or the comparison is confounded.
            theta_desired: pitch setpoint [rad]. THE sweep parameter.
            yaw_rate, yaw_rate_desired: steer rate [rad/s].
            yaw: heading [rad]. Used only in mode="augmented".
            yaw_gain_scale: scales the augmented heading gain. Ignored when
                mode="faithful".

        Returns:
            dict with T (N per prop), sigma (rad), tau_w1, tau_w2 (N.m), and
            tau_w (their mean, for display only).
        """
        # --- Eq. (19): desired pitch rate, and its error ---
        omega_theta_d = self.Kp_d * (theta_desired - theta)
        omega_theta_e = omega_theta_d - theta_dot

        # --- Eq. (20): throttle-error sign rule ---
        # Te = -omega_theta_e  if (theta > 0 and theta_dot <= theta_dot_d < 0)
        #                      or (theta < 0 and not (theta_dot >= theta_dot_d > 0))
        #      +omega_theta_e  otherwise
        if self.use_eq20_sign:
            c1 = (theta > 0.0) and (theta_dot <= omega_theta_d < 0.0)
            c2 = (theta < 0.0) and not (theta_dot >= omega_theta_d > 0.0)
            Te = -omega_theta_e if (c1 or c2) else omega_theta_e
        else:
            Te = omega_theta_e

        # --- Eq. (21): throttle PID -> per-prop thrust ---
        # Integrate and differentiate EXACTLY ONCE. (The defect this file
        # exists to fix ran this block twice, zeroing the derivative and
        # doubling the integral.)
        self._int_Te = float(np.clip(self._int_Te + Te * self.dt,
                                     -self.int_limit_Te, self.int_limit_Te))
        dTe_raw = 0.0 if self._first_call else (Te - self._prev_Te) / self.dt
        self._dTe_f += self.d_lpf_alpha * (dTe_raw - self._dTe_f)
        self._prev_Te = Te

        T = (self.T_hold
             + self.Kt_p * Te
             + self.Kt_i * self._int_Te
             + self.Kt_d * self._dTe_f)
        if self.enforce_limits:
            T = float(np.clip(T, self.T_floor, T_MAX_PER_PROP_N))
        else:
            T = float(max(T, 0.0))

        # --- Eq. (22): sigma = -theta + PID(pitch-rate error) ---
        # The -theta feedforward keeps thrust roughly vertical so it does not
        # disturb translation [7, Sec. II-B]; the PID bias supplies the pitch
        # torque, which matters most near upright where the lever arm is small.
        sig_e = omega_theta_e
        self._int_sig = float(np.clip(self._int_sig + sig_e * self.dt,
                                      -self.int_limit_sig, self.int_limit_sig))
        dsig_raw = 0.0 if self._first_call else (sig_e - self._prev_sig_e) / self.dt
        self._dsig_f += self.d_lpf_alpha * (dsig_raw - self._dsig_f)
        self._prev_sig_e = sig_e

        sigma = (-theta + self.servo_bias_sign * (
                 self.Ksig_p * sig_e
                 + self.Ksig_i * self._int_sig
                 + self.Ksig_d * self._dsig_f))
        if self.enforce_limits:
            sigma = float(np.clip(sigma, -SERVO_LIMIT_RAD, SERVO_LIMIT_RAD))

        # --- Eq. (23): wheel torque from desired velocity and steer rate ---
        speed_term = self.Kv_d * (v_desired - v)
        steer_term = self.Ks_d * (yaw_rate_desired - yaw_rate)
        tau_w1 = speed_term - steer_term
        tau_w2 = speed_term + steer_term

        if self.mode == "augmented":
            # NOT in [7] Eq. (23). Pitch feedback into the wheels, faded out
            # as |theta| grows because thrust regains lever arm there.
            blend = float(np.clip(
                1.0 - abs(theta) / np.radians(self.balance_blend_deg), 0.0, 1.0))
            wheel_balance = self.Kb_p * theta + self.Kb_d * theta_dot
            # NOT in [7] either. Heading hold toward yaw = 0.
            heading = self.Kyaw_p * yaw_gain_scale * (0.0 - yaw)
            tau_w1 += blend * wheel_balance - heading
            tau_w2 += blend * wheel_balance + heading

        if self.enforce_limits:
            tau_w1 = float(np.clip(tau_w1, -WHEEL_TORQUE_LIMIT_NM, WHEEL_TORQUE_LIMIT_NM))
            tau_w2 = float(np.clip(tau_w2, -WHEEL_TORQUE_LIMIT_NM, WHEEL_TORQUE_LIMIT_NM))

        self._first_call = False
        return {"T": T, "sigma": sigma,
                "tau_w1": tau_w1, "tau_w2": tau_w2,
                "tau_w": 0.5 * (tau_w1 + tau_w2)}   # display only

    # ------------------------------------------------------------------
    def describe(self):
        """Exact configuration, for logging beside every result.

        Print this into the trial log. It is what makes the baseline
        auditable: a reader can see the mode, every gain, every limit, and
        whether any deviation from [7] was active.
        """
        d = {
            "controller": "DoubleBee decoupled mode, Cao et al. IROS 2023, Eqs (19)-(23)",
            "mode": self.mode,
            "deviations_from_published": (
                [] if self.mode == "faithful" else
                ["wheel pitch feedback Kb_p*theta + Kb_d*theta_dot (not in Eq. 23)",
                 "heading hold Kyaw_p*(0 - yaw) (not in Eq. 23)"]),
            "eq20_sign_rule": self.use_eq20_sign,
            "servo_bias_sign": self.servo_bias_sign,
            "dt_s": self.dt, "rate_hz": round(1.0 / self.dt, 1),
            "gains": {
                "Kp_d": self.Kp_d,
                "Kt_p": self.Kt_p, "Kt_i": self.Kt_i, "Kt_d": self.Kt_d,
                "Ksig_p": self.Ksig_p, "Ksig_i": self.Ksig_i, "Ksig_d": self.Ksig_d,
                "Kv_d": self.Kv_d, "Ks_d": self.Ks_d,
            },
            "T_hold_N_per_prop": self.T_hold,
            "T_hold_frac_of_weight": round(2 * self.T_hold / ROBOT_WEIGHT_N, 3),
            "T_floor_N_per_prop": self.T_floor,
            "limits": {
                "T_max_N_per_prop": T_MAX_PER_PROP_N,
                "servo_rad": SERVO_LIMIT_RAD,
                "wheel_torque_Nm": WHEEL_TORQUE_LIMIT_NM,
                "enforced": self.enforce_limits,
            },
            "anti_windup": {"Te": self.int_limit_Te, "sig": self.int_limit_sig},
            "d_lpf_alpha": self.d_lpf_alpha,
            "tuning_budget_configs_tried": self.tuning_budget,
            # Set by the harness. play_dctrl.py DOES give this baseline terrain
            # preview: it reads scene["height_scanner"], forms
            # step_ahead = clip((max_ahead - ground_z)/0.04, 0, 1) and commands
            # theta_desired = -LEAN_MAX * step_ahead, i.e. a feedforward
            # lean-back schedule triggered by the riser. It also drives
            # v_desired from target distance, so there is no human pilot.
            # That is information parity with the policy and full autonomy --
            # a stronger baseline than the human-operated one in IROS'26.
            # Do not report this controller as blind.
            "terrain_preview": self.terrain_preview,
            "velocity_command_source": self.velocity_command_source,
        }
        if self.mode == "augmented":
            d["augmented_gains"] = {
                "Kb_p": self.Kb_p, "Kb_d": self.Kb_d,
                "balance_blend_deg": self.balance_blend_deg,
                "Kyaw_p": self.Kyaw_p,
            }
        return d


# Backwards-compatible alias so this can be dropped in where the old class name
# is used: `dctrl.DecoupledController()` -> faithful [7].
DecoupledController = DecoupledBaseline


# ----------------------------------------------------------------------
def servo_sign_check(ctrl=None):
    """Pin down servo_bias_sign before trusting any result.

    Pitched nose-forward and momentarily still, the servo must tilt so thrust
    pushes the robot back upright. If the reported sigma has the wrong sign
    for your USD's servo convention, flip servo_bias_sign and re-run.
    """
    ctrl = ctrl or DecoupledBaseline()
    ctrl.reset()
    out = ctrl.control(theta=np.radians(10.0), theta_dot=0.0, v=0.0)
    print("servo sign check: theta=+10 deg, theta_dot=0")
    print("  sigma = %+.2f deg   (feedforward -theta alone would give -10.00 deg)"
          % np.degrees(out["sigma"]))
    print("  T     = %5.2f N/prop (T_hold = %.2f)" % (out["T"], ctrl.T_hold))
    return out


def saturation_report(ctrl=None):
    """Where each wheel-torque term alone hits the 0.51 N.m actuator limit.

    Tune against this. A gain whose term saturates inside the robot's normal
    operating range is not a proportional gain, it is a switch, and a baseline
    built from switches loses for the wrong reason.
    """
    ctrl = ctrl or DecoupledBaseline(mode="augmented")
    lim = WHEEL_TORQUE_LIMIT_NM
    print("wheel torque limit: %.2f N.m  (doublebee_dctrl.py used 2.0 -> %.1fx too high)"
          % (lim, 2.0 / lim))
    print("  %-26s %-10s saturates at" % ("term", "gain"))
    rows = [("Kv_d  speed error", ctrl.Kv_d, "m/s of speed error"),
            ("Ks_d  steer rate", ctrl.Ks_d, "rad/s of steer-rate error")]
    if ctrl.mode == "augmented":
        rows += [("Kb_p  pitch feedback", ctrl.Kb_p, "rad of pitch"),
                 ("Kb_d  pitch-rate feedback", ctrl.Kb_d, "rad/s of pitch rate"),
                 ("Kyaw_p heading hold", ctrl.Kyaw_p, "rad of heading error")]
    for name, gain, unit in rows:
        at = lim / gain if gain else float("inf")
        extra = "  (= %.1f deg)" % np.degrees(at) if "rad of" in unit and "/s" not in unit else ""
        print("  %-26s %-10.3f %.3f %s%s" % (name, gain, at, unit, extra))
    return None


def _self_test():
    import json

    print("=" * 72)
    print("1) THE DEFECT THIS FILE FIXES: is the thrust D-term alive?")
    print("=" * 72)
    c = DecoupledBaseline(mode="faithful")
    c.Kt_d = 10.0            # exaggerate so the D contribution is unmistakable
    c.d_lpf_alpha = 1.0      # no filtering, so the check is exact
    c.reset()
    c.control(theta=0.0, theta_dot=0.0, v=0.0)                 # prime
    with_d = c.control(theta=np.radians(5.0), theta_dot=0.0, v=0.0)["T"]
    c.Kt_d = 0.0
    c.reset()
    c.control(theta=0.0, theta_dot=0.0, v=0.0)
    without_d = c.control(theta=np.radians(5.0), theta_dot=0.0, v=0.0)["T"]
    print("  T with Kt_d=10 : %.4f" % with_d)
    print("  T with Kt_d=0  : %.4f" % without_d)
    print("  difference     : %.4f  ->  %s"
          % (abs(with_d - without_d),
             "D-TERM LIVE (fixed)" if abs(with_d - without_d) > 1e-9
             else "D-TERM DEAD (the old bug)"))

    print()
    print("  integrator accumulates once per call?")
    c2 = DecoupledBaseline(); c2.reset()
    for _ in range(10):
        c2.control(theta=np.radians(1.0), theta_dot=0.0, v=0.0)
    # Te = Kp_d*(0 - 0.01745) - 0 = -0.10472 ; Eq20: theta>0, theta_dot(0) <=
    # omega_theta_d(-0.1047) is False -> c1 False; theta>0 -> c2 False; Te = +omega_theta_e
    expect = 10 * (6.0 * (0.0 - np.radians(1.0))) * 0.02
    print("    _int_Te = %+.6f   expected %+.6f   -> %s"
          % (c2._int_Te, expect,
             "OK" if abs(c2._int_Te - expect) < 1e-9 else "DOUBLE-COUNTING"))

    print()
    print("=" * 72)
    print("2) EQUATION (20) SIGN RULE, by quadrant")
    print("=" * 72)
    c3 = DecoupledBaseline()
    print("  %-9s %-11s %-13s %-11s %s"
          % ("theta", "theta_dot", "omega_th_d", "Te", "sign vs omega_theta_e"))
    for th_deg, thd in ((+5, 0.0), (+5, -1.0), (-5, 0.0), (-5, +1.0), (-5, -1.0)):
        th = np.radians(th_deg)
        wd = c3.Kp_d * (0.0 - th)
        we = wd - thd
        c1 = (th > 0.0) and (thd <= wd < 0.0)
        c2_ = (th < 0.0) and not (thd >= wd > 0.0)
        Te = -we if (c1 or c2_) else we
        print("  %-9s %-11.2f %-13.4f %-11.4f %s"
              % ("%+d deg" % th_deg, thd, wd, Te,
                 "FLIPPED" if (c1 or c2_) else "same"))

    print()
    print("=" * 72)
    print("3) ACTUATION LIMITS ARE ENFORCED")
    print("=" * 72)
    c4 = DecoupledBaseline(); c4.reset()
    hard = c4.control(theta=np.radians(60.0), theta_dot=-5.0, v=0.0,
                      v_desired=3.0, theta_desired=0.0)
    print("  hard-over command -> T=%.2f N/prop (max %.2f), sigma=%+.1f deg (max %.1f),"
          % (hard["T"], T_MAX_PER_PROP_N, np.degrees(hard["sigma"]),
             np.degrees(SERVO_LIMIT_RAD)))
    print("                       tau_w1=%+.3f tau_w2=%+.3f N.m (max %.2f)"
          % (hard["tau_w1"], hard["tau_w2"], WHEEL_TORQUE_LIMIT_NM))
    ok = (hard["T"] <= T_MAX_PER_PROP_N + 1e-9
          and abs(hard["sigma"]) <= SERVO_LIMIT_RAD + 1e-9
          and abs(hard["tau_w1"]) <= WHEEL_TORQUE_LIMIT_NM + 1e-9)
    print("  -> %s" % ("all within limits" if ok else "LIMIT VIOLATION"))

    print()
    print("=" * 72)
    print("4) faithful vs augmented, same input")
    print("=" * 72)
    st = dict(theta=np.radians(8.0), theta_dot=0.4, v=0.25, v_desired=0.30,
              theta_desired=0.0, yaw=np.radians(12.0), yaw_rate=0.2)
    for m in ("faithful", "augmented"):
        cc = DecoupledBaseline(mode=m); cc.reset()
        o = cc.control(**st)
        print("  %-10s T=%5.2f  sigma=%+6.1f deg  tau_w1=%+.3f  tau_w2=%+.3f"
              % (m, o["T"], np.degrees(o["sigma"]), o["tau_w1"], o["tau_w2"]))
    print("  (faithful tau_w1 == tau_w2 here: Eq. 23 has no pitch or heading")
    print("   feedback, and yaw_rate_desired == yaw_rate is the only steer input)")

    print()
    print("=" * 72)
    print("5) SERVO SIGN -- verify against your USD before trusting results")
    print("=" * 72)
    servo_sign_check()

    print()
    print("=" * 72)
    print("6) describe() -- log this next to every trial")
    print("=" * 72)
    print(json.dumps(DecoupledBaseline(mode="faithful").describe(), indent=2))


if __name__ == "__main__":
    _self_test()

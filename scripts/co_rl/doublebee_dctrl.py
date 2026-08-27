"""
DoubleBee DECOUPLED MODE controller — directly from the paper
(Cao et al., arXiv:2303.05075, Section III-D, equations 19-23).

This is the PAPER'S OWN controller for ground/decoupled mode. It does NOT
try to balance like an inverted pendulum via MPC. Instead it uses the
decoupled scheme:
  - PITCH is controlled by propeller THRUST + SERVO tilt  (eqs 19-22)
  - POSITION is controlled by WHEEL torque                (eq 23)

This is the right way to stabilize DoubleBee. Use it directly, or as the
'decoupled mode' baseline Muqing asked for.

Plug into play.py the same way as the MPC: each step, read state, call
decoupled_control(), apply the action.
"""

import numpy as np


class DecoupledController:
    """Decoupled-mode controller from DoubleBee paper eqs 19-23."""

    def __init__(self):
        # --- pitch control gains (eqs 19, 20, 21, 22) ---
        self.Kp_d   = 6.0     # eq19: desired ang vel from pitch error (Kd_p)
        # throttle PID (eq21) — thrust bias for pitch torque
        self.Kt_p   = 2.0
        self.Kt_i   = 0.1
        self.Kt_d   = 0.05
        # servo PID (eq22) — servo bias on top of -theta
        self.Ksig_p = 0.2 # was 0.5
        self.Ksig_i = 0.02
        self.Ksig_d = 0.08 # was 0.08
        # --- position control gains (eq23) ---
        self.Kv_d   = 1.0     # wheel torque from desired velocity
        self.Ks_d   = 0.08    # steer/yaw gain

        # reference throttle (T_hold): thrust to hold robot, tune to ~hover-ish
        # REAL VALUE from Muqing's test_ok.param: BB_HOV_DC=1335 PWM -> 8.66N
        # via pwm_to_thrust(1335). Very close to prior guess of 8.0 - validated.
        self.T_hold = 8.66     # N per prop baseline (from real robot calibration)

        # integrator clamps (anti-windup). Both loops saturate downstream --
        # T at [0, 33.14] N and sigma at +/-45 deg -- so an unbounded integral
        # can hold the output at a rail well past the point the error reverses.
        self.int_Te_limit = 5.0
        self.int_sig_limit = 5.0

        # derivative low-pass, in [0, 1]. 1.0 = raw difference. The error signal
        # is built from theta_dot, which on hardware comes from quaternion
        # differencing and is genuinely noisy; a raw 1/dt difference amplifies
        # that by 50x at 50 Hz.
        self.d_lpf_alpha = 0.3

        # integrator states
        self._int_Te = 0.0
        self._int_sig = 0.0
        self._prev_Te = 0.0
        self._prev_sig_e = 0.0
        self._d_Te = 0.0
        self._d_sig = 0.0
        # True until the first control() call. The first derivative must be
        # forced to zero: with _prev initialised to 0, the first dsig is
        # (sig_e - 0)/dt, a step-input spike. Measured at theta = +10 deg it was
        # -4.19 rad of the -4.57 rad servo command -- 92% of the output, driving
        # the servo straight to its -45 deg stop, and pulling thrust to 3.95 N
        # (BELOW the 8.66 N hold) on a robot that is already pitching over.
        self._first_call = True
        self.dt = 0.02

    def reset(self):
        self._int_Te = 0.0
        self._int_sig = 0.0
        self._prev_Te = 0.0
        self._prev_sig_e = 0.0
        self._d_Te = 0.0
        self._d_sig = 0.0
        self._first_call = True

    def control(self, theta, theta_dot, v, v_desired=0.0,
            theta_desired=0.0, yaw_rate=0.0, yaw_rate_desired=0.0, yaw=0.0,
            yaw_gain_scale=1.0):
        """Compute decoupled-mode control.

        Args:
          theta:        current pitch [rad]
          theta_dot:    current pitch rate [rad/s]
          v:            current forward velocity [m/s]
          v_desired:    desired forward velocity [m/s]
          theta_desired: desired pitch [rad] (0 = upright)
          yaw_rate, yaw_rate_desired: for steering

        Returns dict with T (thrust per prop, N), sigma (servo rad), tau_w (wheel torque).
        """
        # # --- eq 19: desired angular velocity from pitch error ---
        # omega_theta_d = self.Kp_d * (theta_desired - theta)
        # omega_theta_e = omega_theta_d - theta_dot

        # # --- eq 20: throttle error sign rule ---
        # # (simplified: use omega_theta_e directly; sign logic from paper eq20)
        # if (theta > 0 and theta_dot <= 0) or (theta < 0 and not (theta_dot >= 0)):
        #     Te = -omega_theta_e
        # else:
        #     Te = omega_theta_e

        # --- REPLACE the eq20 conditional logic entirely with this ---
        omega_theta_d = self.Kp_d * (theta_desired - theta)
        omega_theta_e = omega_theta_d - theta_dot

        # --- eq 21: throttle PID -> thrust ---
        #
        # REWRITTEN 2026-08-26. The previous version had three defects that made
        # this a PI controller with a double-rate integrator, not the PID it
        # appears to be. That matters because IROS R1 asked specifically for
        # "a properly tuned version of the baseline" -- a reviewer who reads this
        # file would raise the same objection again.
        #
        #   1. self._int_Te += Te*dt appeared TWICE per call, so the effective
        #      integral gain was 2 * Kt_i and the integrator wound up twice as
        #      fast as the gain implies.
        #   2. self._prev_Te = Te was assigned BEFORE the second derivative was
        #      taken, so dTe = (Te - Te)/dt was identically zero on every call.
        #      Kt_d = 0.05 never contributed anything.
        #   3. The first T, built from abs(correction_effort), was overwritten on
        #      the next line and never used -- dead code that made the intent
        #      ambiguous (magnitude-only vs signed correction).
        #
        # Single update, single integration, derivative taken against the value
        # from the PREVIOUS call, which is what a PID means.
        Te = omega_theta_e
        self._int_Te += Te * self.dt
        # anti-windup: T saturates at [0, 33.14], so an unbounded integrator can
        # park the loop at a rail and stay there long after the error reverses.
        self._int_Te = float(np.clip(self._int_Te, -self.int_Te_limit, self.int_Te_limit))
        raw_dTe = 0.0 if self._first_call else (Te - self._prev_Te) / self.dt
        self._d_Te += self.d_lpf_alpha * (raw_dTe - self._d_Te)
        dTe = self._d_Te
        self._prev_Te = Te

        # MAGNITUDE-ONLY correction. Both propellers receive the SAME throttle, so
        # T produces no pitch torque by itself -- sigma sets the DIRECTION of the
        # righting moment (T*Lprop*sin(sigma)) and T sets its MAGNITUDE. A signed
        # correction therefore reduces thrust exactly when the lean is largest,
        # which is backwards.
        #
        # Measured in the closed-loop test below, with the real geometry:
        #
        #   signed T    5 deg held   15 deg held   30 deg FELL   50 deg FELL
        #   |T| only    5 deg held   15 deg held   30 deg held   50 deg held
        #
        # On the signed runs T collapsed to 0.0 N as it went over. This line was
        # present in the original file, was overwritten by dead code, and was
        # briefly deleted on 2026-08-26 as redundant -- it is not.
        correction = self.Kt_p * Te + self.Kt_i * self._int_Te + self.Kt_d * dTe
        T = self.T_hold + abs(correction)
        T = float(np.clip(T, 0.0, 33.14))   # real thrust max per prop

        # --- eq 22: servo = -theta + PID(pitch error) ---
        sig_e = omega_theta_e
        self._int_sig += sig_e * self.dt
        self._int_sig = float(np.clip(self._int_sig, -self.int_sig_limit, self.int_sig_limit))
        raw_dsig = 0.0 if self._first_call else (sig_e - self._prev_sig_e) / self.dt
        self._d_sig += self.d_lpf_alpha * (raw_dsig - self._d_sig)
        dsig = self._d_sig
        self._prev_sig_e = sig_e
        self._first_call = False
        sigma = (-theta
                 + self.Ksig_p * sig_e
                 + self.Ksig_i * self._int_sig
                 + self.Ksig_d * dsig)
        # sigma = (-theta
                #   - self.Ksig_p * sig_e
                #   - self.Ksig_i * self._int_sig
                #   - self.Ksig_d * dsig)
        sigma = float(np.clip(sigma, -0.785, 0.785))   # your ±45deg servo limit

        # --- gain-scheduled balance term (needed before eq23 below) ---
        # Near upright (small theta), thrust has almost no lever arm, so the
        # WHEELS must do the balancing there, like a normal balance bot.
        balance_authority = float(np.clip(1.0 - abs(theta) / np.radians(20), 0.0, 1.0))
        Kb_p = 5.0 # was 8.0
        Kb_d = 1.0 # was 1.5
        wheel_balance_term = Kb_p * theta + Kb_d * theta_dot

        # --- eq 23: wheel torque from desired velocity ---
        # tau_w1 = (self.Kv_d * (v_desired - v) - self.Ks_d * (yaw_rate_desired - yaw_rate)
        #           + balance_authority * wheel_balance_term)
        # tau_w2 = (self.Kv_d * (v_desired - v) + self.Ks_d * (yaw_rate_desired - yaw_rate)
        #           + balance_authority * wheel_balance_term)
        # Kyaw_p = 0.6  # NEW: heading-hold proportional gain, tune small
        Kyaw_p_base = 0.6
        # scale yaw gain up during active climbing, when disturbance is strongest
        Kyaw_p = Kyaw_p_base * yaw_gain_scale
        heading_correction = Kyaw_p * (0.0 - yaw)  # pull back toward yaw=0

        tau_w1 = (self.Kv_d * (v_desired - v) - self.Ks_d * (yaw_rate_desired - yaw_rate)
                - heading_correction
                + balance_authority * wheel_balance_term)
        tau_w2 = (self.Kv_d * (v_desired - v) + self.Ks_d * (yaw_rate_desired - yaw_rate)
                + heading_correction
                + balance_authority * wheel_balance_term)
        tau_w = 0.5 * (tau_w1 + tau_w2)
        tau_w = float(np.clip(tau_w, -2.0, 2.0))

        return {"T": T, "sigma": sigma, "tau_w": tau_w,
                "tau_w1": tau_w1, "tau_w2": tau_w2}


if __name__ == "__main__":
    # ---------------------------------------------------------------------
    # CLOSED-LOOP SMOKE TEST, added 2026-08-26.
    #
    # The previous test made two disconnected control() calls from reset, which
    # says nothing about stability and produced misleading saturated outputs
    # (a 50 Hz loop never sees the error jump the way that test did).
    #
    # This steps the actual plant: an inverted pendulum about the wheel axle,
    #
    #     I * theta_ddot = W*Lcom*sin(theta) - T*Lprop*sin(sigma)
    #
    # with the MEASURED geometry -- m 3.2182 kg, CoM 0.1016 m above the axle,
    # propellers 0.4476 m above it. Gravity tips, world-vertical thrust rights.
    # Run this before hardware: if it cannot hold the pendulum here, it will not
    # hold it on the robot.
    # ---------------------------------------------------------------------
    m, g = 3.2182, 9.81
    W, Lcom, Lprop = m * g, 0.1016, 0.4476
    I = m * Lcom ** 2 * 3.0          # 3x point-mass, props are far from the axle
    dt = 0.02

    for theta0_deg in (5.0, 15.0, 30.0, 50.0):
        ctrl = DecoupledController()
        ctrl.reset()
        theta, theta_dot, v = np.radians(theta0_deg), 0.0, 0.0
        peak, fell = abs(theta), False
        for k in range(250):                       # 5 s
            out = ctrl.control(theta=theta, theta_dot=theta_dot, v=v, v_desired=0.0)
            T_tot = 2.0 * out["T"]                 # both propellers
            tip = W * Lcom * np.sin(theta)
            # sigma = -theta is what holds thrust world-vertical, so a NEGATIVE
            # sigma must produce a POSITIVE restoring moment: restoring =
            # -T*Lprop*sin(sigma). Getting this backwards makes a working
            # controller look broken (it did, on 2026-08-26).
            right = -T_tot * Lprop * np.sin(out["sigma"])
            theta_dot += (tip - right) / I * dt
            theta += theta_dot * dt
            peak = max(peak, abs(theta))
            if abs(theta) > np.radians(70):
                fell = True
                break
        print("  start %4.1f deg -> %-9s peak %5.1f deg, final %+6.1f deg  (T=%.1fN sigma=%+.0fdeg)"
              % (theta0_deg, "FELL" if fell else "held", np.degrees(peak),
                 np.degrees(theta), 2 * out["T"], np.degrees(out["sigma"])))
    print("\n  'held' at all four means the decoupled controller is worth hardware time.")

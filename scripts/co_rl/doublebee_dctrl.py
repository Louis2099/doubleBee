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

        # integrator states
        self._int_Te = 0.0
        self._int_sig = 0.0
        self._prev_Te = 0.0
        self._prev_sig_e = 0.0
        self.dt = 0.02

    def reset(self):
        self._int_Te = 0.0
        self._int_sig = 0.0
        self._prev_Te = 0.0
        self._prev_sig_e = 0.0

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

        Te = omega_theta_e
        self._int_Te += Te * self.dt
        dTe = (Te - self._prev_Te) / self.dt
        self._prev_Te = Te

        correction_effort = self.Kt_p * Te + self.Kt_i * self._int_Te + self.Kt_d * dTe
        T = self.T_hold + abs(correction_effort)   # magnitude-only — NEVER collapses to 0
        T = float(np.clip(T, 0.0, 33.14))

        # --- eq 21: throttle PID -> thrust ---
        self._int_Te += Te * self.dt
        dTe = (Te - self._prev_Te) / self.dt
        self._prev_Te = Te
        T = self.T_hold + self.Kt_p * Te + self.Kt_i * self._int_Te + self.Kt_d * dTe
        T = float(np.clip(T, 0.0, 33.14))   # your real thrust max per prop

        # --- eq 22: servo = -theta + PID(pitch error) ---
        sig_e = omega_theta_e
        self._int_sig += sig_e * self.dt
        dsig = (sig_e - self._prev_sig_e) / self.dt
        self._prev_sig_e = sig_e
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
    # quick sanity test: robot pitched forward 10deg, should command correction
    ctrl = DecoupledController()
    out = ctrl.control(theta=np.radians(10), theta_dot=0.0, v=0.0, v_desired=0.0)
    print("Pitched +10deg, want upright:")
    print(f"  T={out['T']:.2f}N  sigma={np.degrees(out['sigma']):.1f}deg  tau_w={out['tau_w']:.3f}Nm")
    print("  (servo should tilt ~-10deg to counter the pitch)")

    out = ctrl.control(theta=0.0, theta_dot=0.0, v=0.0, v_desired=0.5)
    print("\nUpright, want to move forward 0.5 m/s:")
    print(f"  T={out['T']:.2f}N  sigma={np.degrees(out['sigma']):.1f}deg  tau_w={out['tau_w']:.3f}Nm")
    print("  (wheel torque should be positive to drive forward)")
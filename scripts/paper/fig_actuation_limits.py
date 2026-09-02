"""Why this platform needs thrust to climb: the actuation limits, derived.

Three panels, all from measured quantities -- no training runs, no rollouts.

(a) Wheel torque required to lift the axle over a riser, against the torque the
    wheels can actually deliver. The requirement is ABOVE the limit at every
    height in the operating range, so a wheels-only version of this robot
    cannot climb a step of any size. That is the figure's whole point: thrust
    is not an optimisation here, it is a requirement.

(b) The propeller thrust needed to unload the wheels enough to close that gap,
    as a fraction of weight, against what the propellers can produce.

(c) The geometric wall. A wheel rotating about a step edge has lever arm
    d = sqrt(2rh - h^2); at h = r the edge is level with the axle, and beyond
    it the edge is ABOVE the axle -- the wheel meets the vertical face at its
    widest point and pushing forward only presses it into the wall. So
    h_max = r regardless of torque, which is why r sets the evaluation height.

MEASURED INPUTS (sources in-line):
    m = 3.2182 kg      weighed 2026-08-25 (doublebee_v1.py: was wrong twice
                       before -- 2.76 kg guessed, 4.4665 kg from authored USD)
    tau = 0.51 N.m     per-wheel effort_limit, doublebee_v1.py
    T_max = 36.56 N    2 x pwm2thrust(1650 us), the simulator's PWM cap
    L_com = 0.14 m     CoM above the wheel axle
    L_prop = 0.44 m    propellers above the wheel axle

    r      wheel radius -- THE ONE NUMBER NOT IN THE CODEBASE. The USD is
           binary crate and deployment commands rad/s directly, so nothing
           converts to metres. 5.5-6.0 cm is a tape measurement. Everything
           in panel (c) and the x-axis of (a)/(b) depends on it, so measure it
           properly before this figure goes in a paper.

    python3 fig_actuation_limits.py -o figs/fig_actuation_limits.pdf --r 0.058
"""
import argparse

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

M_KG = 3.2182
G = 9.81
W = M_KG * G                 # 31.57 N
TAU_WHEEL = 0.51             # N.m per wheel
T_MAX = 36.56                # N, both propellers at the 1650 us PWM cap
L_COM = 0.14
L_PROP = 0.44


def lever(r, h):
    """Horizontal distance from step edge to axle when the wheel touches the edge."""
    return np.sqrt(np.maximum(2 * r * h - h ** 2, 0.0))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-o", "--out", default="fig_actuation_limits.pdf")
    p.add_argument("--r", type=float, default=0.058, help="wheel radius, m (MEASURE THIS)")
    a = p.parse_args()
    r = a.r

    h = np.linspace(0.005, r, 300)
    d = lever(r, h)
    tau_req = (W / 2) * d                       # per wheel, unloaded case
    Tz_req = W - 2 * TAU_WHEEL / d              # thrust needed to close the gap

    fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.1))

    # (a) torque
    ax[0].plot(100 * h, tau_req, lw=2, label="required")
    ax[0].axhline(TAU_WHEEL, color="crimson", ls="--", lw=1.6,
                  label="wheel limit (%.2f N$\\cdot$m)" % TAU_WHEEL)
    ax[0].fill_between(100 * h, TAU_WHEEL, tau_req, where=tau_req > TAU_WHEEL,
                       color="crimson", alpha=0.12)
    ax[0].set_xlabel("riser height (cm)")
    ax[0].set_ylabel("wheel torque (N$\\cdot$m)")
    ax[0].set_title("(a) wheels alone cannot climb", fontsize=9, loc="left")
    ax[0].legend(fontsize=7, loc="lower right")
    ax[0].grid(alpha=0.3)

    # (b) thrust
    frac = 100 * np.clip(Tz_req, 0, None) / W
    ax[1].plot(100 * h, frac, lw=2, label="required")
    # T/W = 116% is so far above the requirement that plotting it to scale
    # squashes everything informative into the bottom third. Annotate instead.
    ax[1].set_ylim(0, 62)
    ax[1].annotate("available: %.0f%% of weight (T/W = %.2f)" % (100 * T_MAX / W, T_MAX / W),
                   xy=(0.03, 0.93), xycoords="axes fraction", fontsize=7,
                   color="seagreen")
    ax[1].axhline(100 * W * L_COM / L_PROP / W, color="darkorange", ls=":", lw=1.6,
                  label="static-stability floor (%.0f%%)" % (100 * L_COM / L_PROP))
    ax[1].set_xlabel("riser height (cm)")
    ax[1].set_ylabel("thrust (% of weight)")
    ax[1].set_title("(b) thrust required to unload the wheels", fontsize=9, loc="left")
    ax[1].legend(fontsize=7, loc="lower right")
    ax[1].grid(alpha=0.3)

    # (c) geometry
    hh = np.linspace(0.005, 1.35 * r, 400)
    dd = np.where(hh <= r, lever(r, np.minimum(hh, r)), np.nan)
    ax[2].plot(100 * hh, 100 * dd, lw=2)
    ax[2].axvline(100 * r, color="k", ls="--", lw=1.4)
    ax[2].annotate("$h_{\\max}=r$", xy=(100 * r, 100 * r * 0.45),
                   xytext=(100 * r * 1.03, 100 * r * 0.45), fontsize=9)
    ax[2].axvspan(100 * r, 100 * hh[-1], color="0.85")
    ax[2].text(100 * r * 1.16, 100 * r * 0.62, "edge above axle:\nno rolling",
               fontsize=7.5, ha="center", va="center")
    ax[2].set_xlabel("riser height (cm)")
    ax[2].set_ylabel("edge-to-axle lever arm (cm)")
    ax[2].set_title("(c) the geometric wall, $r$ = %.1f cm" % (100 * r), fontsize=9, loc="left")
    ax[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(a.out, bbox_inches="tight")

    at6 = min(0.06, r)
    d6 = lever(r, at6)
    print("wrote", a.out)
    print("at h = %.1f cm:  tau needed %.2f N.m (have %.2f, short %.1fx),  thrust %.0f%% of weight"
          % (100 * at6, (W / 2) * d6, TAU_WHEEL, (W / 2) * d6 / TAU_WHEEL,
             100 * (W - 2 * TAU_WHEEL / d6) / W))


if __name__ == "__main__":
    main()

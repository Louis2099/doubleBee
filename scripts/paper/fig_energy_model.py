#!/usr/bin/env python3
"""Energy-model validation figure (paper Fig. N).

Answers IROS R4: "several implementation details would benefit from clearer
reporting or further validation, including the calibration and validation of the
energy/power model ... These details are important because the central claim of
the paper is specifically about energy-aware learning."

Four panels:
  (a) thrust vs PWM      -- bench measurements against the degree-4 fit used in sim
  (b) electrical power vs PWM
  (c) residuals for both, as % of reading
  (d) thrust per watt -- the quantity the energy reward is actually trading against

Panel (d) is the one that makes an argument rather than just validating: it shows
propeller efficiency FALLING monotonically with throttle, which is why a policy
that modulates thrust beats one that holds a constant hover value.

Usage:  python scripts/paper/fig_energy_model.py [--out fig_energy_model.pdf]
"""
from __future__ import annotations

import argparse
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MDP = os.path.normpath(os.path.join(
    HERE, "..", "..", "lab", "doublebee", "tasks", "manager_based",
    "locomotion", "velocity", "mdp"))

# Reference operating points, for annotation.
#
# BB_HOV_DC = 1335 us is NOT aerial hover, despite the parameter name and the
# "~hover-ish" comment in doublebee_dctrl.py. It gives 8.66 N per propeller =
# 17.3 N total against a 31.57 N robot, i.e. T/W = 0.55. It is the DECOUPLED-MODE
# HOLD throttle: enough thrust to partially unload the machine while the wheels
# drive, leaving 45% of the weight on the tyres for traction. True hover needs
# 1578 us. Both are drawn, because the gap between them IS the hybrid argument.
PWM_HOLD = 1335.0       # BB_HOV_DC, decoupled-mode hold
PWM_HOVER = 1578.0      # T = W, computed from the thrust fit
WEIGHT_N = 31.57        # 3.2182 kg, measured


def load():
    rows = list(csv.DictReader(open(os.path.join(MDP, "PWM2TE.csv"))))
    pwm = np.array([float(r["PWM"]) for r in rows])
    thrust = np.array([float(r["Thrust (N)"]) for r in rows])
    power = np.array([float(r["Power (W)"]) for r in rows])
    ct = json.load(open(os.path.join(MDP, "pwm2thrust_params.json")))
    cp = json.load(open(os.path.join(MDP, "pwm2power_params.json")))
    return pwm, thrust, power, ct, cp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "fig_energy_model.pdf"))
    ap.add_argument("--dpi", type=int, default=300)
    a = ap.parse_args()

    pwm, T, P, ct, cp = load()
    fit_T = np.polyval(ct["coeffs"], pwm)
    fit_P = np.polyval(cp["coeffs"], pwm)
    grid = np.linspace(pwm.min(), pwm.max(), 400)
    gT = np.polyval(ct["coeffs"], grid)
    gP = np.polyval(cp["coeffs"], grid)

    rmse_T = float(np.sqrt(((fit_T - T) ** 2).mean()))
    rmse_P = float(np.sqrt(((fit_P - P) ** 2).mean()))

    plt.rcParams.update({
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
        "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.constrained_layout.use": True,
    })
    fig, ax = plt.subplots(2, 2, figsize=(7.0, 4.6))

    def mark(a_, labels=True):
        a_.axvline(PWM_HOLD, color="0.55", ls="--", lw=0.8, zorder=0)
        a_.axvline(PWM_HOVER, color="0.55", ls="-.", lw=0.8, zorder=0)
        if labels:
            a_.annotate("hold\n(T/W 0.55)", xy=(PWM_HOLD, a_.get_ylim()[1]),
                        xytext=(-3, -16), textcoords="offset points",
                        fontsize=6.2, color="0.35", ha="right")
            a_.annotate("hover\n(T/W 1.0)", xy=(PWM_HOVER, a_.get_ylim()[1]),
                        xytext=(3, -16), textcoords="offset points",
                        fontsize=6.2, color="0.35")

    # (a) thrust ----------------------------------------------------------
    A = ax[0, 0]
    A.plot(grid, gT, "-", lw=1.4, color="C0", label="degree-4 fit", zorder=2)
    A.plot(pwm, T, "o", ms=4.5, mfc="white", mec="C0", mew=1.2,
           label="bench measurement", zorder=3)
    A.axhline(WEIGHT_N / 2.0, color="C3", ls=":", lw=0.9, zorder=1)
    A.annotate("W/2 per prop", xy=(pwm.min(), WEIGHT_N / 2.0), xytext=(4, 3),
               textcoords="offset points", fontsize=6.5, color="C3")
    A.set_xlabel(r"PWM [$\mu$s]"); A.set_ylabel("thrust per propeller [N]")
    A.set_title("(a) thrust model", loc="left")
    A.legend(frameon=False, loc="upper left")
    mark(A)

    # (b) power ------------------------------------------------------------
    B = ax[0, 1]
    B.plot(grid, gP, "-", lw=1.4, color="C1", label="degree-4 fit", zorder=2)
    B.plot(pwm, P, "s", ms=4.5, mfc="white", mec="C1", mew=1.2,
           label="bench measurement", zorder=3)
    B.set_xlabel(r"PWM [$\mu$s]"); B.set_ylabel("electrical power [W]")
    B.set_title("(b) power model", loc="left")
    B.legend(frameon=False, loc="upper left")
    mark(B)

    # (c) residuals --------------------------------------------------------
    C = ax[1, 0]
    C.axhline(0, color="0.6", lw=0.8)
    C.plot(pwm, 100 * (fit_T - T) / T, "o-", ms=3.5, lw=1.0, color="C0",
           label=r"thrust, RMSE %.4f N" % rmse_T)
    C.plot(pwm, 100 * (fit_P - P) / P, "s-", ms=3.5, lw=1.0, color="C1",
           label=r"power, RMSE %.2f W" % rmse_P)
    C.set_xlabel(r"PWM [$\mu$s]"); C.set_ylabel("fit residual [% of reading]")
    C.set_title("(c) residuals", loc="left")
    C.legend(frameon=False, loc="lower left")

    # (d) efficiency -- the panel that makes the argument -------------------
    D = ax[1, 1]
    D.plot(grid, gT / np.maximum(gP, 1e-9), "-", lw=1.5, color="C2")
    D.plot(pwm, T / P, "^", ms=4.5, mfc="white", mec="C2", mew=1.2)
    D.set_xlabel(r"PWM [$\mu$s]"); D.set_ylabel("thrust per watt [N/W]")
    D.set_title("(d) propeller efficiency", loc="left")
    mark(D)
    e_lo = np.polyval(ct["coeffs"], 1100.0) / np.polyval(cp["coeffs"], 1100.0)
    e_hi = np.polyval(ct["coeffs"], 2000.0) / np.polyval(cp["coeffs"], 2000.0)
    D.annotate("%.0f%% drop\n1100 to 2000" % (100 * (1 - e_hi / e_lo)),
               xy=(1700, np.polyval(ct["coeffs"], 1700.0) /
                   np.polyval(cp["coeffs"], 1700.0)),
               xytext=(-6, 26), textcoords="offset points", fontsize=6.5,
               color="C2", ha="right",
               arrowprops=dict(arrowstyle="-", lw=0.6, color="C2"))

    fig.savefig(a.out, dpi=a.dpi, bbox_inches="tight")
    png = os.path.splitext(a.out)[0] + ".png"
    fig.savefig(png, dpi=a.dpi, bbox_inches="tight")
    print("wrote %s and %s" % (a.out, png))

    # numbers for the caption / text --------------------------------------
    rng_T = T.max() - T.min()
    rng_P = P.max() - P.min()
    print("\n  FOR THE CAPTION")
    print("     thrust  RMSE %.4f N over %.1f-%.1f N   = %.3f%% of range"
          % (rmse_T, T.min(), T.max(), 100 * rmse_T / rng_T))
    print("     power   RMSE %.2f W over %.1f-%.1f W   = %.3f%% of range"
          % (rmse_P, P.min(), P.max(), 100 * rmse_P / rng_P))
    print("     worst thrust residual %.3f%% of reading"
          % np.abs(100 * (fit_T - T) / T).max())
    print("     worst power  residual %.3f%% of reading"
          % np.abs(100 * (fit_P - P) / P).max())
    print("     efficiency %.4f N/W at 1100 -> %.4f N/W at 2000  (%.0f%% drop)"
          % (e_lo, e_hi, 100 * (1 - e_hi / e_lo)))
    for lab, p in (("hold  (BB_HOV_DC)", PWM_HOLD), ("hover (T = W)", PWM_HOVER)):
        print("     %-20s pwm %.0f: %5.2f N/prop, %6.1f W/prop, T/W %.2f, %.4f N/W"
              % (lab, p, np.polyval(ct["coeffs"], p), np.polyval(cp["coeffs"], p),
                 2 * np.polyval(ct["coeffs"], p) / WEIGHT_N,
                 np.polyval(ct["coeffs"], p) / np.polyval(cp["coeffs"], p)))
    ph, pv = np.polyval(cp["coeffs"], PWM_HOLD), np.polyval(cp["coeffs"], PWM_HOVER)
    print("     holding costs %.0f%% of what hovering costs" % (100 * ph / pv))


if __name__ == "__main__":
    main()

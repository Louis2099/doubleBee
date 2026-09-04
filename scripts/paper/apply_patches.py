"""Apply the 2026-09-03 changes IN PLACE, on whatever copy of the tree it runs in.

Why in place and not scp: the /home checkout and the /data training box have
diverged, and copying whole files across silently removed the four actuation
ablation classes from hybrid_stair_cfg.py (ImportError on
DoubleBeeHybridStairWheelsOnlyCfg, 2026-09-04 00:14). Patching only the lines
that need to change cannot do that.

Idempotent: re-running it is a no-op. Every existing weight is preserved as the
default, so it works regardless of which copy's numbers are current.

    python3 scripts/paper/apply_patches.py            # from the repo root
    python3 scripts/paper/apply_patches.py --check    # report only, write nothing
"""
import argparse
import os
import re
import sys

BASE = "lab/doublebee/tasks/manager_based/locomotion/velocity/"
H = BASE + "doublebee_env/flat_env/hybrid_stair/hybrid_stair_cfg.py"
R = BASE + "mdp/rewards.py"

DR_BLOCK = '''
{i}# DOUBLEBEE_NO_DR=1 turns domain randomization off.
{i}#
{i}# Measured 2026-09-03: mean episode length at iterations 300-400 was
{i}# 36.0-39.2 across six runs WITHOUT randomization and 23.5-25.6 across five
{i}# runs WITH it. Zero overlap, both clusters tight, so systematic rather than
{i}# seed noise. At 24 steps the robot falls in 0.45 s and cannot reach a target
{i}# sampled 0.5-1.2 m away, starving every downstream signal.
{i}#
{i}# The 11 cm hardware climb (2026-08-27) used baseline_4000, trained with
{i}# randomization OFF, so transfer is already demonstrated without it.
{i}#
{i}# Only the randomization terms are removed. reset_base, reset_robot_joints
{i}# and propeller_aerodynamics are environment setup and stay.
{i}if os.environ.get("DOUBLEBEE_NO_DR", "0") not in ("0", "", "false", "False"):
{i}    _dr_terms = (
{i}        "sample_thrust_scale_dr", "randomize_robot_mass", "randomize_com",
{i}        "push_robot", "randomize_joint_actuator_gains",
{i}        "randomize_servo_actuator_gains", "randomize_friction",
{i}    )
{i}    _off = [t for t in _dr_terms if getattr(self.events, t, None) is not None]
{i}    for _t in _off:
{i}        setattr(self.events, _t, None)
{i}    print("[cfg] domain randomization DISABLED (DOUBLEBEE_NO_DR): "
{i}          + ", ".join(_off), flush=True)'''

V2_HELPER = '''
# DOUBLEBEE_REWARD_V2: rebalance task reward against posture reward.
#
# Measured 2026-09-03 iter 326 (contribution = weight x mean value):
#     posture  alive_upright .0572 + props_upright .0448
#              + vertical_thrust_support .0156 + thrust_recovery .0217 = 0.139
#     task     terminal_goal .0455 + reach_target .0213 + forward .0009
#              + progress_to_target -.0004 + climb -.0024              = 0.065
# Standing upright paid 2.1x what doing the task paid, which IS the flat-ground
# local optimum the energy sweep kept rediscovering.
#
# V2 halves the posture bloc and triples the dense task term: ~3:1 toward task.
# Safe ONLY because runs are warm-started from a policy that already balances
# (scripts/paper/transplant_obs.py). From a random init these terms bootstrap
# balance and halving them would make the falling worse.
#
# NOT changed: reward_progress_to_target sits at weight 10.0 and contributed
# -0.0004, so its underlying signal is ~0. Scaling a zero signal does nothing.
_REWARD_V2 = os.environ.get("DOUBLEBEE_REWARD_V2", "0") not in ("0", "", "false", "False")
_V2_WEIGHTS = {
    "reward_alive_upright": 0.5,
    "reward_props_upright": 2.0,
    "reward_vertical_thrust_support": 1.5,
    "reward_thrust_recovery_under_lean": 3.0,
    "reach_terrain_target": 15.0,
    "terminal_goal_reached": 20.0,
}


def _w(name, default):
    """Reward weight, overridden when DOUBLEBEE_REWARD_V2 is set."""
    return _V2_WEIGHTS[name] if (_REWARD_V2 and name in _V2_WEIGHTS) else default

'''


def patch_dr(check):
    if not os.path.exists(H):
        return "MISSING %s" % H
    s = open(H).read()
    before = [n for n in re.findall(r"^class (\w+)", s, re.M)]
    if "DOUBLEBEE_NO_DR" in s:
        return "already applied (%d classes)" % len(before)
    if not re.search(r"^import os$", s, re.M):
        s = re.sub(r"^import math$", "import math\nimport os", s, count=1, flags=re.M)
    lines = s.split("\n")
    for i, ln in enumerate(lines):
        if ln.strip() == "super().__post_init__()":
            indent = ln[:len(ln) - len(ln.lstrip())]
            lines.insert(i + 1, DR_BLOCK.format(i=indent))
            break
    else:
        return "FAILED: no super().__post_init__() found"
    s = "\n".join(lines)
    after = [n for n in re.findall(r"^class (\w+)", s, re.M)]
    if set(before) != set(after):
        return "FAILED: class list changed, refusing to write"
    compile(s, H, "exec")
    if not check:
        open(H, "w").write(s)
    return "patched, %d classes preserved: %s" % (len(after), ", ".join(after))


def patch_v2(check):
    if not os.path.exists(R):
        return "MISSING %s" % R
    s = open(R).read()
    if "_REWARD_V2" in s:
        return "already applied"
    anchor = "@configclass\nclass RewardsCfg:"
    if anchor not in s:
        return "FAILED: RewardsCfg anchor not found"
    s = s.replace(anchor, V2_HELPER + "\n" + anchor, 1)
    lines = s.split("\n")
    hit = []
    for name in _V2_NAMES:
        for i, ln in enumerate(lines):
            if re.match(r"^    %s = RewTerm\(" % re.escape(name), ln):
                for j in range(i, min(i + 9, len(lines))):
                    m = re.match(r"^(\s*)weight=([0-9.]+),(.*)$", lines[j])
                    if m:
                        # keep the EXISTING number as the default, so this works
                        # even if the two checkouts disagree on current weights
                        lines[j] = '%sweight=_w("%s", %s),%s' % (
                            m.group(1), name, m.group(2), m.group(3))
                        hit.append("%s(was %s)" % (name, m.group(2)))
                        break
                break
    s = "\n".join(lines)
    compile(s, R, "exec")
    if not check:
        open(R, "w").write(s)
    missing = set(_V2_NAMES) - {h.split("(")[0] for h in hit}
    return "patched %d/%d: %s%s" % (len(hit), len(_V2_NAMES), ", ".join(hit),
                                    "  MISSING: %s" % ", ".join(missing) if missing else "")


_V2_NAMES = ["reward_alive_upright", "reward_props_upright",
             "reward_vertical_thrust_support", "reward_thrust_recovery_under_lean",
             "reach_terrain_target", "terminal_goal_reached"]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--check", action="store_true")
    a = p.parse_args()
    if not os.path.isdir("lab"):
        sys.exit("run this from the repo root (the directory containing lab/)")
    print("DOUBLEBEE_NO_DR   : %s" % patch_dr(a.check))
    print("DOUBLEBEE_REWARD_V2: %s" % patch_v2(a.check))
    if a.check:
        print("\n--check: nothing was written")

"""Unit test for SwitchedPropellerAction's step detector.

The detector is the only new maths in the mode-switching baseline, and it is
the part that fails silently: a wrong yaw sign or a wrong median mask still
produces plausible-looking numbers, just measured against the wrong half of the
scan. Isaac Lab is not importable on the hardware desktop, so the geometry is
re-implemented here verbatim and driven with synthetic ray hits.

Run:  python3 scripts/paper/tests/test_switch_trigger.py
"""
import math
import sys

import torch


def detect(hits, org, quat, lookahead=0.105, step_thresh=0.02):
    """Verbatim copy of SwitchedPropellerAction._step_detected's geometry.

    SIM FORWARD IS BODY +Y (rewards.py:264). This file originally built its
    synthetic terrain along +X and projected onto +X, so it agreed with a
    detector that was scanning sideways and passed while the real thing was
    broken. A test that shares the code's assumption tests nothing.
    """
    hits = torch.nan_to_num(hits, nan=0.0, posinf=0.0, neginf=0.0)
    q = quat
    siny = 2.0 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2])
    cosy = 1.0 - 2.0 * (q[:, 2] ** 2 + q[:, 3] ** 2)
    yaw = torch.atan2(siny, cosy).unsqueeze(-1)

    dx = hits[..., 0] - org[:, 0:1]
    dy = hits[..., 1] - org[:, 1:2]
    fwd = -dx * torch.sin(yaw) + dy * torch.cos(yaw)   # body +Y
    z = hits[..., 2]

    ahead = fwd > 0.0
    behind = ~ahead
    within = ahead & (fwd <= lookahead)

    neg_inf = torch.finfo(z.dtype).min
    top = torch.where(within, z, torch.full_like(z, neg_inf)).max(dim=1)[0]
    pos_inf = torch.finfo(z.dtype).max
    ground = torch.where(behind, z, torch.full_like(z, pos_inf)).median(dim=1)[0]

    ok = within.any(dim=1) & behind.any(dim=1)
    return ok & ((top - ground) >= step_thresh)


def yaw_quat(psi):
    return torch.tensor([[math.cos(psi / 2), 0.0, 0.0, math.sin(psi / 2)]])


def grid(step_x=None, step_h=0.06, yaw=0.0, org=(0.0, 0.0)):
    """4x4 scan at 0.07 m spacing. Terrain rises by step_h beyond step_x,
    where step_x is measured along the robot's FORWARD axis, body +Y.

    Offsets are built in the sensor's yaw frame and then rotated into world, so
    the test exercises the same rotation the detector undoes.
    """
    off = [-0.105, -0.035, 0.035, 0.105]
    pts = []
    for gx in off:                      # gx = lateral (body +X)
        for gy in off:                  # gy = forward (body +Y)
            z = 0.0 if (step_x is None or gy < step_x) else step_h
            wx = org[0] + gx * math.cos(yaw) - gy * math.sin(yaw)
            wy = org[1] + gx * math.sin(yaw) + gy * math.cos(yaw)
            pts.append([wx, wy, z])
    return (torch.tensor([pts]),
            torch.tensor([[org[0], org[1], 0.5]]),
            yaw_quat(yaw))


FAIL = []


def check(name, got, want):
    ok = bool(got) == bool(want)
    print("  %-58s %s" % (name, "ok" if ok else "FAIL (got %s)" % bool(got)))
    if not ok:
        FAIL.append(name)


print("flat ground and a step directly ahead")
check("flat ground -> no step", detect(*grid(None))[0], False)
check("step at +0.035 m -> detected", detect(*grid(0.0))[0], True)
check("6 cm step ahead at every yaw", all(
    detect(*grid(0.0, yaw=p))[0].item()
    for p in (0.0, 0.7, 1.6, 3.0, -2.2)), True)

print("\nthe detector must not fire on terrain BEHIND the robot")
# Robot has climbed: everything behind is high, everything ahead is flat.
off = [-0.105, -0.035, 0.035, 0.105]
pts = [[gx, gy, (0.06 if gy < 0 else 0.0)] for gx in off for gy in off]
behind_only = (torch.tensor([pts]), torch.tensor([[0.0, 0.0, 0.5]]), yaw_quat(0.0))
check("step behind only -> no fire", detect(*behind_only)[0], False)

print("\nthreshold and lookahead bound the trigger")
check("1 cm rise under a 2 cm threshold -> no fire",
      detect(*grid(0.0, step_h=0.01))[0], False)
check("3 cm rise over a 2 cm threshold -> fires",
      detect(*grid(0.0, step_h=0.03))[0], True)
check("step only at +0.105, lookahead 0.05 -> out of reach",
      detect(*grid(0.07), lookahead=0.05)[0], False)
check("step only at +0.105, lookahead 0.105 -> in reach",
      detect(*grid(0.07), lookahead=0.105)[0], True)

print("\nstraddling the edge is NOT a step ahead, and the latch is what covers it")
# 12 of 16 rays already on the upper plane: the robot is ON the step, and there
# is no higher terrain within reach. The detector SHOULD go quiet here. What
# must not happen is thrust dropping mid-climb, and that is the latch's job,
# not the detector's -- asserted in the sequence test below.
h = torch.tensor([[[gx, gy, (0.0 if gy < -0.07 else 0.03)]
                   for gx in off for gy in off]])
o = torch.tensor([[0.0, 0.0, 0.5]])
check("mid-climb, nothing higher ahead -> detector quiet",
      detect(h, o, yaw_quat(0.0))[0], False)


def latched(seq, latch_steps):
    """Detector output run through the same latch as process_actions."""
    lat, out = 0.0, []
    for d in seq:
        lat = float(latch_steps) if d else max(0.0, lat - 1.0)
        out.append(lat > 0.0)
    return out


# An approach: flat, flat, step detected, then straddling the edge for 10
# control steps (0.2 s at 50 Hz) with the detector quiet.
approach = [False, False, True] + [False] * 10
held = latched(approach, latch_steps=25)          # 0.5 s at 50 Hz
check("thrust high for every step of the climb after one detection",
      all(held[2:]), True)
check("thrust low before the step is seen", any(held[:2]), False)
check("latch expires on flat ground afterwards",
      latched([True] + [False] * 40, 25)[-1], False)

print("\naction tensor shapes (regression: a 1-D mask broadcast to (N,N))")
# _raw_actions is (N, 1) for the tied propeller term and _tied_scale is (1, 2).
# A 1-D (N,) mask in torch.where silently produces (N, N) instead of selecting
# per environment. This is the exact shape path in process_actions.
N = 8
raw = torch.zeros(N, 1)
tied_scale = torch.tensor([[320.0, -320.0]])
tied_offset = torch.tensor([[320.0, -320.0]])
latch = torch.tensor([3.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 5.0])

hot = latch > 0.0
bad = torch.where(hot, torch.full_like(raw, 1.0), torch.full_like(raw, -0.45))
check("1-D mask really does produce the wrong shape", tuple(bad.shape) == (N, N), True)

hot_col = hot.unsqueeze(-1)
held = torch.where(hot_col, torch.full_like(raw, 1.0), torch.full_like(raw, -0.45))
check("(N,1) mask keeps the action shape", tuple(held.shape) == (N, 1), True)
out = held * tied_scale + tied_offset
check("processed actions are (N, 2)", tuple(out.shape) == (N, 2), True)
check("high envs get counter-rotating full thrust",
      torch.allclose(out[0], torch.tensor([640.0, -640.0])), True)
check("low envs get the low hold",
      torch.allclose(out[1], torch.tensor([320.0 * 0.55, -320.0 * 0.55]), atol=1e-4), True)
check("duty counter stays 1-D", tuple((torch.zeros(N) + hot.float()).shape) == (N,), True)

print("\ntranslation invariance")
check("same step, sensor at (12.4, -7.1)",
      detect(*grid(0.0, org=(12.4, -7.1)))[0], True)

print()
if FAIL:
    print("FAILED: %d" % len(FAIL))
    for f in FAIL:
        print("  -", f)
    sys.exit(1)
print("all trigger-geometry checks passed")

"""Do the two propellers tilt the SAME way, or against each other?

Settles it from geometry rather than from correlations. Holds a fixed nonzero
servo action and prints each propeller's thrust axis in WORLD coordinates, split
into vertical and horizontal parts.

  thrust axis is the propeller body's local +Z (matches apply_propeller_aerodynamics)

Read the last block:
  * horizontal components pointing the SAME way  -> servos agree, thrust vectors
  * horizontal components OPPOSED, sum near zero -> they cancel, and the servo
    action does nothing but waste power. That would be a real bug.

    python3 check_servo_dirs.py --servo 0.8
"""
import argparse
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="Isaac-Velocity-HybridStair-DoubleBee-Play-v1-ppo")
    p.add_argument("--servo", type=float, default=0.8,
                   help="servo action to hold, -1..1")
    p.add_argument("--steps", type=int, default=120)
    a = p.parse_args()

    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True).app

    import torch, gymnasium as gym, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import co_rl  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg
    from isaaclab.utils.math import quat_apply

    env_cfg = parse_env_cfg(a.task, num_envs=1)
    env = gym.make(a.task, cfg=env_cfg)
    base = env.unwrapped
    base.reset()

    robot = base.scene["robot"]
    nact = base.action_manager.total_action_dim
    # find the servo slot by term name so an action remap cannot break this
    idx, servo_i = 0, None
    for name, term in base.action_manager._terms.items():
        if name == "propeller_servo_pos":
            servo_i = idx
            break
        idx += term.action_dim
    if servo_i is None:
        sys.exit("no propeller_servo_pos action term found")
    print("servo action index = %d of %d" % (servo_i, nact))

    act = torch.zeros(1, nact, device=base.device)
    act[0, servo_i] = a.servo
    for _ in range(a.steps):
        base.step(act)

    jn = robot.joint_names
    lj, rj = jn.index("leftPropellerServo"), jn.index("rightPropellerServo")
    print("\nservo JOINT positions: left %+.4f rad   right %+.4f rad"
          % (robot.data.joint_pos[0, lj].item(), robot.data.joint_pos[0, rj].item()))

    bn = robot.body_names
    lp, rp = bn.index("leftPropeller"), bn.index("rightPropeller")
    ids = torch.tensor([lp, rp], device=robot.device)
    quat = robot.data.body_quat_w[:, ids, :]
    local = torch.zeros(1, 2, 3, device=robot.device)
    local[:, :, 2] = 1.0
    w = quat_apply(quat, local)[0]        # [2,3] world thrust axes

    print("\nthrust axis in WORLD frame (x, y, z):")
    for nm, v in (("left ", w[0]), ("right", w[1])):
        print("  %s  (%+.3f, %+.3f, %+.3f)   vertical %+.3f  horizontal %.3f"
              % (nm, v[0], v[1], v[2], v[2], torch.norm(v[:2]).item()))
    hsum = (w[0, :2] + w[1, :2])
    hmag = torch.norm(w[0, :2]).item() + torch.norm(w[1, :2]).item()
    print("\n  horizontal SUM   (%+.3f, %+.3f)  magnitude %.3f" % (hsum[0], hsum[1], torch.norm(hsum)))
    print("  sum of magnitudes %.3f" % hmag)
    if hmag > 1e-6:
        ratio = torch.norm(hsum).item() / hmag
        print("  retained fraction %.2f" % ratio)
        print("\n  %s" % ("AGREE -- the servos vector thrust together." if ratio > 0.8
                          else "PARTIAL cancellation." if ratio > 0.2
                          else "OPPOSED -- the horizontal components cancel. REAL BUG."))
    env.close(); app.close()


if __name__ == "__main__":
    main()

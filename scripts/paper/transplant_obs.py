"""Warm-start the new observation layout from an old checkpoint.

WHY
  Every episode in the current runs ends with the robot falling over
  (Episode_Constraint/time_out is 0.0000, tilt is ~43). So most of training is
  spent relearning balance, a skill baseline_4000 already has: it reaches
  terrain level 1.18 and climbed 11 cm on hardware on 2026-08-27.

  A plain --resume cannot load it because the goal command gained a range
  channel on 2026-09-03, so the first layer's input width changed. This
  rewrites that layer: old columns are copied into their new positions and the
  one new column is zero-initialised, which makes the loaded policy compute
  EXACTLY what it computed before (a zero column contributes nothing) while
  leaving the new input free to acquire weight during training.

LAYOUT (from db_inference.py, one frame)
    [0:2]   wheel_vel          [2:5]   base_lin_vel     [5:8]   base_ang_vel
    [8:11]  projected_gravity  [11:27] height_scan      [27:29] wheel_contact
    [29:32] velocity_commands  [32:38] actions
  The new channel is appended to velocity_commands, so it lands at index 32 and
  everything from the old index 32 onward shifts up by one. With
  num_policy_stacks frames concatenated, that insertion repeats once per frame.

BIAS, STATED UP FRONT
  Warm-starting every w_E arm from one checkpoint trained at w_E=0.25 biases
  the sweep toward that setting, and effects will be understated. In exchange
  it REMOVES the initialisation variance that made the 2026-09-03 sweep
  unusable, where two runs with byte-identical configs finished at 1.18 and
  0.04. A common initialisation makes the arms a controlled comparison rather
  than six independent lottery tickets. Report it: "all arms were initialised
  from a common pretrained policy and fine-tuned under different w_E".

    python3 transplant_obs.py --in model_3999.pt --out warm_start.pt
    # then: train.py --resume True --load_run <dir> --checkpoint warm_start.pt
"""
import argparse
import sys

import torch


def expand(W, old_obs, new_obs, frame_old, frame_new, insert_at, stacks):
    """Widen a [out, in] weight, inserting a zero column per stacked frame."""
    extra = W.shape[1] - old_obs          # critic inputs carry action dims too
    if extra < 0:
        return None
    out = torch.zeros(W.shape[0], new_obs + extra, dtype=W.dtype, device=W.device)
    for s in range(stacks):
        o, n = s * frame_old, s * frame_new
        out[:, n:n + insert_at] = W[:, o:o + insert_at]
        # out[:, n + insert_at] stays zero: the new range channel
        out[:, n + insert_at + 1:n + frame_new] = W[:, o + insert_at:o + frame_old]
    if extra:
        out[:, new_obs:] = W[:, old_obs:]
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="src", required=True)
    p.add_argument("--out", dest="dst", required=True)
    p.add_argument("--frame-old", type=int, default=38)
    p.add_argument("--frame-new", type=int, default=39)
    p.add_argument("--insert-at", type=int, default=32,
                   help="index WITHIN one frame where the new channel goes")
    p.add_argument("--stacks", type=int, default=2, help="num_policy_stacks")
    a = p.parse_args()

    old_obs = a.frame_old * a.stacks
    new_obs = a.frame_new * a.stacks
    ck = torch.load(a.src, map_location="cpu")
    print("checkpoint keys: %s" % sorted(ck.keys()))
    print("obs %d -> %d  (frame %d -> %d, insert at %d, %d stacks)"
          % (old_obs, new_obs, a.frame_old, a.frame_new, a.insert_at, a.stacks))

    n_done = 0
    for sd_key in ("actor_state_dict", "critic_state_dict", "target_critic_state_dict"):
        if sd_key not in ck:
            continue
        sd = ck[sd_key]
        # The input layer is the only 2-D weight whose column count starts at
        # old_obs (actor) or old_obs + action_dim (critic).
        hits = [k for k, v in sd.items()
                if isinstance(v, torch.Tensor) and v.ndim == 2 and v.shape[1] >= old_obs
                and v.shape[1] - old_obs <= 16]
        if not hits:
            print("  %-26s no input layer matched, left alone" % sd_key)
            continue
        for k in hits:
            W = sd[k]
            new_W = expand(W, old_obs, new_obs, a.frame_old, a.frame_new,
                           a.insert_at, a.stacks)
            sd[k] = new_W
            print("  %-26s %-22s %s -> %s"
                  % (sd_key, k, tuple(W.shape), tuple(new_W.shape)))
            n_done += 1

    if not n_done:
        sys.exit("nothing was expanded. Check --frame-old/--stacks against the "
                 "checkpoint: print the shapes above and pick the layer whose "
                 "column count equals obs_dim.")

    # Adam moments are shaped to the OLD layer and are meaningless now. Dropping
    # them is the same reasoning as DOUBLEBEE_RESUME_FRESH_OPTIM: stale moment
    # estimates against a changed parameterisation destroyed a policy twice.
    for k in list(ck.keys()):
        if "optimizer" in k:
            del ck[k]
            print("  dropped %s (shapes no longer valid)" % k)
    ck["iter"] = 0          # spend a full training budget, not the leftover
    torch.save(ck, a.dst)
    print("\nwrote %s" % a.dst)
    print("VALIDATE: train 20 iterations and check Mean episode length. A good\n"
          "transplant starts near the OLD policy's length, not near 30. If it\n"
          "starts at ~30 the column mapping is wrong and this is worse than\n"
          "random init, so check before committing hours to it.")


if __name__ == "__main__":
    main()

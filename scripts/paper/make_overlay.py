"""Multi-exposure overlay of a climb, from a fixed-camera video.

The strip-of-frames figure spends five panels to show what one composite shows
better: the whole traversal, in one image, with the geometry of the staircase
visible once instead of five times.

METHOD. With a static camera the background is the per-pixel median over the
clip, which is robust to the robot passing through. Each selected frame is then
differenced against that background, thresholded, and the robot pasted onto the
composite. Later positions are drawn more opaquely so the eye reads the
direction of travel.

REQUIRES A TRIPOD. Any camera motion makes the whole frame differ from the
median and the mask becomes the entire image. If the result looks like a smear,
that is what happened.

    python3 make_overlay.py climb.mp4 -o fig_climb_overlay.png --n 6
    python3 make_overlay.py climb.mp4 --start 1.2 --end 4.0 --n 7 --thresh 30
"""
import argparse
import sys

import numpy as np

try:
    import cv2
except ImportError:
    sys.exit("need opencv: pip install opencv-python")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("video")
    p.add_argument("-o", "--out", default="fig_climb_overlay.png")
    p.add_argument("--n", type=int, default=6, help="robot positions to composite")
    p.add_argument("--start", type=float, default=0.0, help="seconds")
    p.add_argument("--end", type=float, default=-1.0, help="seconds, -1 = end")
    p.add_argument("--thresh", type=int, default=25,
                   help="difference threshold, 0-255. Raise if the background "
                        "bleeds in, lower if the robot comes out holey.")
    p.add_argument("--min-alpha", dest="min_alpha", type=float, default=0.35,
                   help="opacity of the earliest position; the last is always 1.0")
    p.add_argument("--bg-frames", dest="bg_frames", type=int, default=60,
                   help="frames sampled for the median background plate")
    a = p.parse_args()

    cap = cv2.VideoCapture(a.video)
    if not cap.isOpened():
        sys.exit("cannot open %s" % a.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("%s: %d frames @ %.1f fps (%.1f s)" % (a.video, total, fps, total / fps))

    i0 = int(a.start * fps)
    i1 = total - 1 if a.end < 0 else min(total - 1, int(a.end * fps))
    if i1 - i0 < a.n:
        sys.exit("window too short: %d frames for %d positions" % (i1 - i0, a.n))

    # background plate: per-pixel median over frames spread across the clip
    idx = np.linspace(0, total - 1, min(a.bg_frames, total)).astype(int)
    stack = []
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if ok:
            stack.append(f)
    if not stack:
        sys.exit("could not read frames")
    bg = np.median(np.stack(stack), axis=0).astype(np.uint8)
    print("background plate from %d frames" % len(stack))

    comp = bg.copy().astype(np.float32)
    picks = np.linspace(i0, i1, a.n).astype(int)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    used = 0
    for k, i in enumerate(picks):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if not ok:
            continue
        d = cv2.absdiff(f, bg).max(axis=2)
        m = (d > a.thresh).astype(np.uint8)
        # close holes, drop speckle, then feather so edges do not look cut out
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=2)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel, iterations=1)
        if m.sum() < 200:
            print("  frame %d: mask nearly empty, skipped (lower --thresh)" % i)
            continue
        mf = cv2.GaussianBlur(m.astype(np.float32), (7, 7), 0)
        mf = np.clip(mf, 0, 1)[..., None]
        alpha = a.min_alpha + (1.0 - a.min_alpha) * (k / max(a.n - 1, 1))
        comp = comp * (1 - mf * alpha) + f.astype(np.float32) * (mf * alpha)
        used += 1
    cap.release()

    if used < 2:
        sys.exit("only %d positions composited. Camera probably moved, or "
                 "--thresh is wrong for this footage." % used)
    cv2.imwrite(a.out, comp.astype(np.uint8))
    print("wrote %s  (%d positions, alpha %.2f -> 1.00)" % (a.out, used, a.min_alpha))
    print("\nIf the robot looks holey, lower --thresh. If the background bleeds\n"
          "in as ghosting, raise it. If the whole frame is smeared, the camera\n"
          "moved and no threshold will fix it.")


if __name__ == "__main__":
    main()

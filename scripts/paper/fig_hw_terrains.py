"""Hardware results: three unseen terrains, success on top, failure below.

Styled to match fig_gen_panels.py so the simulation and hardware figures read as
a pair: a Greek badge in the corner of each panel and nothing else. Everything
the badges stand for goes in the caption, which is what buys the space.

Images are CROPPED to the cell aspect, never stretched.
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.image as mpimg

# Badges continue from the generalisation figure, which used alpha/beta/gamma,
# now alpha/beta/gamma (sim generalisation figure removed). Only the SUCCESS row is badged; each failure
# sits directly under its own terrain, so the caption can refer to "delta and
# its failure mode" without needing a second set of symbols.
COLS = [("678",  r"$\alpha$", "Narrow"),
        ("ramp", r"$\beta$", "Ramp"),
        ("89",   r"$\gamma$", "Foam")]


def crop_to(arr, aspect, bias=0.5):
    """Crop to `aspect`, keeping the part of the frame that matters.

    bias is where the kept window sits vertically: 0.5 centres it, 1.0 keeps the
    BOTTOM. These photos put the wheels, the step, the tick and the circled
    detail low in the frame and the propellers high, so a centred crop deletes
    the annotations first. Bias down and the propeller tops go instead.
    """
    h, w = arr.shape[:2]
    if w / h > aspect:
        nw = int(round(h * aspect)); x0 = (w - nw) // 2
        return arr[:, x0:x0 + nw]
    nh = int(round(w / aspect))
    y0 = int(round((h - nh) * float(np.clip(bias, 0.0, 1.0))))
    return arr[y0:y0 + nh]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--img_dir", default="figs/res_figs")
    p.add_argument("-o", "--out", default="figs/fig_hw_terrains.pdf")
    p.add_argument("--width", type=float, default=3.45)
    p.add_argument("--height", type=float, default=3.30)
    p.add_argument("--wspace", type=float, default=0.025)
    p.add_argument("--hspace", type=float, default=0.030)
    p.add_argument("--badge", type=float, default=6.4)
    p.add_argument("--crop_bias", type=float, default=0.80,
                   help="0.5 centres the crop, 1.0 keeps the bottom of the frame")
    p.add_argument("--pad", type=float, default=0.30,
                   help="padding inside the badge box")
    p.add_argument("--name_size", type=float, default=1.0,
                   help="0 to drop the terrain names and keep bare letters")
    a = p.parse_args()

    fig = plt.figure(figsize=(a.width, a.height))
    gs = GridSpec(2, 3, figure=fig, wspace=a.wspace, hspace=a.hspace,
                  left=0.004, right=0.996, top=0.996, bottom=0.004)

    for r, tag in enumerate(("s", "f")):
        for c, (key, badge, name) in enumerate(COLS):
            ax = fig.add_subplot(gs[r, c])
            arr = mpimg.imread(os.path.join(a.img_dir, "%s_%s_anno.png" % (key, tag)))
            pos = ax.get_position()
            cell = (pos.width * a.width) / (pos.height * a.height)
            ax.imshow(crop_to(arr, cell, a.crop_bias), aspect="auto",
                      interpolation="lanczos")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_linewidth(0.4); s.set_color("0.4")
            if r == 0:
                # Label lives INSIDE the black box with the letter. White text
                # on the bare photo would fight the light floor and the wood.
                lab = badge if a.name_size <= 0 else ("%s %s" % (badge, name))
                ax.text(0.028, 0.965, lab, transform=ax.transAxes,
                        fontsize=a.badge, color="white", va="top", ha="left",
                        bbox=dict(boxstyle="round,pad=%.2f,rounding_size=0.10"
                                  % a.pad, fc="black", ec="none"))

    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    fig.savefig(a.out, dpi=400)
    print("wrote %s" % a.out)


main()

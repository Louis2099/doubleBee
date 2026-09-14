"""Training terrain with the policy's height scan as an inset.

One single-column figure replacing the prose description of the terrain and the
height-scan observation. The main image is a wide render of a training tile;
the inset is a close render of the robot at a step edge with the 16 ray hits
drawn (DOUBLEBEE_SCAN_VIS=1 in play.py). Styled like fig_hw_terrains.py and
fig_gen_panels.py: black badge, white text, thin grey spines, nothing else.

Images are CROPPED to their cell aspect, never stretched.

    python3 fig_terrain_scan.py --wide wide.png --close close.png \
        --wide_crop X Y W H --close_crop X Y W H -o figs/fig_terrain_scan.pdf
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def load_crop(path, box):
    arr = mpimg.imread(path)
    if box is None:
        return arr
    x, y, w, h = (int(v) for v in box)
    return arr[y:y + h, x:x + w]


def fit_aspect(arr, aspect):
    """Centre-crop arr to width/height = aspect."""
    h, w = arr.shape[:2]
    if w / h > aspect:
        nw = int(round(h * aspect)); x0 = (w - nw) // 2
        return arr[:, x0:x0 + nw]
    nh = int(round(w / aspect)); y0 = (h - nh) // 2
    return arr[y0:y0 + nh]


def badge(ax, text, x, y, size, ha="left", va="top"):
    ax.text(x, y, text, transform=ax.transAxes, fontsize=size, color="white",
            ha=ha, va=va,
            bbox=dict(boxstyle="round,pad=0.30,rounding_size=0.10",
                      fc="black", ec="none"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wide", required=True)
    p.add_argument("--close", required=True)
    p.add_argument("--wide_crop", type=float, nargs=4, default=None,
                   metavar=("X", "Y", "W", "H"), help="pixel box in the wide render")
    p.add_argument("--close_crop", type=float, nargs=4, default=None,
                   metavar=("X", "Y", "W", "H"), help="pixel box in the close render")
    p.add_argument("-o", "--out", default="figs/fig_terrain_scan.pdf")
    p.add_argument("--width", type=float, default=3.45, help="inches, IEEE column")
    p.add_argument("--aspect", type=float, default=1.55, help="main image w/h")
    # inset placement in axes fractions of the main image: left, bottom, width
    p.add_argument("--inset", type=float, nargs=3, default=[0.56, 0.04, 0.42],
                   metavar=("L", "B", "W"))
    p.add_argument("--inset_aspect", type=float, default=1.25)
    # optional zoom box on the main image, in axes fractions: left bottom w h
    p.add_argument("--zoom_box", type=float, nargs=4, default=None,
                   metavar=("L", "B", "W", "H"))
    p.add_argument("--badge", type=float, default=6.4)
    p.add_argument("--labels", nargs=2, default=["Training terrain", "Height scan"])
    p.add_argument("--dpi", type=int, default=600)
    a = p.parse_args()

    height = a.width / a.aspect
    fig = plt.figure(figsize=(a.width, height))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(fit_aspect(load_crop(a.wide, a.wide_crop), a.aspect),
              aspect="auto", interpolation="lanczos")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.4); s.set_color("0.4")
    badge(ax, a.labels[0], 0.02, 0.97, a.badge)

    L, B, W = a.inset
    H = W * a.aspect / a.inset_aspect          # keep the inset's own aspect
    ins = ax.inset_axes([L, B, W, H])
    ins.imshow(fit_aspect(load_crop(a.close, a.close_crop), a.inset_aspect),
               aspect="auto", interpolation="lanczos")
    ins.set_xticks([]); ins.set_yticks([])
    for s in ins.spines.values():
        s.set_linewidth(0.9); s.set_color("white")
    badge(ins, a.labels[1], 0.04, 0.95, a.badge * 0.9)

    if a.zoom_box is not None:
        zl, zb, zw, zh = a.zoom_box
        ax.add_patch(plt.Rectangle((zl, zb), zw, zh, transform=ax.transAxes,
                                   fill=False, ec="white", lw=0.8))
        # two connectors: zoom box right edge to the inset's left edge, so the
        # pair reads as a magnifier rather than a pointer
        for zy, iy in ((zb + zh, B + H), (zb, B)):
            ax.plot([zl + zw, L], [zy, iy], transform=ax.transAxes,
                    color="white", lw=0.5, alpha=0.9)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    fig.savefig(a.out, dpi=a.dpi)
    fig.savefig(os.path.splitext(a.out)[0] + ".png", dpi=250)
    print("wrote %s  (%.2f x %.2f in)" % (a.out, a.width, height))


main()
